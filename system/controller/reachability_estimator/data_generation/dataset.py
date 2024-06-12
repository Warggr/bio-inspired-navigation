''' This code has been adapted from:
***************************************************************************************
*    Title: "Scaling Local Control to Large Scale Topological Navigation"
*    Author: "Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox"
*    Date: 2020
*    Availability: https://github.com/xymeng/rmp_nav
*
***************************************************************************************
'''
import h5py
import torch.utils.data as data
import numpy as np
import itertools
import bisect
import random
import matplotlib.pyplot as plt
from typing import Tuple, Iterator, Dict, Optional, Protocol, Callable

import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from system.controller.reachability_estimator.reachability_utils import ViewOverlapReachabilityController

from system.controller.simulation.pybullet_environment import PybulletEnvironment, all_possible_textures
from system.controller.simulation.environment_config import environment_dimensions
from system.controller.simulation.environment_cache import EnvironmentCache
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.controller.simulation.environment.map_occupancy_helpers.map_utils import path_length
from system.plotting.plotResults import plotStartGoalDataset
from system.types import types, FlatSpikings, WaypointInfo, Vector2D
from system.controller.reachability_estimator.types import Sample, PlaceInfo, ReachabilityController

def get_path():
    """ returns path to data storage folder """
    dirname = os.path.join(os.path.dirname(__file__), "..")
    return dirname


plotting = os.getenv('PLOTTING', False)  # if True: plot every reachability rollout
debug = os.getenv('DEBUG', False)  # if True: print debug output


def print_debug(*params):
    """ output only when in debug mode """
    if debug:
        print(*params)


def spikings_reshape(img_array : FlatSpikings) -> types.Spikings:
    """ image stored in array form to image in correct shape for nn """
    img = np.reshape(img_array, (6, 1600))
    return img

def place_info(
    data: WaypointInfo,
    env: PybulletEnvironment,
) -> PlaceInfo:
    pos, angle, spikings = data
    spikings = spikings_reshape(spikings)
    img = env.camera([pos, angle])
    lidar, _ = env.lidar([pos, angle])
    return PlaceInfo(pos, angle, spikings, img, lidar)

class SampleGenerator(Protocol, Iterator[Tuple[Sample, float, str]]):
    def env_model(self) -> str:
        ...
    def __next__(self) -> Tuple[Sample, float, str]:
        ...

class TrajectoriesDataset(data.Dataset):
    '''
    Generate data for the reachability estimator training.

    arguments:
    hd5_files           -- hd5_files containing trajectories used to generate reachability data
                            format: attributes: agent, map; data per timestep: xy_coordinates, orientation, grid cell spiking
                            see gen_trajectories.py for details
    env_kwargs          -- keyword arguments to pass to the PybulletEnvironment constructor

    distance_min/max    --  min/max distance between goal and start
    range_min/max       --  min/max timesteps between goal and start
    frame_interval      --  number of timesteps between frames

    For details see original source code: https://github.com/xymeng/rmp_nav
    '''

    def __init__(self, hd5_files, env_cache : EnvironmentCache):
        self.hd5_files = sorted(list(hd5_files))

        # open hd5 files
        maps = []
        fds = []
        for fn in self.hd5_files:
            try:
                fds.append(h5py.File(fn, 'r'))
                maps.append(fds[-1].attrs["map_type"])
            except:
                print('unable to open', fn)
                raise

        def flatten(ll):
            return list(itertools.chain.from_iterable(ll))

        # A list of tuples (dataset_idx, trajectory_id)
        self.traj_ids = flatten(
            zip(itertools.repeat(i),
                list(fds[i].keys())[0:len(fds[i])])
            for i in range(len(fds)))
        print_debug('total trajectories:', len(self.traj_ids))

        # Map (dataset_idx, traj_id) to trajectory length
        self.traj_len_dict = {(i, tid): fds[i][tid].shape[0] for i, tid in self.traj_ids}
        self.traj_len_cumsum = np.cumsum([self.traj_len_dict[_] for _ in self.traj_ids])

        def maybe_decode(s):
            # This is to deal with the breaking change in how h5py 3.0 deals with
            # strings.
            if type(s) == str:
                return s
            else:
                return s.decode('ascii')

        # Map (dataset_idx, traj_id) to its corresponding map
        traj_id_map = {(dset_idx, traj_id): maybe_decode(fds[dset_idx].attrs['map_type'])
                       for dset_idx, traj_id in self.traj_ids}

        self.map_names = maps
        map_name_set = set(maps)
        self.layout = MapLayout(self.map_names[0])

        traj_ids_per_map = {_: [] for _ in self.map_names}
        for dset_idx, traj_id in self.traj_ids:
            map_name = traj_id_map[(dset_idx, traj_id)]
            if map_name in map_name_set:
                traj_ids_per_map[map_name].append((dset_idx, traj_id))
        self.traj_ids_per_map = traj_ids_per_map

        self.samples_per_map = {
            map_name: sum([self.traj_len_dict[_] for _ in traj_ids])
            for map_name, traj_ids in traj_ids_per_map.items()}
        print_debug(self.samples_per_map)

        # Map a map name to cumulative sum of trajectory lengths.
        self.traj_len_cumsum_per_map = {
            # Note than when computing cumsum we must ensure the ordering. Hence we must
            # not use .values().
            map_name: np.cumsum([self.traj_len_dict[_] for _ in traj_ids])
            for map_name, traj_ids in traj_ids_per_map.items()}

        self.envs = env_cache
        for map_name in self.map_names:
            try:
                self.envs.load(map_name)
            except KeyError: pass

        self.opened = False
        self.load_to_mem = False
        self.first = True

    def _init_once(self, seed):
        # Should be called after the dataset runs in a separate process
        if self.first:
            self._open_datasets()
            self.rng = np.random.RandomState(12345 + (seed % 1000000) * 666)
            self.first = False
            print("Init finished")

    # open the dataset
    def _open_datasets(self):
        if not self.opened:
            driver = None
            if self.load_to_mem:
                driver = 'core'
                print('loading dataset into memory... it may take a while')
            self.fds = [h5py.File(fn, 'r', driver=driver)
                        for fn in self.hd5_files]
            self.opened = True

    def _locate_sample(self, idx):
        traj_idx = bisect.bisect_right(self.traj_len_cumsum, idx)
        dataset_idx, traj_id = self.traj_ids[traj_idx]

        if traj_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.traj_len_cumsum[traj_idx - 1]

        return dataset_idx, traj_id, sample_idx

    def _locate_sample_single_map(self, idx, map_name):
        """
        Similar to _locate_sample(), but only considers a single map.
        :param idx: sample index in the range of [0, total number of samples of this map - 1]
        """
        cumsum = self.traj_len_cumsum_per_map[map_name]
        assert 0 <= idx < cumsum[-1], 'Map index %d out of range [0, %d)' % (idx, cumsum[-1])

        trajs = self.traj_ids_per_map[map_name]

        traj_idx = bisect.bisect_right(cumsum, idx)
        dataset_idx, traj_id = trajs[traj_idx]

        if traj_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - cumsum[traj_idx - 1]

        return dataset_idx, traj_id, sample_idx

    def __len__(self):
        return self.traj_len_cumsum[-1]

    def _draw_sample_same_traj(self, idx) -> Optional[Tuple[str, WaypointInfo, WaypointInfo, float]]:
        """ Draw a source and goal sample from the same trajectory.
            Their distance will be between distance_min and distance_max.
            They will be separated by timesteps in range of range_min to range_max.

        returns:
        name of map, source sample, destination sample
        """

        # Get a good pair of samples
        src_dataset_idx, src_traj_id, src_idx = self._locate_sample(idx)

        src_traj = self.fds[src_dataset_idx][src_traj_id]  # todo self.map_names[0] like in layout?
        map_name = self.fds[src_dataset_idx].attrs['map_type']

        src_sample = src_traj[src_idx]
        src_pos = src_sample[0]

        p = self.rng.uniform()
        x = 0
        while True:
            x += 1
            if p <= 0.75:
                dst_idx = max(0, min(len(src_traj) - 1, src_idx + self.rng.randint(-10, 10)))
            else:
                dst_idx = self.rng.randint(0, len(src_traj))
            if dst_idx != src_idx:
                break
            if x >= 20:
                return None

        dst_sample = src_traj[dst_idx]
        dst_pos = dst_sample[0]

        # return path_length
        goal_pos = list(dst_pos)
        src_pos = list(src_pos)
        waypoints = self.layout.find_path(src_pos, goal_pos)
        if waypoints is None:
            return None
        path_l = path_length(waypoints)

        return map_name, src_sample, dst_sample, path_l

    def _draw_sample_same_traj_multiple_tries(self, idx) -> (str, WaypointInfo, WaypointInfo, float):
        while True:
            result = self._draw_sample_same_traj(idx)
            if result is not None:
                return result
            else:
                idx = (idx + self.rng.randint(1000)) % len(self)

    def _draw_sample_diff_traj(self, idx) -> (str, WaypointInfo, WaypointInfo, float):
        """ Draw a source and goal sample from two different trajectories on the same map.
        
        returns:
        name of map, source sample, destination sample, length of path between start and goal
        """

        while True:
            # Get a good pair of samples
            src_dataset_idx, src_traj_id, src_idx = self._locate_sample(idx)

            src_traj = self.fds[src_dataset_idx][src_traj_id]
            map_name = self.fds[src_dataset_idx].attrs['map_type']

            idx2 = self.rng.randint(self.samples_per_map[map_name])
            dst_dataset_idx, dst_traj_id, dst_idx = self._locate_sample_single_map(idx2, map_name)
            dst_traj = self.fds[dst_dataset_idx][dst_traj_id]

            src_sample = src_traj[src_idx]
            dst_sample = dst_traj[dst_idx]

            # return path_length
            waypoints = self.layout.find_path(src_sample[0], dst_sample[0])
            # no path found between source and goal -> skip this sample
            if not waypoints:
                print_debug("No path found.")
                continue
            path_l = path_length(waypoints)
            return map_name, src_sample, dst_sample, path_l

    def _draw_sample(self, idx, diff_traj_prob = 0.1) -> (str, WaypointInfo, WaypointInfo, float):
        # choose with probability p from same/different trajectory
        p = self.rng.uniform(0.0, 1.0)

        if p < diff_traj_prob:
            return self._draw_sample_diff_traj(idx)
        else:
            return self._draw_sample_same_traj_multiple_tries(idx)

    class _Iterator:
        def __init__(self, parent : 'TrajectoriesDataset', parent_function : Callable[int, Tuple[str, WaypointInfo, WaypointInfo, float]]):
            self.parent = parent
            self.parent_function = parent_function
            # parent_function is a method of parent which can return samples
        def __next__(self) -> Tuple[Sample, float, str]:
            random_idx = random.randint(0, len(self.parent))
            self.parent._init_once(random_idx)
            pair = self.parent_function(idx=random_idx)
            map_name, src_sample, dst_sample, path_l = pair

            env = self.parent.envs[map_name]
            return Sample(place_info(src_sample, env), place_info(dst_sample, env)), path_l, map_name
        def env_model(self):
            return self.parent.map_names[0]

    def iterate(self, mode='same_traj') -> SampleGenerator:
        functions = {
            'same_traj': self._draw_sample_same_traj_multiple_tries,
            'diff_traj': self._draw_sample_diff_traj,
            'maybe_diff_traj': self._draw_sample,
        }
        return self._Iterator(self, parent_function=functions[mode])


DATASET_KEY = 'positions'

def assert_conforms_to_type(data, dtype):
    for dtyp, field in zip(dtype.fields.values(), data):
        dtyp = dtyp[0]
        if dtyp.char == '?':
            assert isinstance(field, np.bool_) or isinstance(field, bool) # TODO: surely there's a more concise way
        elif dtyp == np.dtype('float32'):
            assert isinstance(field, np.float32) or isinstance(field, float), f"{field} != np.float32"
        else:
            assert dtyp.shape[0] == len(field), f"{dtyp.shape} != {len(field)}"

from system.controller.local_controller.local_navigation import setup_gc_network

def random_coordinates(xmin, xmax, ymin, ymax):
    return np.array([ random.uniform(xmin, ymin), random.uniform(ymin, ymax) ])

def in_rect(x : Vector2D, rect):
    xmin, xmax, ymin, ymax = rect
    return x[0] >= xmin and x[1] >= ymin and \
        x[0] <= xmax and x[1] <= ymax

class RandomSamples:
    def __init__(self, env : PybulletEnvironment):
        self.env = env
        self.map = MapLayout(env.env_model)
    def env_model(self):
        return self.env.env_model

    def points(self) -> Tuple[Vector2D, Vector2D]:
        result = []
        for i in range(2):
            i = 0
            while True:
                i += 1
                p = random_coordinates(*environment_dimensions(self.env.env_model))
                print(f"[i={i}]Trying", p, "...")
                if self.map.suitable_position_for_robot(p):
                    break
            result.append(p)
        return result

    def points_to_sample(self, p1 : Vector2D, p2 : Vector2D) -> Tuple[Sample, float, str]:
        waypoints = self.map.find_path(p1, p2)
        assert waypoints is not None
        path_l = path_length(waypoints)

        dt = 1
        # TODO Pierre: does that actually work?
        gc_network = setup_gc_network(dt)
        start_spikings : FlatSpikings = gc_network.consolidate_gc_spiking().flatten()
        gc_network.track_movement(xy_speed=(p2 - p1)/dt)
        goal_spikings : FlatSpikings = gc_network.consolidate_gc_spiking().flatten()

        a1, a2 = [ random.uniform(-np.pi, np.pi) for _ in range(2) ]
        return Sample(place_info((p1, a1, start_spikings), self.env), place_info((p2, a2, goal_spikings), self.env)), path_l, self.map.name

    def sample(self) -> Tuple[Sample, float, str]:
        return self.points_to_sample(*self.points())

    def __next__(self):
        return self.sample()

class RandomSamplesWithLimitedDistance(RandomSamples):
    def __init__(self, *args, r_max = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_max = r_max

    def points(self) -> Tuple[Vector2D, Vector2D]:
        i = 0
        dims = environment_dimensions(self.env.env_model)
        while True:
            i += 1
            p1 = random_coordinates(*dims)
            if not self.map.suitable_position_for_robot(p1):
                continue
            for j in range(100):
                r = np.sqrt(random.uniform(0, self.r_max**2)) # sqrt ensures that the samples are drawn uniformly from the circle
                theta = random.uniform(-np.pi, np.pi)
                distance = np.array([ np.cos(theta), np.sin(theta) ]) * r
                p2 = p1 + distance
                if (not in_rect(p2, dims)) or (not self.map.suitable_position_for_robot(p2)):
                    continue
                else:
                    return p1, p2

def create_and_save_reachability_samples(
    samples : SampleGenerator, f : h5py.File,
    envs : EnvironmentCache,
    reachability_controller : ReachabilityController = ViewOverlapReachabilityController(),
    nr_samples=1000,
    flush_freq=50,
) -> h5py.File:
    """ Create reachability samples.

    arguments:
    rd : dataset to draw positions from
    """

    env_model = samples.env_model()
    print_debug("env_model: ", env_model)
    f.attrs.create('map_type', env_model)

    dtype = Sample.dtype

    try:
        dset = f[DATASET_KEY]
        old_size = dset.size
        start_index = old_size

        if old_size < nr_samples:
            # Hint: this might fail if somehow the dtype changed from one dataset to the other
            dset = f.create_dataset('tmp', dtype=dtype, data=dset[:], maxshape=(nr_samples,))
            del f[DATASET_KEY]
            f.move('tmp', DATASET_KEY)
    except KeyError:
        dset = f.create_dataset(DATASET_KEY, data=np.array([], dtype=dtype), dtype=dtype, maxshape=(nr_samples,))
        start_index = 0

    from tqdm import tqdm

    sum_r = 0
    buffer = []

    for i in (bar := tqdm(range(start_index, nr_samples), initial=start_index, total=nr_samples)):
        reachable = None
        while reachable is None:
            try:
                sample, path_l, map_name = next(samples)
            except ValueError:
                tqdm.write(f"Error at sample {i}")
                continue

            env = envs[map_name]

            if True: #try:
                reachable = reachability_controller.reachable(env, sample.src, sample.dst, path_l)
            #except AssertionError:
            #    continue

        sum_r += int(reachable)

        buffer.append(sample.to_tuple(reachable))
        if (i+1) % flush_freq == 0 or i+1 == nr_samples:
            tqdm.write(f'Flushing at {i+1}')
            old_size = dset.size
            try:
                dset.resize((i+1,))
                assert_conforms_to_type(buffer[0], dtype)
                dset[old_size:] = np.asarray(buffer, dtype=dtype)
            except Exception: # roll back if anything happens - else the dataset will get saved with a bunch of empty fields
                dset.resize((old_size,))
                raise
            f.flush()
            bar.set_description(f"percentage reachable: {sum_r / (i-start_index+1)}")
            buffer = []
    return f


def display_samples(hf : h5py.File, imageplot=False):
    """ Display information about dataset file
    
    if imageplot: plot the stored images
    Calculate the percentage of reached/failed samples.
    """

    env_model = hf.attrs["map_type"]
    dataset = hf[DATASET_KEY]

    print(f"Number of samples:", dataset.size)
    reached = 0
    count = dataset.size
    reach = []
    starts_goals = []
    for i, datapoint in enumerate(dataset):
        if i % 1000 == 0:
            print("At sample number", i)
        if imageplot and i > 5000:
            break

        if imageplot and i < 5:
            img = np.reshape(datapoint[0], (64, 64, 4))
            imgplot = plt.imshow(img)
            plt.show()

            img = np.reshape(datapoint[1], (64, 64, 4))
            imgplot = plt.imshow(img)
            plt.show()
        if datapoint[2] == 1.0:
            reached += 1
        starts_goals.append((datapoint[3], datapoint[4]))
        reach.append(datapoint[2])
    print("overall", count)
    print("reached", reached)
    print("failed", count - reached)
    print("percentage reached/all", reached / count)
    if imageplot:
        plotStartGoalDataset(env_model, starts_goals)


if __name__ == "__main__":
    """ Generate a dataset of reachability samples or load from an existing one.

    Testing:
    Generate or load and display a few samples.

    Parameterized call:
        Save to/load from filename
        Generate until there are num_samples samples
        Use trajectories from traj_file

    To time the creation of 50 samples, run `time python dataset.py reachability_fifty_samples trajectories_Savinov3.hd5 -n 50 --no-image-plot`
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('basename', help='Base file name of generated dataset file', nargs='?', default='dataset')
    parser.add_argument('traj_file', help='Dataset of trajectories', nargs='?', default='trajectories.hd5')
    parser.add_argument('-n', '--num-samples', type=int, dest='num_samples', default=200000)
    parser.add_argument('--flush-freq', type=int, dest='flush_freq', default=1000)
    parser.add_argument('--extension', help='extension of the dataset file', default='.hd5')
    parser.add_argument('--image-plot', action=argparse.BooleanOptionalAction, help='Show image of samples taken')
    parser.add_argument('-w', '--wall-colors', help='how to color the walls', choices=['1color', '3colors', 'patterns'], default='1color')
    parser.add_argument('--re',
        choices=['view_overlap', 'network', 'distance', 'simulation'], default='view_overlap',
        help='The reachability estimator to generate ground truth reachability values',
    )
    parser.add_argument('--gen', '--generate-point-pairs',
        choices=['same_traj', 'diff_traj', 'maybe_diff_traj', 'random', 'random_circle'], default='same_traj',
        help='How to generate pairs of points',
    )
    args = parser.parse_args()

    if args.wall_colors == '1color':
        textures = [ os.path.join( 'yellow_wall.png') ]
    elif args.wall_colors == '3colors':
        textures = all_possible_textures[:args.wall_colors]
    elif args.wall_colors == 'patterns':
        textures = lambda i : f'pattern-{i+1}.png'

    # Input file
    filename = os.path.join(get_path(), "data", "trajectories", args.traj_file)
    filename = os.path.realpath(filename)
    re = ReachabilityController.factory(controller_type=args.re)

    suffix = '' if args.wall_colors == '1color' else f'-{args.wall_colors}'

    # Output file
    filename = os.path.join(get_path(), "data", "reachability", args.basename + suffix + args.extension)
    filename = os.path.realpath(filename)
    f = h5py.File(filename, 'a')

    env_kwargs={ 'wall_kwargs': { 'textures': textures } }
    with EnvironmentCache(override_env_kwargs=env_kwargs) as env_cache:
        if args.gen.endswith('_traj'):
            rd = TrajectoriesDataset([filename], env_cache=env_cache)
            samples = rd.iterate(mode=args.gen)
        else:
            if args.gen == 'random':
                samples = RandomSamples(env_cache["Savinov_val3"])
            elif args.gen == 'random_circle':
                samples = RandomSamplesWithLimitedDistance(env_cache["Savinov_val3"])
            else: raise ValueError(args.gen)

        create_and_save_reachability_samples(
            samples, f,
            envs=env_cache,
            reachability_controller=re,
            nr_samples=args.num_samples,
            flush_freq=args.flush_freq,
        )
        if args.image_plot:
            print("Finished creating samples. Now displaying them")
            display_samples(f, imageplot=True)
