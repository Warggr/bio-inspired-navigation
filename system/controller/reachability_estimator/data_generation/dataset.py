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
from typing import Tuple, Generator, Dict

import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from system.controller.reachability_estimator.reachability_utils import ViewOverlapReachabilityController

from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.controller.simulation.environment.map_occupancy_helpers.map_utils import path_length
from system.plotting.plotResults import plotStartGoalDataset
from system.controller.simulation.pybullet_environment import types

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


class Sample:
    envs : Dict[str, PybulletEnvironment] = {}

    def __init__(
        self,
        src: Tuple[types.Vector2D, types.Angle, types.Spikings],
        dst: Tuple[types.Vector2D, types.Angle, types.Spikings],
        map_name: str,
    ):
        self.src_pos, self.src_angle, self.src_spikings = src
        self.dst_pos, self.dst_angle, self.dst_spikings = dst

        try:
            env = Sample.envs[map_name]
        except KeyError:
            env = PybulletEnvironment(map_name, mode="analytical", build_data_set=True, contains_robot=False)
            Sample.envs[map_name] = env

        self.src_img = env.camera([self.src_pos, self.src_angle])
        self.dst_img = env.camera([self.dst_pos, self.dst_angle])
        # TODO: src_img.flatten(), dst_img.flatten()
        self.src_distances = env.lidar([self.src_pos, self.src_angle]).distances
        self.dst_distances = env.lidar([self.dst_pos, self.dst_angle]).distances


class SampleConfig:
    def __init__(self,
        grid_cell_spikings=False,
        lidar=False,
    ):
        self.with_grid_cell_spikings = grid_cell_spikings
        self.with_lidar = lidar
    def dtype(self):
        fields = [
            ('start_observation', (np.int32, 16384)), # 64 x 64 x 4
            ('goal_observation', (np.int32, 16384)), # using (64, 64, 4) would be more elegant but H5py doesn't accept it
            ('reached', bool),
            ('start', (np.float32, 2)),  # x, y
            ('goal', (np.float32, 2)),  # x, y
            ('start_orientation', np.float32),  # theta
            ('goal_orientation', np.float32)  # theta
        ]
        if self.with_grid_cell_spikings:
            fields += [
                ('start_spikings', (np.float32, 9600)),  # 40 * 40 * 6
                ('goal_spikings', (np.float32, 9600))  # 40 * 40 * 6
            ]
        if self.with_lidar:
            NUMBER_OF_WHISKERS = 52
            fields += [
                ('start_boundary_spikings', (np.float32, NUMBER_OF_WHISKERS)),
                ('goal_boundary_spikings', (np.float32, NUMBER_OF_WHISKERS)),
            ]
        dtype = np.dtype(fields)
        return dtype
    def suffix(self) -> str:
        return (''
            + ('+spikings' if self.with_grid_cell_spikings else '')
            + ('+lidar' if self.with_lidar else '')
        )

    def to_tuple(self, sample : Sample, reachable : bool) -> Tuple:
        """ Returns a tuple which can be put into a Numpy array of type self.dtype() """
        tup = [
            sample.src_img.flatten(), sample.dst_img.flatten(),
            reachable,
            sample.src_pos, sample.dst_pos,
            sample.src_angle, sample.dst_angle,
        ]
        if self.with_grid_cell_spikings:
            tup += [ sample.src_spikings, sample.dst_spikings ]
        if self.with_lidar:
            tup += [ sample.src_boundary_cell_spikings, sample.dst_boundary_cell_spikings ]
        return tuple(tup)


class ReachabilityDataset(data.Dataset):
    '''
    Generate data for the reachability estimator training.

    arguments:
    hd5_files           -- hd5_files containing trajectories used to generate reachability data
                            format: attributes: agent, map; data per timestep: xy_coordinates, orientation, grid cell spiking
                            see gen_trajectories.py for details

    distance_min/max    --  min/max distance between goal and start
    range_min/max       --  min/max timesteps between goal and start
    frame_interval      --  number of timesteps between frames

    For details see original source code: https://github.com/xymeng/rmp_nav
    '''

    def __init__(self, hd5_files):
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

        self.opened = False
        self.load_to_mem = False
        self.first = True
        self.view_overlap_reachability_controller = ViewOverlapReachabilityController(self.layout)

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

    def _draw_sample_same_traj(self, idx) -> (str, Tuple[types.Vector2D, types.Angle, types.Spikings], Tuple[types.Vector2D, types.Angle, types.Spikings], float):
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

    def _draw_sample_diff_traj(self, idx):
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

    def __getitem__(self, idx) -> Tuple[Sample, bool]:
        ''' Loads or creates a sample. Sample contains ... 

        returns:

        sample containing start and end points
        reachability
        '''
        self._init_once(idx)

        # choose with probability p from same/different trajectory
        # p = self.rng.uniform(0.0, 1.0)

        # self.sample_diff_traj_prob = 0.1
        # if p < self.sample_diff_traj_prob:
        #     map_name, src_sample, dst_sample, path_l = self._draw_sample_diff_traj(idx)
        # else:
        pair = self._draw_sample_same_traj(idx)
        while pair is None:
            idx = (idx + self.rng.randint(1000)) % len(self)
            pair = self._draw_sample_same_traj(idx)
        map_name, src_sample, dst_sample, path_l = pair

        sample = Sample(src_sample, dst_sample, map_name=map_name)

        #print(f"Computing reachability for {sample.src_pos}, {sample.dst_pos}")
        try:
            r = self.view_overlap_reachability_controller.reachable(map_name, src_sample, dst_sample, path_l, sample.src_img, sample.dst_img)
        except ValueError:
            return None
        #print(f"Reachability computed {r}")

        return sample, r


DATASET_KEY = 'positions'

def create_and_save_reachability_samples(
    rd : ReachabilityDataset, f : h5py.File,
    nr_samples=1000,
    flush_freq=50,
    config = SampleConfig(),
) -> h5py.File:
    """ Create reachability samples.

    arguments:
    rd : dataset to draw positions from
    """

    env_model = rd.map_names[0]
    print_debug("env_model: ", env_model)
    f.attrs.create('map_type', env_model)

    dtype = config.dtype()

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
        item = None
        while item is None:
            random_index = random.randrange(rd.traj_len_cumsum[-1])
            item = rd[random_index]

        sample, reachable = item
        sum_r += int(reachable)

        buffer.append(config.to_tuple(sample, reachable))
        if (i+1) % flush_freq == 0 or i+1 == nr_samples:
            tqdm.write(f'Flushing at {i+1}')
            old_size = dset.size
            dset.resize((i+1,))
            print(f"Flush from {old_size} until {i+1}")
            dset[old_size:] = np.array(buffer, dtype=dtype)
            f.flush()
            bar.set_description(f"percentage reachable: {sum_r / (i+1)}")
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

    Default: time the generation of 50 samples
    """
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action')
    time_parser = subparsers.add_parser('time')
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('basename', help='Base file name of generated dataset file', nargs='?', default='dataset')
    time_parser.add_argument('basename', help='Base file name of generated dataset file', nargs='?', default='reachability_fifty_samples')
    test_parser.add_argument('traj_file', help='Dataset of trajectories', nargs='?', default='trajectories.hd5')
    time_parser.add_argument('traj_file', help='Dataset of trajectories', nargs='?', default='trajectories_Savinov3.hd5')
    test_parser.add_argument('-n', '--num-samples', type=int, dest='num_samples', default=200000)
    time_parser.add_argument('-n', '--num-samples', type=int, dest='num_samples', default=50)
    test_parser.add_argument('--flush-freq', type=int, dest='flush_freq', default=1000)
    test_parser.add_argument('--spikings', help='Grid cell spikings are included in the dataset', action='store_true')
    test_parser.add_argument('--lidar', help='LIDAR distances are included in the dataset', action='store_true')
    test_parser.add_argument('--extension', help='extension of the dataset file', default='.hd5')
    test_parser.add_argument('--image-plot', help='Show image of samples taken', action='store_true')

    args = parser.parse_args()
    if args.action == 'time':
        import time

        start = time.time()
        create_and_save_reachability_samples(args.basename, args.num_samples, args.traj_file)
        end = time.time()
        print("Time elapsed:", end - start)

    elif args.action == 'test':
        # Input file
        filename = os.path.join(get_path(), "data", "trajectories", args.traj_file)
        filename = os.path.realpath(filename)
        rd = ReachabilityDataset([filename])

        config = SampleConfig(grid_cell_spikings=args.spikings, lidar=args.lidar)

        # Output file
        dataset_name = args.basename + config.suffix()

        filename = os.path.join(get_path(), "data", "reachability", dataset_name + args.extension)
        filename = os.path.realpath(filename)
        f = h5py.File(filename, 'a')

        create_and_save_reachability_samples(
            rd, f,
            nr_samples=args.num_samples,
            flush_freq=args.flush_freq,
            config=config,
        )
        print("Finished creating samples. Now displaying them")
        display_samples(f, imageplot=True)
        # create_and_save_reachability_samples("test2", 1, "test_2.hd5")
        # display_samples("test2.hd5")
        # create_and_save_reachability_samples("test3", 1, "test_3.hd5")
        # display_samples("test3.hd5")
