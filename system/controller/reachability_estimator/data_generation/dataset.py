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

import sys
import os

from system.controller.reachability_estimator.reachability_utils import ViewOverlapReachabilityController

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.append("/home/tp2/Documents/Anna/bio-inspired-navigation/range_libc/pywrapper/common_msgs/geometry_msgs/")

from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.controller.simulation.environment.map_occupancy_helpers.map_utils import path_length
from system.plotting.plotResults import plotStartGoalDataset


def get_path():
    """ returns path to data storage folder """
    dirname = os.path.join(os.path.dirname(__file__), "..")
    return dirname


plotting = False  # if True: plot every reachability rollout
debug = False  # if True: print debug output


def print_debug(*params):
    """ output only when in debug mode """
    if debug:
        print(*params)


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

    def _draw_sample_same_traj(self, idx):
        """ Draw a source and goal sample from the same trajectory.
            Their distance will be between distance_min and distance_max.
            They will be seperated by timesteps in range of range_min to range_max.
        
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

    def get_camera_view(self, map_name, sample):
        # return img of position
        dt = 1e-2
        env = PybulletEnvironment(map_name, dt, "analytical", build_data_set=True)
        img = env.camera((sample[0], sample[1]))
        env.end_simulation()
        return img

    def __getitem__(self, idx):
        ''' Loads or creates a sample. Sample contains ... 
        
        returns:
        
        src_image
        dst_image
        reachability
        start and goal position
        first decoded goal vector
        src img after turn
        distance between goal and agent
        src grid cell spikings
        dst grid cell spikings'''
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

        src_img = self.get_camera_view(map_name, src_sample)[2]
        dst_img = self.get_camera_view(map_name, dst_sample)[2]

        print(f"Computing reachability for {(src_sample[0], src_sample[1])}, {(dst_sample[0], dst_sample[1])}")
        try:
            r = 1.0 if self.view_overlap_reachability_controller.reachable(map_name, src_sample, dst_sample, path_l, src_img, dst_img) else 0.0
        except ValueError:
            return None
        print(f"Reachability computed {r}")

        # image transformation
        return src_img.flatten(), dst_img.flatten(), r, [src_sample[0], dst_sample[0]], [src_sample[1], dst_sample[1]], src_sample[2], dst_sample[2]


def create_and_save_reachability_samples(filename, nr_samples, traj_file, with_grid_cell_spikings=False):
    """ Create reachability samples.
    
    arguments:
    filename    -- save the data in filename
    nr_samples  -- the number of samples that will be generated
    traj_file   -- name of the file used for trajectories
    """
    dirname = get_path()
    dirname = os.path.join(dirname, "data/reachability")
    directory = os.path.join(dirname, filename)
    filepath = os.path.realpath(directory)
    f = h5py.File(filepath + ".hd5", 'a')

    # Trajectories
    dirname = get_path()
    dirname = os.path.join(dirname, "data/trajectories")
    directory = os.path.join(dirname, traj_file)
    filename = os.path.realpath(directory)

    rd = ReachabilityDataset([filename])

    env_model = rd.map_names[0]
    print_debug("env_model: ", env_model)
    f.attrs.create('map_type', env_model)

    if with_grid_cell_spikings:
        dtype = np.dtype([
            ('start_observation', (np.int32, 16384)),  # 64 * 64 * 4
            ('goal_observation', (np.int32, 16384)),
            ('reached', np.float32),
            ('start', (np.float32, 2)),  # x, y
            ('goal', (np.float32, 2)),  # x, y
            ('start_orientation', np.float32),  # theta
            ('goal_orientation', np.float32),  # theta
            ('start_spikings', (np.float32, 9600)),  # 40 * 40 * 6
            ('goal_spikings', (np.float32, 9600))  # 40 * 40 * 6
        ])
    else:
        dtype = np.dtype([
            ('start_observation', (np.int32, 16384)),  # 64 * 64 * 4
            ('goal_observation', (np.int32, 16384)),
            ('reached', np.float32),
            ('start', (np.float32, 2)),  # x, y
            ('goal', (np.float32, 2)),  # x, y
            ('start_orientation', np.float32),  # theta
            ('goal_orientation', np.float32)  # theta
        ])

    seed = 555555
    rng_sampleid = np.random.RandomState(seed)

    i = 0
    sum_r = 0
    while i < nr_samples:
        import time
        start_time = time.time()
        sample_id = rng_sampleid.randint(0xfffffff)
        dset_name = '/%08x' % sample_id

        print('processing sample %d id: %08x' % (i, sample_id))

        if dset_name in f:
            print('dataset %s exists. skipped' % dset_name)
            i += 1
            continue

        random_index = random.randrange(rd.traj_len_cumsum[-1])
        item = rd.__getitem__(random_index)
        if not item:
            print(f'Failed to get item {random_index}')
            continue
        src_img, dst_img, r, s, orientations, src_spikings, dst_spikings = item

        sample = (
            src_img,
            dst_img,
            r,
            s[0],
            s[1],
            orientations[0],
            orientations[1]
            # first_gv, #not used
            # orientations[1] - orientations[0], #not used
            # img_after_turn[2].flatten(), #not used
            # dist #not used
        )
        if with_grid_cell_spikings:
            sample = (*sample, src_spikings, dst_spikings)

        dset = f.create_dataset(
            dset_name,
            data=np.array([sample], dtype=dtype),
            maxshape=(None,), dtype=dtype)

        f.flush()
        i += 1
        sum_r += r
        print(f"--- {time.time() - start_time} seconds --- to create {i}th sample, percentage reachable {sum_r / i}")


def display_samples(filename, imageplot=False):
    """ Display information about dataset file
    
    if imageplot: plot the stored images
    Calculate the percentage of reached/failed samples.
    """

    dirname = get_path()
    directory = os.path.join(dirname, "data/reachability")
    directory = os.path.realpath(directory)
    hf = h5py.File(directory + "/" + filename, 'r')
    env_model = hf.attrs["map_type"]

    print("Number of samples: " + str(len(hf.keys())))
    reached = 0
    count = len(hf.keys())
    reach = []
    starts_goals = []
    for i, key in enumerate(list(hf.keys())):
        if i % 1000 == 0:
            print("At sample number", i)
        if imageplot and i > 5000:
            break
        d = hf[key][()][0]

        if imageplot:
            img = np.reshape(d[0], (64, 64, 4))
            imgplot = plt.imshow(img)
            plt.show()

            dat = np.array_split(d[1], 10)
            for dt in dat:
                img = np.reshape(dt, (64, 64, 4))
                imgplot = plt.imshow(img)
                plt.show()
        if d[2] == 1.0:
            reached += 1
        starts_goals.append((d[3], d[4]))
        reach.append(d[2])
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

    test = True
    if test:
        # create_and_save_reachability_samples("dataset_spikings", 200000, "trajectories.hd5", with_grid_cell_spikings=True)
        display_samples("dataset_spikings.hd5")
        # create_and_save_reachability_samples("test2", 1, "test_2.hd5")
        # display_samples("test2.hd5")
        # create_and_save_reachability_samples("test3", 1, "test_3.hd5")
        # display_samples("test3.hd5")
    elif len(sys.argv) == 4:
        _, filename, num_samples, traj_file = sys.argv
        print_debug(sys.argv)
        create_and_save_reachability_samples(filename, int(num_samples), traj_file)
    else:
        import time

        start = time.time()
        create_and_save_reachability_samples("reachability_fifty_sampels", 50, "trajectories_Savinov3.hd5")
        end = time.time()
        print(end - start)
