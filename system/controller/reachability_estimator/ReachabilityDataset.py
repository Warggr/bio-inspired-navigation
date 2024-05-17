''' This code has been adapted from:
https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU
'''
import torch
import h5py
import numpy as np
import bisect
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from system.controller.reachability_estimator.training.utils import img_reshape, spikings_reshape
from system.plotting.plotResults import plotStartGoalDataset
from system.bio_model.bc_network import bcActivityForLidar, BoundaryCellNetwork, HDCActivity, BoundaryCellActivity

import sys
import os
from typing import Tuple

boundaryCellEncoder = BoundaryCellNetwork.load()

class SampleConfig:
    def __init__(self,
        grid_cell_spikings=False,
        lidar=None,
    ):
        self.with_grid_cell_spikings = grid_cell_spikings
        self.lidar = lidar
    def suffix(self) -> str:
        return (''
            + ('+spikings' if self.with_grid_cell_spikings else '')
            + (f'+lidar--{self.lidar}' if self.lidar else '')
        )


class ReachabilityDataset(torch.utils.data.Dataset):
    """ create a pytorch compatible dataset from a reachability sample hd5 file

    arguments:
        path            -- path to the hd5 file
        externalLink    -- true if the dataset is a combination of various datasets using external links (default False)

    returns:
        src_img         -- image from agent's pov at start
        dst_img         -- image from agent's pov at goal
        reachability    -- reachability score (1.0 for reachable, 0.0 for unreachable)
        transformation  -- transformation between source position and goal position
    """

    def __init__(self, path, external_link=False, sample_config : SampleConfig = SampleConfig()):
        self.file_path = path
        self.dataset = None
        self.externalLink = external_link
        self.dataset = h5py.File(self.file_path, 'r')
        self.config = sample_config

        if external_link:
            self.dataset_len = 0
            self.cumsum = [0]
            self.keys = []
            for k in list(self.dataset.keys()):
                self.dataset_len += len(self.dataset[k]['positions'])
                self.cumsum.append(len(self.dataset[k]['positions']) + self.cumsum[-1])
        else:
            self.dataset_len = len(list(self.dataset['positions']))

    def sample(self, index):
        if self.externalLink:
            return self.dataset[str(self._get_link_index(index))][self.keys[index]][()]
        else:
            return self.dataset['positions'][index]

    def __getitem__(self, index):
        src_img, dst_img, \
            reachability, start_position, goal_position, \
            start_orientation, goal_orientation, \
            start_spikings, goal_spikings, \
            start_lidar, goal_lidar = self.sample(index)
        src_img = img_reshape(src_img)
        dst_img = img_reshape(dst_img)
        reachability = torch.tensor(reachability).clamp(0.0, 1.0) # make it a tensor of float
        transformation = torch.tensor( np.append(goal_position, goal_orientation) - np.append(start_position, start_orientation))

        result = [src_img, dst_img, reachability, transformation]

        if self.config.with_grid_cell_spikings:
            result += [ start_spikings, goal_spikings ]

        if self.config.lidar in ['allo_bc', 'ego_bc']:
            start_lidar = bcActivityForLidar(start_lidar)
            goal_lidar = bcActivityForLidar(goal_lidar)

        if self.config.lidar == 'allo_bc':
            def ego_to_allo(ego, angle):
                heading = HDCActivity.headingCellsActivityTraining(angle)
                _, allo = boundaryCellEncoder.calculateActivities(ego, heading)
                return allo
            start_lidar = ego_to_allo(start_lidar, start_orientation)
            goal_lidar = ego_to_allo(goal_lidar, goal_orientation)

        if self.config.lidar:
            result += [ start_lidar, goal_lidar ]

        return result

    def __len__(self):
        return self.dataset_len

    def _get_link_index(self, index):
        return bisect.bisect_right(self.cumsum, index) - 1

    def display_sample(self, index):
        """ Display a single sample """
        sample = self[index]
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(sample[0].transpose(1, 2, 0))
        plt.show()

        fig = plt.figure()
        columns = 5
        rows = 2
        plt.axis('off')
        for i in range(10):
            fig.add_subplot(rows, columns, i + 1)
            ax = plt.gca()

            # hide x-axis
            ax.get_xaxis().set_visible(False)

            # hide y-axis
            ax.get_yaxis().set_visible(False)
            plt.imshow(sample[1].transpose(1, 2, 0))
        plt.show()
        print(sample[2])
        print(sample[3])
        plotStartGoalDataset("Savinov_val3", [(sample[4], sample[5])])

    def display_dataset(self):
        """ Display a histogram of start and goal positions of the samples """
        x = []
        y = []
        reach = []
        for i in range(len(self)):
            x.append(self[i][4][0])
            y.append(self[i][4][1])
            x.append(self[i][5][0])
            y.append(self[i][5][1])
            reach.append(self[i][2])
            if i % 1000 == 0:
                print("progress", i)

        # get dimensions
        fig, ax = plt.subplots()
        hh = ax.hist2d(x, y, bins=[np.arange(-9, 6, 0.1)
            , np.arange(-5, 4, 0.1)], norm="symlog")
        fig.colorbar(hh[3], ax=ax)
        plt.show()

        print("reached", reach.count(1.0))
        print("failed", reach.count(0.0))
        print("percentage reached/failed", reach.count(1.0) / len(reach))


def create_balanced_datasets(new_file, filename, filepath, length):
    """ Combine two hd5 files into one. Keys need to be different

    arguments:

    new_file    -- new filename
    filenames   -- files to be combined
    filepath    -- path to storage folder
    """
    new_f = h5py.File(new_file, 'w')

    dtype = np.dtype([
        ('start_observation', (np.int32, 16384)),
        ('goal_observation', (np.int32, 16384 * 10)),
        ('reached', np.float32),
        ('start', (np.float32, 2)),
        ('goal', (np.float32, 2)),
        ('start_orientation', np.float32),  # theta
        ('goal_orientation', np.float32),  # theta
        ('decoded_goal_vector', (np.float32, 2)),
        ('rotation', np.float32),  # dtheta
        ('start_observation_after_turn', (np.int32, 16384)),
        ('distance', (np.float32, 2))
    ])

    dataset = h5py.File(filepath + filename, 'r')
    keys = list(dataset.keys())
    p = 0
    n = 0
    for k in keys:
        if k in new_f:
            print('dataset %s exists. skipped' % k)
            continue
        if p >= length and n >= length:
            break
        sample = dataset[k][()]
        if sample[0][2] == 1:
            p += 1
            if p > length:
                continue
        if sample[0][2] == 0:
            n += 1
            if n > length:
                continue
        dset = new_f.create_dataset(
            k,
            data=np.array(sample, dtype=dtype),
            maxshape=(None,), dtype=dtype)
        new_f.flush()


def combine_datasets(new_file, filenames, filepath):
    """ Combine multiple datasets through external links """
    myfile = h5py.File(new_file, 'w')
    for i, fn in enumerate(filenames):
        myfile[str(i)] = h5py.ExternalLink(fn, filepath)


def get_path():
    """ returns path to data storage folder """
    dirname = os.path.join(os.path.dirname(__file__), "..")
    return dirname


if __name__ == '__main__':
    """Test H5 reachability datasets by displaying their content."""
    path = get_path()
    path = os.path.join(path, "data/reachability")
    new_file = os.path.join(path, "reachability_combined_dataset.hd5")

    # Combine multiple datasets in case you have multiple
    filenames = ["trajectories.hd5", "reachability_dataset.hd5"]
    filenames = [os.path.join(path, fn) for fn in filenames]
    combine_datasets(new_file, filenames, "/")

    dataset = H5Dataset(new_file, external_link=False)
    dataset.display_sample(0)
    dataset.display_dataset()
