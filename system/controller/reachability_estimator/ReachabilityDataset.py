''' This code has been adapted from:
https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?fbclid=IwAR2jFrRkKXv4PL9urrZeiHT_a3eEn7eZDWjUaQ-zcLP6BRtMO7e0nMgwlKU
'''
import torch
import h5py
import numpy as np
import bisect
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from system.plotting.plotResults import plotStartGoalDataset
from system.bio_model.bc_network import bcActivityForLidar, BoundaryCellNetwork, HDCActivity
from system.controller.reachability_estimator.types import Sample, Prediction, ModelInput

import sys
import os
from typing import Literal, Optional
from dataclasses import dataclass

boundaryCellEncoder = BoundaryCellNetwork.load()

@dataclass
class SampleConfig:
    def __init__(self,
        grid_cell_spikings=False,
        lidar: Optional[Literal['raw_lidar', 'ego_bc', 'allo_bc']]=None,
        images: bool|Literal['zeros', 'fixed'] = True,
        image_crop: int|None = None,
        dist = False,
    ):
        self.with_grid_cell_spikings = grid_cell_spikings
        self.lidar = lidar
        self.images = images
        self.image_crop = image_crop
        self.with_dist = dist

    def suffix(self) -> str:
        return (''
            + ('+spikings' if self.with_grid_cell_spikings else '')
            + (f'+lidar--{self.lidar}' if self.lidar else '')
            + ('+noimages' if not self.images else '' if self.images is True else f'+{self.images}images')
            + (('+crop' + ('X' if self.image_crop > 0 else 'N') + str(abs(self.image_crop))) if self.image_crop is not None else '')
            + ('+dist' if self.with_dist else '')
        )

    @staticmethod
    def from_filename(filename: str) -> tuple['SampleConfig', dict]:
        filename, _extension = filename.split('.')
        filename, *tags = filename.split('+')
        config = SampleConfig()
        other_attrs = {'basename': filename}
        for tag in tags:
            if tag == 'spikings': config.with_grid_cell_spikings = True
            elif tag.startswith('lidar--'):
                config.lidar = tag.removeprefix('lidar--')
            elif tag.endswith('images'):
                tag = tag.removesuffix('images')
                if tag == 'no': config.images = False
                else: config.images = tag
            elif tag.startswith('crop'):
                tag = tag.removeprefix('crop')
                sign, value = {'X': +1, 'N': -1}[tag[0]], int(tag[1:])
                config.image_crop = sign * value
            elif tag == 'dist': config.with_dist = True
            elif tag in ('conv', 'fc'):
                other_attrs['image_encoder'] = tag
            elif tag.startswith('fc'):
                other_attrs['fc_layers'] = list(map(int, tag[2:].split(',')))
            elif tag == 'dropout':
                other_attrs['dropout'] = True
            else:
                raise ValueError('Unrecognized tag: ', tag)
        return config, other_attrs

DATA_STORAGE_FOLDER = os.path.join(os.path.dirname(__file__), "data", "reachability")


def ego_to_allo(ego, angle):
    heading = HDCActivity.headingCellsActivityTraining(angle)
    _, allo = boundaryCellEncoder.calculateActivities(ego, heading)
    return allo


# TODO Pierre: this is conceptually the dataset that is created by data_generation/dataset.py, why aren't they in the same file?
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

    def __init__(self, filename, external_link=False, sample_config: SampleConfig = SampleConfig(), dirname = DATA_STORAGE_FOLDER):
        self.file_path = os.path.join(dirname, filename)
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
            self.dataset_len = len(self.dataset['positions'])

        if self.config.images == 'fixed':
            self.fixed_image = self.sample(0)[0].src.img

    def sample(self, index) -> tuple[Sample, bool]:
        if self.externalLink:
            tup = self.dataset[str(self._get_link_index(index))][self.keys[index]][()]
        else:
            tup = self.dataset['positions'][index]
        return Sample.from_tuple(tup)

    def __getitem__(self, index) -> tuple[ModelInput, Prediction]:
        sample, reachability = self.sample(index)

        reachability = torch.tensor(reachability).clamp(0.0, 1.0) # make it a tensor of float
        position = sample.dst.pos - sample.src.pos
        angle = sample.dst.angle - sample.src.angle
        ground_truth = (reachability, torch.tensor(position), torch.tensor(angle))

        None_tensor = torch.tensor(torch.nan) # we can't use None because it has to be a tensor to be collated into batches by the dataloader
        model_args = []

        if self.config.images:
            if self.config.images == 'zeros':
                img_src, img_dst = np.zeros((64, 64, 4), dtype=float), np.zeros((64, 64, 4), dtype=float)
            elif self.config.images == 'fixed':
                img_src, img_dst = self.fixed_image, self.fixed_image
            else:
                img_src, img_dst = sample.src.img, sample.dst.img
            for image in (img_src, img_dst):
                if self.config.image_crop is not None and self.config.image_crop > 0:
                    image = image.copy()
                    lowbound, upbound = self.config.image_crop, 64-self.config.image_crop
                    image[:lowbound, :].fill(0) # top and bottom margin
                    image[upbound:, :].fill(0)
                    image[lowbound:upbound, :lowbound].fill(0) # left and right margin
                    image[lowbound:upbound, upbound:].fill(0)
                elif self.config.image_crop is not None and self.config.image_crop < 0:
                    image = image.copy()
                    lowbound, upbound = -self.config.image_crop, 64+self.config.image_crop
                    image[lowbound:upbound, lowbound:upbound].fill(0)
                model_args += [torch.tensor(image).float()]
        else:
            model_args += [ None_tensor, None_tensor ]

        if self.config.with_dist:
            model_args += [ torch.tensor(np.append(position, angle)) ]
        else:
            model_args += [ None_tensor ]

        if self.config.with_grid_cell_spikings:
            model_args += [ torch.tensor(sample.src.spikings), torch.tensor(sample.dst.spikings) ]
        else:
            model_args += [ None_tensor, None_tensor ]

        if self.config.lidar:
            src_lidar, dst_lidar = sample.src.lidar, sample.dst.lidar
            if self.config.lidar in ['allo_bc', 'ego_bc']:
                src_lidar = bcActivityForLidar(src_lidar)
                dst_lidar = bcActivityForLidar(dst_lidar)
            else:
                src_lidar = src_lidar.distances
                dst_lidar = dst_lidar.distances
            assert not np.isnan(np.min(src_lidar))

            if self.config.lidar == 'allo_bc':
                src_lidar = ego_to_allo(src_lidar, sample.src.angle)
                dst_lidar = ego_to_allo(dst_lidar, sample.dst.angle)
            assert not np.isnan(np.min(src_lidar))

            model_args += [ torch.tensor(src_lidar).float(), torch.tensor(dst_lidar).float() ]
        else:
            model_args += [ None_tensor, None_tensor ]


        return model_args, ground_truth

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
            maxshape=(None,), dtype=dtype,
            compression="gzip",
        )
        new_f.flush()


def combine_datasets(new_file, filenames, filepath):
    """ Combine multiple datasets through external links """
    myfile = h5py.File(new_file, 'w')
    for i, fn in enumerate(filenames):
        myfile[str(i)] = h5py.ExternalLink(fn, filepath)


if __name__ == '__main__':
    """Test H5 reachability datasets by displaying their content."""
    path = DATA_STORAGE_FOLDER
    new_file = os.path.join(path, "reachability_combined_dataset.hd5")

    # Combine multiple datasets in case you have multiple
    filenames = ["trajectories.hd5", "reachability_dataset.hd5"]
    filenames = [os.path.join(path, fn) for fn in filenames]
    combine_datasets(new_file, filenames, "/")

    dataset = ReachabilityDataset(new_file, external_link=False)
    dataset.display_sample(0)
    dataset.display_dataset()
