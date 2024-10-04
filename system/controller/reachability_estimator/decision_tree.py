from scipy.sparse import data
from sklearn import tree
import torch
from torch.utils.data import DataLoader
import numpy as np

import sys, os
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from system.controller.reachability_estimator.networks import Model, get_grid_cell
from system.controller.reachability_estimator.autoencoders import ImageEncoder
from system.controller.reachability_estimator._types import Batch, Prediction, Sample
from system.controller.reachability_estimator.ReachabilityDataset import ReachabilityDataset, SampleConfig

from system.types import Image
from typing import Any

def relevant_info_from_lidar(batch_lidar: Batch[list[float]]) -> Batch[list[float]]:
    return batch_lidar[:, [0, 25, 26, 51]]

class DecisionTree(Model):
    def __init__(self, encoder: ImageEncoder, tree: tree.DecisionTreeClassifier, sample_config: SampleConfig):
        self.encoder = encoder
        self.tree = tree
        self.sample_config = sample_config

    def n_features(self):
        """ The number of features passed to the decision tree. """
        n_features = 0
        if self.sample_config.images:
            n_features += 2*self.encoder.code_dim
        if self.sample_config.with_dist:
            n_features += 1
        if self.sample_config.with_grid_cell_spikings:
            n_features += 1
        if self.sample_config.lidar == "raw_lidar":
            n_features += 8
        return n_features

    def process_batch(self,
        batch_src_images=None, batch_dst_images=None,
        batch_transformation=None,
        batch_src_spikings=None, batch_dst_spikings=None,
        batch_src_lidar=None, batch_dst_lidar=None,
    ):
        features = []
        if self.sample_config.images:
            with torch.no_grad():
                src_images, dst_images = self.encoder(batch_src_images).numpy(), self.encoder(batch_dst_images).numpy()
            features += [src_images, dst_images]
        if self.sample_config.with_dist:
            features += [ np.expand_dims(np.linalg.norm(batch_transformation[:, :2], axis=1), axis=1) ]
        if self.sample_config.with_grid_cell_spikings:
            features += [ get_grid_cell(batch_src_spikings, batch_dst_spikings) ]
        if self.sample_config.lidar == "raw_lidar":
            features += [ relevant_info_from_lidar(batch_src_lidar), relevant_info_from_lidar(batch_dst_lidar) ]
        features = np.concatenate(features, axis=1)
        return features

    def get_prediction(self,
        *model_args,
    ) -> Batch[Prediction]:
        features = self.process_batch(*model_args)
        result = np.zeros((len(features), 4))
        result[:,0] = self.tree.predict(features)
        return result

    def train(self, dataset: 'Dataset[Sample]'):
        data = np.zeros((len(dataset), self.n_features()))
        labels = np.zeros((len(dataset), 1))
        i = 0

        loader = DataLoader(dataset, batch_size=64)

        for batch_model_args, batch_labels in loader:
            ilast = i+len(batch_labels[0])
            labels[i:ilast] = np.expand_dims(batch_labels[0], 1)
            batch_model_args = self.process_batch(*batch_model_args)
            data[i:ilast] = batch_model_args

        self.tree.fit(data, labels)

    def get_args(self) -> dict[str, Any]:
        return {}


if __name__ == "__main__":
    import argparse
    from system.controller.reachability_estimator.training.utils import load_model
    from system.controller.reachability_estimator.training.train_multiframe_dst import DATA_STORAGE_FOLDER
    from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--code-dim', help='dimension of the image encoding', type=int, default=16)
    parser.add_argument('--dataset-features', nargs='+', default=[])
    parser.add_argument('--dataset-basename', help='The base name of the reachability dataset HD5 file', default='dataset')
    parser.add_argument('--images', help='Images are included in the dataset', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--spikings', help='Grid cell spikings are included in the dataset', action='store_true')
    parser.add_argument('--lidar', help='LIDAR distances are included in the dataset', choices=['raw_lidar', 'ego_bc', 'allo_bc'])
    parser.add_argument('--dist', help='Provide the distance and angle to the reachability estimator', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--only-validate', action='store_true')
    group.add_argument('--no-validate', help='No testing of the trained network on the validation set', action='store_true')

    args = parser.parse_args()

    config = SampleConfig(
        grid_cell_spikings=args.spikings,
        lidar=args.lidar,
        images=args.images,
        dist=args.dist,
    )

    state, epoch = load_model(os.path.join(DATA_STORAGE_FOLDER, f'autoencoder{args.code_dim}'))
    encoder = ImageEncoder(args.code_dim)
    encoder.load_state_dict(state['nets']['encoder'])

    clf = DecisionTree(encoder, tree.DecisionTreeClassifier(), config)

    args.dataset_features = ''.join([ f'-{feature}' for feature in args.dataset_features ])
    suffix = str(args.code_dim) + args.dataset_features + config.suffix()

    filename = args.dataset_basename + args.dataset_features + ".hd5"
    dataset = ReachabilityDataset(filename, sample_config=config)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    import pickle
    tree_filename = os.path.join(DATA_STORAGE_FOLDER, 'tree' + suffix + '.pkl')
    if not args.only_validate:
        clf.train(train_dataset)

        with open(tree_filename, 'xb') as file:
            pickle.dump(clf.tree, file)
    else:
        with open(tree_filename, 'rb') as file:
            clf.tree = pickle.load(file)

    if not args.no_validate:
        metrics = dict(
            accuracy = BinaryAccuracy(),
            precision = BinaryPrecision(),
            recall = BinaryRecall(),
            f1 = BinaryF1Score(),
        )

        values = { key: .0 for key in metrics }

        loader = DataLoader(valid_dataset, batch_size=64)
        for batch_model_args, batch_labels in loader:
            prediction = clf.get_prediction(*batch_model_args)[:, 0]
            reachability = batch_labels[0]

            prediction, reachability = torch.tensor(prediction).int(), reachability.int()

            for key in metrics:
                values[key] += metrics[key](prediction, reachability)   

        for key in metrics:
            values[key] /= len(loader)
            print(f"{key}: {values[key]}")

    #tree.plot_tree(clf.tree)
