import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from system.controller.reachability_estimator.autoencoders import ImageAutoencoder
from system.controller.reachability_estimator.training.utils import load_model
from system.controller.reachability_estimator.ReachabilityDataset import ReachabilityDataset
from system.controller.reachability_estimator.training.train_multiframe_dst import DATA_STORAGE_FOLDER
from system.controller.reachability_estimator.training.train_autoencoder import ImageDataset
from system.types import Image

Batch = list

def show(net: ImageAutoencoder, images: np.ndarray[Image]):
    with torch.no_grad():
        decodeds = net(torch.tensor(images).float()).int().numpy()
    for image, decoded in zip(images, decodeds):
        fig, axes = plt.subplots(2, 2)
        for i, img in enumerate([image, decoded]):
            axes[i,0].imshow(img)
            axes[i,1].imshow(img[:,:,3], cmap='gray')
        plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('code_dim', help='The dimension of the encoding', type=int, default=16)
    parser.add_argument('--dataset-features', type=str, default='', choices=['3colors', 'patterns'])
    parser.add_argument('-n', help='number of samples', type=int, default=10)
    args = parser.parse_args()

    dataset = ReachabilityDataset(filename='dataset' + args.dataset_features + ".hd5")
    dataset = ImageDataset(dataset)
    net = ImageAutoencoder(args.code_dim)

    filepath = os.path.join(DATA_STORAGE_FOLDER, f'autoencoder{args.code_dim}')

    state, epoch = load_model(filepath)
    net.load_state_dict(state)
    print('Loaded model at epoch', epoch)

    images = np.array([ dataset[i] for i in range(args.n) ])
    show(net, images)
