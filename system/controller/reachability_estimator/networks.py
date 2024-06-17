""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
from system.bio_model.bc_network.parameters import BoundaryCellActivity
from system.types import types

import math
import numpy as np
import os

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics import MeanSquaredError
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

Batch = List # only used for type hints - doesn't actually do anything

class AutoAdamOptimizer: # Sentinel value
    pass

class NNModuleWithOptimizer:
    __slots__ = ('net', 'opt')
    def __init__(self, net : nn.Module, opt : torch.optim.Optimizer|None = AutoAdamOptimizer):
        if opt is AutoAdamOptimizer:
            opt = torch.optim.Adam(net.parameters(), lr=3.0e-4, eps=1.0e-5)
        self.net = net
        self.opt = opt

class Model(ABC):
    Prediction = Tuple[float, types.Vector2D, types.Angle]

    """ Interface for networks """
    def __init__(self, nets : Dict[str, NNModuleWithOptimizer]):
        self.nets = { name: val.net for name, val in nets.items() }
        self.optimizers = { name: val.opt for name, val in nets.items() }

    @abstractmethod
    def get_prediction(self,
        batch_src_images=None, batch_dst_images=None,
        batch_transformation=None,
        batch_src_spikings=None, batch_dst_spikings=None,
        batch_src_distances=None, batch_dst_distances=None,
    ) -> Batch['Model.Prediction']:
        ...

    @staticmethod
    def create_from_config(backbone_classname : str, *model_args, **model_kwargs) -> 'Model':
        backbone_classes : Dict[str, Type[Model]] = {
            'convolutional': CNN, 'resnet': ResNet, 'siamese': Siamese
        }
        return backbone_classes[backbone_classname](*model_args, **model_kwargs)

    def save(self, epoch, global_args, model_file):
        """ save current state of the model """
        state = {
            'epoch': epoch,
            'global_args': global_args,
            'optims': {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            },
            'nets': {
                name: net.state_dict() for name, net in self.nets.items()
            }
        }

        path = os.path.join('', '%s.%d' % (model_file, epoch))
        torch.save(state, path)


class Siamese(Model):
    def __init__(self):
        nets = {}
        net = GridCellSiameseNetwork()
        nets["grid_cell_network"] = NNModuleWithOptimizer(
            net = net,
            opt = torch.optim.Adam(net.parameters(), lr=3.0e-3, eps=1.0e-5)
        )
        super().__init__(nets)

    def get_prediction(self,
        src_batch, dst_batch,
        batch_transformation=None,
        batch_src_spikings=None, batch_dst_spikings=None
    ) -> Batch[Model.Prediction]:
        return get_grid_cell(batch_src_spikings, batch_dst_spikings), None, None


class CNN(Model):
    def __init__(self, config : 'SampleConfig', with_conv_layer=True):
        self.with_conv_layer = with_conv_layer
        self.sample_config = config
        # Defining the NN and optimizers

        nets = initialize_regressors({})
        input_dim = 0

        if self.sample_config.images:
            input_dim += 512
            nets["img_encoder"] = NNModuleWithOptimizer( ImagePairEncoderV2(init_scale=1.0))
        if self.sample_config.with_dist:
            input_dim += 3
        if self.sample_config.with_grid_cell_spikings:
            input_dim += 1
        if self.with_conv_layer:
            nets["conv_encoder"] = NNModuleWithOptimizer( ConvEncoder(init_scale=1.0, input_dim=512, no_weight_init=False))
        if self.sample_config.lidar:
            one_lidar_input_dim = 52 if self.sample_config.lidar == 'raw_lidar' else BoundaryCellActivity.size
            nets["lidar_encoder"] = NNModuleWithOptimizer( LidarEncoder(2*one_lidar_input_dim, 10) )
            input_dim += 10
        nets["fully_connected"] = NNModuleWithOptimizer( FCLayers(init_scale=1.0, input_dim=input_dim, no_weight_init=False))

        super().__init__(nets)

    def get_prediction(self,
        batch_src_images=None, batch_dst_images=None,
        batch_transformation=None,
        batch_src_spikings=None, batch_dst_spikings=None,
        batch_src_distances=None, batch_dst_distances=None,
    ) -> Model.Prediction:
        # Extract features
        all_x = []

        if self.sample_config.images:
            batch_size = batch_dst_images.size()[0]
            x = self.nets['img_encoder'](batch_src_images, batch_dst_images)

            if self.with_conv_layer:
                # Convolutional Layer
                x = self.nets['conv_encoder'](x.view(batch_size, -1, 1))
                assert not torch.any(x.isnan())
            all_x.append(x)

        if self.sample_config.with_grid_cell_spikings:
            x = get_grid_cell(batch_src_spikings, batch_dst_spikings)
            assert not torch.any(x.isnan())
            all_x.append(x)

        if self.sample_config.with_dist:
            all_x.append(batch_transformation)

        if self.sample_config.lidar:
            assert not torch.any(batch_src_distances.isnan())
            assert not torch.any(batch_dst_distances.isnan())
            assert not torch.any(self.nets['lidar_encoder'].fc.weight.isnan())
            lidar_features = self.nets['lidar_encoder'](batch_src_distances, batch_dst_distances)
            assert not torch.any(lidar_features.isnan())
            all_x.append(lidar_features)

        x = torch.cat(all_x, 1)
        assert not torch.any(x.isnan())

        # Get prediction
        linear_features = self.nets['fully_connected'](x)

        assert not torch.any(linear_features.isnan())

        reachability_prediction = self.nets["reachability_regression"](linear_features)
        position_prediction = self.nets["position_regression"](linear_features)
        angle_prediction = self.nets["angle_regression"](linear_features)

        return reachability_prediction, position_prediction, angle_prediction

def initialize_regressors(nets) -> Dict[str, NNModuleWithOptimizer]:
    nets["angle_regression"] = NNModuleWithOptimizer( AngleRegression(init_scale=1.0, no_weight_init=False))
    nets["position_regression"] = NNModuleWithOptimizer( PositionRegression(init_scale=1.0, no_weight_init=False))
    nets["reachability_regression"] = NNModuleWithOptimizer( ReachabilityRegression(init_scale=1.0, no_weight_init=False))
    return nets

class ResNet(Model):
    def __init__(self):
        # Defining the NN and optimizers
        input_dim = 64
        nets = initialize_regressors({})

        nets["fully_connected"] = NNModuleWithOptimizer(**{
            'net': FcWithDropout(init_scale=1.0, input_dim=input_dim * 2, no_weight_init=False),
        })

        net = torchvision.models.resnet18(pretrained=True, num_classes=input_dim)
        weight1 = net.conv1.weight.clone()
        new_first_layer = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).requires_grad_()
        new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
        net.conv1 = new_first_layer

        nets["res_net"] = NNModuleWithOptimizer(**{
            'net': net,
            'opt': None
        })
        super().__init__(nets)

    def get_prediction(self,
        src_batch, dst_batch,
        batch_transformation=None,
        batch_src_spikings=None, batch_dst_spikings=None,
        batch_src_distances=None, batch_dst_distances=None,
    ) -> Batch[Model.Prediction]:
        # Extract features
        src_features = self.nets['res_net'](src_batch.view(batch_size, c, h, w))
        dst_features = self.nets['res_net'](dst_batch.view(batch_size, c, h, w))

        # Convolutional Layer
        pair_features = torch.cat([src_features, dst_features], dim=1)

        # Get prediction
        linear_features = self.nets['fully_connected'](pair_features)
        reachability_prediction = self.nets["reachability_regression"](linear_features)
        position_prediction = self.nets["position_regression"](linear_features)
        angle_prediction = self.nets["angle_regression"](linear_features)

        return reachability_prediction, position_prediction, angle_prediction


compare_mse = MeanSquaredError()
module_weights = torch.FloatTensor([32, 16, 8, 4, 2, 1])


def get_grid_cell(batch_src_spikings, batch_dst_spikings) -> [float]:
    """
    Calculate the similarity between two arrays of grid cell modules using Structural Similarity Index (SSIM)
    with weighted aggregation.

    Args:
    array1 (list of numpy arrays): First array of grid cell modules.
    array2 (list of numpy arrays): Second array of grid cell modules.

    Returns:
    float: Weighted SSIM-based similarity score.
    """
    batch_size = batch_src_spikings.shape[0]
    batch_similarity_scores = torch.zeros(6, batch_size)

    for ch in range(6):
        batch_similarity_scores[ch] = compare_mse(batch_src_spikings[:, ch].flatten(),
                                                  batch_dst_spikings[:, ch].flatten())
    batch_similarity_scores = torch.FloatTensor(batch_similarity_scores)
    batch_similarity_scores = torch.transpose(batch_similarity_scores, 0, 1)
    batch_similarity_scores = (batch_similarity_scores * module_weights).sum(1) / module_weights.sum()
    batch_similarity_scores = (torch.max(batch_similarity_scores,
                                         torch.fill(torch.zeros((batch_size,)), 0.99)) - 0.99) / 0.01
    return batch_similarity_scores.unsqueeze(1)


class GridCellSiameseNetwork(nn.Module):
    def __init__(self, no_weight_init=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 10 * 10, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.fc1, self.fc2):
                nn.init.orthogonal_(layer.weight, 1.0)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    with torch.no_grad():
                        layer.bias.zero_()

    def forward(self, x1, x2):
        x1 = self.pool(self.relu1(self.conv1(x1)))
        x1 = self.pool(self.relu2(self.conv2(x1)))
        x1 = x1.view(-1, 32 * 10 * 10)

        # Forward pass for the second grid cell module
        x2 = self.pool(self.relu1(self.conv1(x2)))
        x2 = self.pool(self.relu2(self.conv2(x2)))
        x2 = x2.view(-1, 32 * 10 * 10)

        x = torch.cat((x1, x2), dim=1)
        x = self.relu3(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)


class AngleRegression(nn.Module):
    def __init__(self, init_scale=1.0, bias=True, no_weight_init=False):
        super().__init__()

        self.fc = nn.Linear(4, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.two_pi = nn.Parameter(torch.tensor(math.pi * 2).squeeze(0))
        self.pi = nn.Parameter(torch.tensor(math.pi).squeeze(0))

        if not no_weight_init:
            nn.init.orthogonal_(self.fc.weight, init_scale)
            if hasattr(self.fc, 'bias') and self.fc.bias is not None:
                with torch.no_grad():
                    self.fc.bias.zero_()

    def forward(self, x) -> Batch[types.Angle]:
        x = self.fc(x)
        x = self.sigmoid(x)
        x = x * self.two_pi - self.pi
        return x.squeeze(1)


class PositionRegression(nn.Module):
    def __init__(self, init_scale=1.0, bias=True, no_weight_init=False):
        super().__init__()

        self.fc = nn.Linear(4, 2, bias=bias)

        if not no_weight_init:
            nn.init.orthogonal_(self.fc.weight, init_scale)
            if hasattr(self.fc, 'bias') and self.fc.bias is not None:
                with torch.no_grad():
                    self.fc.bias.zero_()

    def forward(self, x) -> Batch[types.Vector2D]:
        x = self.fc(x)
        return x


class ReachabilityRegression(nn.Module):
    def __init__(self, init_scale=1.0, bias=True, no_weight_init=False):
        super(ReachabilityRegression, self).__init__()

        self.fc = nn.Linear(4, 1, bias=bias)
        self.sigmoid = nn.Sigmoid()

        if not no_weight_init:
            nn.init.orthogonal_(self.fc.weight, init_scale)
            if hasattr(self.fc, 'bias') and self.fc.bias is not None:
                with torch.no_grad():
                    self.fc.bias.zero_()

    def forward(self, x):
        x = self.fc(x)
        assert not np.isnan(sum(x.detach().numpy()))
        x = self.sigmoid(x)
        assert not np.isnan(sum(x.detach().numpy()))
        return x.squeeze(1)


class FcWithDropout(nn.Module):
    def __init__(self, input_dim=2000, init_scale=1.0, bias=True, no_weight_init=False):
        super(FcWithDropout, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim, bias=bias)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(input_dim, input_dim // 4, bias=bias)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(input_dim // 4, input_dim // 4, bias=bias)
        self.fc4 = nn.Linear(input_dim // 4, 4, bias=bias)

        if not no_weight_init:
            for layer in (self.fc1, self.fc2, self.fc3):
                nn.init.orthogonal_(layer.weight, init_scale)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    with torch.no_grad():
                        layer.bias.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class LidarEncoder(nn.Module):
    def __init__(self, input_dim=2*52, output_dim=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x_src, x_dst):
        x = torch.cat([x_src, x_dst], dim=1)
        x = self.fc(x)
        return x

class FCLayers(nn.Module):
    def __init__(self, input_dim=512, init_scale=1.0, bias=True, no_weight_init=False):
        super(FCLayers, self).__init__()

        self.fc1 = nn.Linear(input_dim, input_dim // 2, bias=bias)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 2, bias=bias)
        self.fc3 = nn.Linear(input_dim // 2, 4, bias=bias)

        if not no_weight_init:
            for layer in (self.fc1, self.fc2, self.fc3):
                nn.init.orthogonal_(layer.weight, init_scale)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    with torch.no_grad():
                        layer.bias.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class ImagePairEncoderV2(nn.Module):
    def __init__(self, init_scale=1.0, bias=True, no_weight_init=False):
        super(ImagePairEncoderV2, self).__init__()

        # Input: 12 x 64 x 64
        self.layers = nn.Sequential(
        # img1, img2, img1 - img2 total 12 channels
            nn.Conv2d(12, 64, kernel_size=5, stride=2, bias=bias),
            nn.ReLU(),
        # 64 x 30 x 30
            nn.Conv2d(64, 128, kernel_size=5, stride=2, bias=bias),
            nn.ReLU(),
        # 128 x 13 x 13
            nn.Conv2d(128, 256, kernel_size=5, stride=2, bias=bias),
            nn.ReLU(),
        # 256 x 5 x 5
            nn.Conv2d(256, 512, kernel_size=5, stride=1, bias=bias),
            nn.ReLU(),
        # 512 x 1 x 1
        )

        if not no_weight_init:
            for layer in self.convs:
                nn.init.orthogonal_(layer.weight, init_scale)

    @property
    def convs(self):
        for layer in self.layers:
                if isinstance(layer, nn.Conv2d):
                    yield layer

    def forward(self, src_imgs, dst_imgs):
        assert src_imgs.shape[1:] == (64, 64, 4) # that's the format returned by env.camera() and used in the rest of the code
        src_imgs = src_imgs.transpose(1, 3).transpose(2, 3) # reorder 0123 -> 0321 -> 0312, i.e. batch size first
        dst_imgs = dst_imgs.transpose(1, 3).transpose(2, 3)
        assert src_imgs.shape[1:] == (4, 64, 64)

        x = torch.cat([src_imgs, dst_imgs, src_imgs - dst_imgs], dim=1)
        x = self.layers(x)
        return x.view(x.size(0), -1)


class SiameseNetwork(nn.Module):
    def __init__(self, init_scale=1.0, bias=True, no_weight_init=False):
        super(SiameseNetwork, self).__init__()

        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=bias)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=bias)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 10 * 10, 128, bias=bias)
        self.pool3 = nn.MaxPool2d(2)
        self.fc2 = nn.Linear(128, 8, bias=bias)

        if not no_weight_init:
            for layer in (self.conv1, self.conv2, self.fc1, self.fc2):
                nn.init.orthogonal_(layer.weight, init_scale)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    with torch.no_grad():
                        layer.bias.zero_()

    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        embed1 = self.forward_one(input1)
        embed2 = self.forward_one(input2)
        return torch.cat([embed1, embed2, embed1 - embed2], dim=1)


class ConvEncoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, kernel_size=1, init_scale=1.0,
                 no_weight_init=False):
        super(ConvEncoder, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size)
        if not no_weight_init:
            for layer in (self.conv,):
                nn.init.orthogonal_(layer.weight, init_scale)
                with torch.no_grad():
                    layer.bias.zero_()

    def forward(self, x):
        # Input size: batch_size x feature_dim x seq_len
        x = self.conv(x)
        x = F.relu(x)
        return x.flatten(1)


class ImageEncoderV3(nn.Module):
    def __init__(self, output_dim=512, init_scale=1.0, residual_link=False):
        super(ImageEncoderV3, self).__init__()
        self.residual_link = residual_link

        # Input: 3 x 64 x 64
        self.conv1 = nn.Conv2d(3, output_dim // 8, kernel_size=5, stride=2)
        if residual_link:
            self.res_fc1 = nn.Conv2d(output_dim // 8, output_dim // 4, kernel_size=1, stride=2)

        # 30 x 30
        self.conv2 = nn.Conv2d(output_dim // 8, output_dim // 4, kernel_size=5, stride=2)
        if residual_link:
            self.res_fc2 = nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=1, stride=2)

        # 13 x 13
        self.conv3 = nn.Conv2d(output_dim // 4, output_dim // 2, kernel_size=5, stride=2)
        if residual_link:
            self.res_fc3 = nn.Conv2d(output_dim // 2, output_dim, kernel_size=1, stride=1)

        # 5 x 5
        self.conv4 = nn.Conv2d(output_dim // 2, output_dim, kernel_size=5, stride=1)
        # 1 x 1

        for layer in (self.conv1, self.conv2, self.conv3, self.conv4):
            nn.init.orthogonal_(layer.weight, init_scale)

    def forward(self, imgs):
        if self.residual_link:
            x = F.relu(self.conv1(imgs))
            x = F.relu(self.res_fc1(x[:, :, 2:-2, 2:-2]) + self.conv2(x))
            x = F.relu(self.res_fc2(x[:, :, 2:-2, 2:-2]) + self.conv3(x))
            x = F.relu(self.res_fc3(x[:, :, 2:-2, 2:-2]) + self.conv4(x))
        else:
            x = F.relu(self.conv1(imgs))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))

        return x.view(x.size(0), -1)
