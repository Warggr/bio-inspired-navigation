# coding: utf-8
import torch
from system.controller.reachability_estimator.reachability_estimation import WEIGHTS_FOLDER
from system.controller.reachability_estimator.training.train_multiframe_dst import Hyperparameters
from dataclasses import asdict
import sys
import os

weights_files = sys.argv[1:]

for file in weights_files:
    weights_filepath = os.path.join(WEIGHTS_FOLDER, file)
    print(f'{file}:', end='')
    try:
        state_dict = torch.load(weights_filepath, map_location='cpu')
    except RuntimeError:
        print('Couldn\'t load network')
        continue

    image_encoder = ('conv' if '+conv' in file else 'pretrained' if '+pretrained' in file else 'fc')
    if 'global_args' in state_dict and (type(state_dict['global_args']) is Hyperparameters or state_dict['global_args'] is None):
        state_dict['global_args'] = {
            'backbone': 'convolutional',
            'image_encoder': image_encoder,
            'hyperparameters': (None if state_dict['global_args'] is None else asdict(state_dict['global_args']))
        }
        print('replaced Hyperparameters')
    elif 'global_args' in state_dict and 'with_conv_layer' in state_dict['global_args']:
        del state_dict['global_args']['with_conv_layer']
        state_dict['global_args']['image_encoder'] = image_encoder
        print('my bad')
    else:
        print('nothing to do')
        continue
    torch.save(state_dict, weights_filepath)
