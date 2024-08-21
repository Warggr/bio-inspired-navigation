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

    dirty = False
    image_encoder = ('conv' if '+conv' in file else 'pretrained' if '+pretrained' in file else 'fc')
    if 'global_args' in state_dict and (type(state_dict['global_args']) is Hyperparameters or state_dict['global_args'] is None):
        state_dict['global_args'] = {
            'backbone': 'convolutional',
            'image_encoder': image_encoder,
            'hyperparameters': (None if state_dict['global_args'] is None else asdict(state_dict['global_args']))
        }
        print('replaced Hyperparameters', end=',')
        dirty = True
    elif 'global_args' in state_dict and 'with_conv_layer' in state_dict['global_args']:
        del state_dict['global_args']['with_conv_layer']
        state_dict['global_args']['image_encoder'] = image_encoder
        print('write correct conv_layer key', end=',')
        dirty = True

    if 'fully_connected' in state_dict['nets']:
        fc = state_dict['nets']['fully_connected']
        sizes = [fc[key].shape[0] for key in fc if 'bias' in key]
        assert sizes[-1] == 4; sizes = sizes[:-1]
        if sizes != [256, 256] and ('hidden_fc_layers' not in state_dict['global_args'] or sizes != state_dict['global_args']['hidden_fc_layers']):
            state_dict['global_args']['hidden_fc_layers'] = sizes
            print('set nonstandard hidden_fc_layers', end='')
            dirty = True

    if dirty:
        torch.save(state_dict, weights_filepath)
        print('')
    else:
        print('nothing to do')
