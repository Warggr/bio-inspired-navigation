''' This code has been adapted from:
***************************************************************************************
*    Title: "Scaling Local Control to Large Scale Topological Navigation"
*    Author: "Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox"
*    Date: 2020
*    Availability: https://github.com/xymeng/rmp_nav
*
***************************************************************************************
'''

import sys
from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    batch_size: int = 64
    samples_per_epoch: int = 10000
    max_epochs: int = 25
    lr: float = 3e-4
    lr_decay_epoch: int = 1
    lr_decay_rate: float = 0.7
    eps: float = 1e-5


def optional(typ):
    def _parser(st):
        if st.lower() in ['none', 'off']:
            return None
        return typ(st)
    return _parser


import argparse

def suffix_for(cmdline: list[str]):
    hyperparams_parser = argparse.ArgumentParser(add_help=False)
    for field in Hyperparameters.__dataclass_fields__.values():
        hyperparams_parser.add_argument('--' + field.name.replace('_', '-'), help='hyperparameter ' + field.name, type=field.type, default=field.default)

    model_basename = 'reachability_network'
    parser = argparse.ArgumentParser(parents=[hyperparams_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-features', nargs='+', default=[])
    parser.add_argument('--dataset-basename', help='The base name of the reachability dataset HD5 file', default='dataset')
    parser.add_argument('--tag', help=f'Network saved in `{model_basename}-{{tag}}`', default='')

    parser.add_argument('--images', help='Images are included in the dataset', nargs='?', default=True, choices=[True, False, 'zeros', 'fixed'], type=lambda s: False if s in ['no', 'off'] else s)
    parser.add_argument('--spikings', help='Grid cell spikings are included in the dataset', action='store_true')
    parser.add_argument('--lidar', help='LIDAR distances are included in the dataset', choices=['raw_lidar', 'ego_bc', 'allo_bc'])
    parser.add_argument('--dist', help='Provide the distance and angle to the reachability estimator', action='store_true')
    parser.add_argument('--image-crop', help='Cover the border (+n) or center (-n) pixels in white', type=int)

    parser.add_argument('--image-encoder', help='Image encoder', choices=['fc', 'conv', 'pretrained'], default='conv')
    parser.add_argument('--hidden-fc-layers', help='Hidden FC layer dimensions as a comma-separated list', type=lambda s: [int(i) for i in s.split(',')])
    parser.add_argument('--dropout', help='Use dropout in the hidden FC layers', action='store_true')

    parser.add_argument('--resume', action='store_true', help='Continue training from last saved model')
    parser.add_argument('--save-interval', type=optional(int))

    args = parser.parse_args(cmdline)

    if args.images is None:
        args.images = True # --images means --images=True but I don't know how to make argparse do that automatically

    config = SampleConfig(
        grid_cell_spikings=args.spikings,
        lidar=args.lidar,
        images=args.images,
        image_crop=args.image_crop,
        dist=args.dist,
    )

    suffix = ''
    if args.tag:
        suffix += '-' + args.tag
    args.dataset_features = ''.join([ f'-{feature}' for feature in args.dataset_features ])
    suffix += args.dataset_features
    suffix += config.suffix()
    if args.image_encoder:
        suffix += '+' + args.image_encoder
    if args.hidden_fc_layers:
        suffix += '+fc' + ','.join(map(str, args.hidden_fc_layers))
    if args.dropout:
        suffix += '+dropout'

    return suffix

def flags_for(netname: str):
    config, attrs = SampleConfig.from_filename(netname)

    flags = []

    if config.images is False:
        flags.append('--images off')
    elif config.images in ['zeros', 'fixed']:
        flags.append('--images ' + config.images)
    if config.with_grid_cell_spikings:
        flags.append('--spikings')
    if config.lidar is not None:
        flags.append('--lidar=' + config.lidar)
    if config.with_dist:
        flags.append('--dist')
    if config.image_crop is not None:
        flags.append(f'--image-crop={config.image_crop}')

    if 'image_encoder' in attrs:
        flags.append('--image-encoder=' + attrs['image_encoder'])
    if 'fc_layers' in attrs:
        flags.append('--hidden-fc-layers ' + ','.join(map(str, attrs['fc_layers'])))
    if 'dropout' in attrs:
        assert attrs['dropout'] == True
        flags.append('--dropout')

    tag = None
    basename, *dataset_features = attrs['basename'].split('-')
    if dataset_features[0] not in ('3colors', 'patterns', 'boolor', 'simulation'):
        tag, *dataset_features = dataset_features

    dataset_features_and_tags = dataset_features
    dataset_features = []
    dataset_tags = []
    for i in dataset_features_and_tags:
        if i in ('3colors', 'patterns'): dataset_features.append(i)
        else: dataset_tags.append(i)

    if basename != 'reachability_network':
        raise ValueError('Unexpected network base name' + basename)
    if dataset_features:
        flags.append(f'--dataset-features ' + ' '.join(dataset_features))
    if dataset_tags:
        flags.append(f'--train-dataset-tags ' + ' '.join(dataset_tags))
    if tag is not None:
        flags.append(f'--tag {tag}')

    return ' '.join(flags)


if __name__ == "__main__":
    import sys
    import os

    reverse = False
    lines = map(lambda line: line.strip(), sys.stdin)

    if len(sys.argv) == 1:
        reverse = False
    else:
        reverse = {'cmdline': True, 'tags': False}[sys.argv[1]]
        if len(sys.argv) > 2:
            lines = sys.argv[2:]

    if reverse:
        for line in lines:
            line = os.path.basename(line)
            try:
                print(flags_for(line))
            except ValueError as err:
                print(f'"{line.strip()}": couldn\'t parse: {err}', file=sys.stderr)
    else:
        for line in lines:
            print(suffix_for(line.split(' ')))
