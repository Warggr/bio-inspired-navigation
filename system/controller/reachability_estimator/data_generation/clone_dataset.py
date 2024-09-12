import h5py
import numpy as np
from tqdm import tqdm
from system.controller.simulation.pybullet_environment import PybulletEnvironment, all_possible_textures
from system.controller.reachability_estimator.data_generation.dataset import DATASET_KEY, display_samples
from system.controller.reachability_estimator.types import Sample
import sys
from typing import Literal


def clone_dataset(
    infile: h5py.File, out_filename: str,
    env: PybulletEnvironment,
    nr_samples: int|Literal['all']=1000,
    flush_freq=50,
) -> h5py.File:
    """ Create reachability samples.

    arguments:
    rd : dataset to draw positions from
    """

    in_dset = infile[DATASET_KEY]
    if nr_samples is not 'all':
        assert len(in_dset) >= nr_samples, f"Trying to clone {nr_samples} samples from dataset with only {len(in_dset)}"

    f = h5py.File(out_filename, 'a')

    attrs = dict(infile.attrs)
    attrs.update({'wall_colors': args.wall_colors})
    if attrs['map_type'] == '':
        attrs['map_type'] = env.env_model
    for attr in attrs:
        if attr in f.attrs:
            assert f.attrs[attr] == attrs[attr], f"{attr}: expected {attrs[attr]}, is {f.attrs[attr]}"
        else:
            f.attrs[attr] = attrs[attr]

    dtype = Sample.dtype

    try:
        out_dset = f[DATASET_KEY]
        old_size = out_dset.size
        start_index = old_size

        if old_size < nr_samples:
            # Hint: this might fail if somehow the dtype changed from one dataset to the other
            out_dset = f.create_dataset('tmp', dtype=dtype, data=out_dset[:], maxshape=(nr_samples,), compression="gzip")
            del f[DATASET_KEY]
            f.move('tmp', DATASET_KEY)
    except KeyError:
        out_dset = f.create_dataset(DATASET_KEY, data=np.array([], dtype=dtype), dtype=dtype, maxshape=(nr_samples,), compression="gzip")
        start_index = 0

    bar = tqdm(initial=start_index, total=nr_samples)
    for i in range(start_index, nr_samples, flush_freq):
        actual_block_size = min(flush_freq, nr_samples - i)
        data = in_dset[i:i + actual_block_size] # this does a copy, so we can modify data
        for row in data:
            row['start_observation'] = env.camera((row['start'], row['start_orientation'])).flatten()
            row['goal_observation'] = env.camera((row['goal'], row['goal_orientation'])).flatten()
        try:
            out_dset.resize((i+actual_block_size,))
            out_dset[i:] = data
        except Exception: # roll back if anything happens - else the dataset will get saved with a bunch of empty fields
            out_dset.resize((i,))
            raise
        f.flush()
        bar.update(i+actual_block_size)
    return f


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='Dataset file to copy')
    parser.add_argument('outfile', help='Output file', nargs='?')
    parser.add_argument('-w', '--wall-colors', help='how to color the walls', choices=['1color', '3colors', 'patterns'], default='1color')
    parser.add_argument('-n', '--num-samples', type=lambda s: 'all' if s == 'all' else int(s), dest='num_samples', default=200000)
    parser.add_argument('--flush-freq', type=int, dest='flush_freq', default=1000)
    parser.add_argument('--image-plot', action=argparse.BooleanOptionalAction, help='Show image of samples taken')
    args = parser.parse_args()

    if args.wall_colors == '1color':
        textures = ['yellow_wall.png']
    elif args.wall_colors == '3colors':
        textures = all_possible_textures[:3]
    elif args.wall_colors == 'patterns':
        textures = lambda i : f'pattern-{i+1}.png'
    else:
        raise ValueError(f"Unrecognized textures: '{args.wall_colors}'")

    if args.outfile is not None:
        out_filename = args.outfile
    else:
        out_filename = args.infile.removesuffix('.hd5') + '-' + args.wall_colors + '.hd5'

    infile = h5py.File(args.infile)
    env_model, = infile.attrs['map_type'].split(',')
    if env_model == '':
        print('Warning: no env model provided in dataset, assuming Savinov_val3', file=sys.stderr)
        env_model = 'Savinov_val3'

    with PybulletEnvironment(env_model, contains_robot=False, wall_kwargs={'textures': textures}) as env:
        outfile = clone_dataset(infile, out_filename, env, nr_samples=args.num_samples, flush_freq=args.flush_freq)

    if args.image_plot:
        print("Finished creating samples. Now displaying them")
        display_samples(outfile, imageplot=True)
