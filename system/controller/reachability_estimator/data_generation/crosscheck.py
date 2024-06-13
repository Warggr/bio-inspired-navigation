if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from system.controller.reachability_estimator.types import ReachabilityController
from dataset import SampleGenerator, get_path, TrajectoriesDataset
from system.controller.simulation.environment_cache import EnvironmentCache

from typing import Tuple

def crosscheck_re(
    res : Tuple[ReachabilityController, ReachabilityController], samples : SampleGenerator,
    env_cache : EnvironmentCache,
    num_samples : int,
    flush_freq : int = 10,
):
    from tqdm import tqdm

    results = { (True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0 }

    for i in (bar := tqdm(range(num_samples))):
        sample = None
        while sample is None:
            try:
                sample, path_l, map_name = next(samples)

                env = env_cache[map_name]

                reachable = tuple( reachability_controller.reachable(env, sample.src, sample.dst, path_l) for reachability_controller in res )
            except (ValueError, AssertionError):
                tqdm.write(f"Error at sample {i}")
                continue
        results[reachable] += 1

        if i % flush_freq == (flush_freq - 1):
            bar.set_description('/'.join(map(str, results.values())))

    print('\t2>True\t2>False\t|')
    for i, re1pred in enumerate([True, False]):
        print(f'1>{re1pred}', end='\t')
        for j, re2pred in enumerate([True, False]):
            percentage = results[re1pred, re2pred] / num_samples
            print(f'{results[re1pred, re2pred]} ({percentage:2.1f}%)', end='\t')
        print('|')

    print(f'Assuming the first RE ({repr(res[0])}) to be the ground truth:')
    precision = results[True, True] / (results[True, True] + results[False, True])
    recall = results[True, True] / (results[True, True] + results[True, False])
    print('Accuracy:', (results[True, True] + results[False, False]) / num_samples)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1Score:', 2*precision*recall/(precision+recall))

if __name__ == "__main__":
    """
    Check how much different methods for estimating reachability agree with each other.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('re_types',
        nargs=2,
        choices=['view_overlap', 'network', 'distance', 'simulation'],
        help='RE algorithms to compare',
    )
    parser.add_argument('-n', '--num-samples', type=int, default=400)
    parser.add_argument('-w', '--wall-colors', help='how to color the walls', choices=['1color', '3colors', 'patterns'], default='1color')
    parser.add_argument('--gen', '--generate-point-pairs',
        choices=['same_traj', 'diff_traj', 'maybe_diff_traj', 'random', 'random_circle'], default='same_traj',
        help='How to generate pairs of points',
    )
    parser.add_argument('--traj_file', help='Dataset of trajectories used for the _traj point-pair generators', nargs='?', default='trajectories.hd5')
    args = parser.parse_args()

    res = [ ReachabilityController.factory(controller_type=retype) for retype in args.re_types ]

    if args.wall_colors == '1color':
        textures = [ os.path.join( 'yellow_wall.png') ]
    elif args.wall_colors == '3colors':
        textures = all_possible_textures[:args.wall_colors]
    elif args.wall_colors == 'patterns':
        textures = lambda i : f'pattern-{i+1}.png'
    env_kwargs={ 'wall_kwargs': { 'textures': textures } }

    with EnvironmentCache(override_env_kwargs=env_kwargs) as env_cache:
        if args.gen.endswith('_traj'):
            filename = os.path.join(get_path(), "data", "trajectories", args.traj_file)
            filename = os.path.realpath(filename)
            rd = TrajectoriesDataset([filename], env_cache=env_cache)
            samples = rd.iterate(mode=args.gen)
        else:
            if args.gen == 'random':
                samples = RandomSamples(env_cache["Savinov_val3"])
            elif args.gen == 'random_circle':
                samples = RandomSamplesWithLimitedDistance(env_cache["Savinov_val3"])
            else: raise ValueError(args.gen)

        crosscheck_re(
            res, samples,
            env_cache=env_cache,
            num_samples=args.num_samples,
        )
 