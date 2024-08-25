import os

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from system.controller.reachability_estimator.types import ReachabilityController
from system.controller.reachability_estimator.reachability_estimation import ReachabilityEstimator, reachability_estimator_factory
from dataset import SampleGenerator, get_path, TrajectoriesDataset
from system.controller.simulation.environment_cache import EnvironmentCache
from tqdm import tqdm
import numpy as np


def crosscheck_rc(
    res: tuple[ReachabilityController, ReachabilityController], samples: SampleGenerator,
    env_model: str,
    num_samples: int,
    flush_freq: int = 10,
) -> dict[tuple[bool, bool], int]:
    results = { (True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0 }

    for i in (bar := tqdm(range(num_samples))):
        sample = None
        while sample is None:
            try:
                sample, path_l, map_name = next(samples)
                assert map_name == env_model

                reachable = tuple(reachability_controller.reachable(sample.src, sample.dst, path_l) for reachability_controller in res)
            except (ValueError, AssertionError):
                tqdm.write(f"Error at sample {i}")
                continue
        results[reachable] += 1

        if i % flush_freq == (flush_freq - 1):
            bar.set_description('/'.join(map(str, results.values())))
    return results


def crosscheck_re(
    res: tuple[ReachabilityEstimator, ReachabilityEstimator], samples: SampleGenerator,
    num_samples: int,
):
    results = np.zeros((num_samples, 2), dtype=float)

    for i in tqdm(range(num_samples)):
        sample, path_l, map_name = next(samples)
        factors = tuple(re.reachability_factor(sample.src, sample.dst) for re in res)
        results[i] = factors
    return results


if __name__ == "__main__":
    """
    Check how much different methods for estimating reachability agree with each other.
    """
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('re_types',
        nargs=2,
        help='RE algorithms to compare',
    )
    parser.add_argument('mode', choices=['binary', 'distribution'])
    parser.add_argument('--re-from', help='File from which to take the reachability estimator', default='re_mse_weights.50')
    parser.add_argument('-n', '--num-samples', type=int, default=400, help='Number of samples on which to test')
    parser.add_argument('-w', '--wall-colors', help='how to color the walls', choices=['1color', '3colors', 'patterns'], default='1color')
    parser.add_argument('--gen', '--generate-point-pairs',
        choices=['same_traj', 'diff_traj', 'maybe_diff_traj', 'random', 'random_circle'], default='same_traj',
        help='How to generate pairs of points',
    )
    parser.add_argument('--traj_file', help='Dataset of trajectories used for the _traj point-pair generators', nargs='?', default='trajectories.hd5')
    args = parser.parse_args()

    if args.wall_colors == '1color':
        textures = [ os.path.join( 'yellow_wall.png') ]
    elif args.wall_colors == '3colors':
        from system.controller.simulation.pybullet_environment import all_possible_textures
        textures = all_possible_textures[:args.wall_colors]
    elif args.wall_colors == 'patterns':
        textures = lambda i : f'pattern-{i+1}.png'
    env_kwargs={ 'wall_kwargs': { 'textures': textures } }

    with EnvironmentCache(override_env_kwargs=env_kwargs) as env_cache:
        env_model = 'Savinov_val3'
        env = env_cache[env_model]
        res = [reachability_estimator_factory(retype, weights_file=args.re_from, env_model=env_model, env=env) for retype in args.re_types]

        if args.gen.endswith('_traj'):
            filename = os.path.join(get_path(), "data", "trajectories", args.traj_file)
            filename = os.path.realpath(filename)
            rd = TrajectoriesDataset([filename], env_cache=env_cache)
            samples = rd.iterate(mode=args.gen)
        else:
            if args.gen == 'random':
                from system.controller.reachability_estimator.data_generation.dataset import RandomSamples
                samples = RandomSamples(env)
            elif args.gen == 'random_circle':
                from system.controller.reachability_estimator.data_generation.dataset import RandomSamplesWithLimitedDistance
                samples = RandomSamplesWithLimitedDistance(env)
            else: raise ValueError(args.gen)

        if args.mode == 'binary':
            results = crosscheck_rc(
                res, samples,
                env_model=env_model,
                num_samples=args.num_samples,
            )
            print('\t2>True\t2>False\t|')
            for i, re1pred in enumerate([True, False]):
                print(f'1>{re1pred}', end='\t')
                for j, re2pred in enumerate([True, False]):
                    percentage = results[re1pred, re2pred] / args.num_samples
                    print(f'{results[re1pred, re2pred]} ({percentage:2.1f}%)', end='\t')
                print('|')

            print(f'Assuming the first RE ({repr(res[0])}) to be the ground truth:')
            precision = results[True, True] / (results[True, True] + results[False, True])
            recall = results[True, True] / (results[True, True] + results[True, False])
            print('Accuracy:', (results[True, True] + results[False, False]) / args.num_samples)
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1Score:', 2*precision*recall/(precision+recall))

        elif args.mode == 'distribution':
            results = crosscheck_re(
                res, samples, num_samples=args.num_samples
            )
            import matplotlib.pyplot as plt
            plt.scatter(results[:, 0], results[:, 1])
            plt.xlabel(args.re_types[0])
            plt.ylabel(args.re_types[1])
            plt.show()
        else:
            raise ValueError(args.mode)
