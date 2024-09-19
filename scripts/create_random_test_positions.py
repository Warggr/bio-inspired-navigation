# coding: utf-8
from system.controller.simulation.environment.map_occupancy import environment_dimensions
from system.controller.reachability_estimator.data_generation.dataset import TrajectoriesDataset, get_path
from system.types import PositionAndOrientation
import random
import math
import os
from typing import Generator, Iterator

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['random_grid', 'traj'])
parser.add_argument('-n', help='Number of samples', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

rng = random.Random()
rng.seed(args.seed)

def random_position_on_grid(rng: random.Random):
    (xmin, ymin), (width, height) = environment_dimensions('Savinov_val3')
    return [(rng.randint(0, height) + 0.5 + xmin, rng.randint(0, width) + 0.5 + ymin) for _ in range(2)]

def random_positions_and_angles(rng: random.Random):
    p0, p1 = random_position_on_grid(rng=rng)
    a0, a1 = [2 * math.pi * rng.random() for _ in range(2)]
    return (p0, a0), (p1, a1)

def random_positions_from_traj(rng: random.Random) -> Generator[tuple[PositionAndOrientation, PositionAndOrientation], None, None]:
    dataset = TrajectoriesDataset([os.path.join(get_path(), "data", "trajectories", "trajectories.hd5")], env_cache=None)
    samepath_points = dataset.subset(map_name="Savinov_val3", seed=1)
    while True:
        yield rng.choice(samepath_points)

points: Iterator[tuple[PositionAndOrientation, PositionAndOrientation]]
if args.mode == 'random_grid':
    points = iter(lambda: random_positions_and_angles(rng), None)
elif args.mode == 'traj':
    points = random_positions_from_traj(rng)
else:
    raise ValueError(args.mode)

def format(p: 'PositionAndOrientation') -> str:
    return ','.join(map(str, [p[0][0], p[0][1], p[1]]))

for _ in range(args.n):
    (p0, a0), (p1, a1) = next(points)
    print("Savinov_val3", format((p0, a0)), format((p1, a1)))
