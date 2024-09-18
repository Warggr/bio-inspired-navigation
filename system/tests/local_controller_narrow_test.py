import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.controller.local_controller.compass import AnalyticalCompass
from system.controller.local_controller.local_navigation import vector_navigation
from system.controller.local_controller.local_controller import LocalController, controller_rules
import numpy as np

WALL_CENTER = np.array([-1, 0])


def width_test(
    controller: LocalController,
    env: PybulletEnvironment,
    *,
    width: float,
    start_distance,
    goal_distance,
) -> bool:
    env.switch_variant(width)

    corner = (-0.75, 1.25)
    start = (corner[0], corner[1] - start_distance)
    goal = (corner[0] + goal_distance, corner[1])
    compass = AnalyticalCompass(start, goal)
    with Robot(env, base_position=start):
        success, _ = vector_navigation(env, compass, controller=controller, goal_pos=goal)
    return success

import argparse
from system.parsers import controller_creator, controller_parser

parser = argparse.ArgumentParser(parents=[controller_parser])
parser.add_argument('vary', nargs='?', choices=['width', 'start_distance', 'goal_distance'], default='width')
parser.add_argument('precision', nargs='?', type=float, default=0.1, help='Precision, in meters')
parser.add_argument('--width', type=float)
parser.add_argument('--start-distance', '--start', type=float)
parser.add_argument('--goal-distance', '--goal', type=float)
parser.add_argument('--starting-value', '-s', type=float)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

bounds = {
    'width': (0, 2),
    'start_distance': (0, 3),
    'goal_distance': (0, 3),
}
defaults = {
    'width': 1.0,
    'start_distance': 2.5,
    'goal_distance': 1.75,
}
large_is_easy = {'width': True, 'start_distance': False, 'goal_distance': False}

controller = controller_creator(args, env_model="obstacle_map_0")

min_value, max_value = bounds[args.vary]
assert getattr(args, args.vary) is None, f"{args.vary} both set and provided as variable"
kwargs = {key: getattr(args, key) for key in defaults}
kwargs = {k: v if v is not None else defaults[k] for k, v in kwargs.items()}
large_is_easy = large_is_easy[args.vary]
bounds = {
    'width': (0, 2),
    'start_distance': (0, 3),
    'goal_distance': (0, 3),
}
defaults = {
    'width': 1.0,
    'start_distance': 2.5,
    'goal_distance': 1.75,
}
large_is_easy = {'width': True, 'start_distance': False, 'goal_distance': False}

controller = controller_creator(args)

min_value, max_value = bounds[args.vary]
assert getattr(args, args.vary) is None, f"{args.vary} both set and provided as variable"
kwargs = {key: getattr(args, key) for key in defaults}
kwargs = {k: v if v is not None else defaults[k] for k, v in kwargs.items()}
large_is_easy = large_is_easy[args.vary]

with PybulletEnvironment(env_model="obstacle_map_0", visualize=args.visualize, contains_robot=False) as env:
    while(max_value - min_value > args.precision):
        if args.starting_value is not None:
            value = args.starting_value
            args.starting_value = None
        else:
            value = (min_value + max_value) / 2

        print(f"Trying {args.vary}={value}...", file=sys.stderr)
        kwargs[args.vary] = value

        success = width_test(controller, env, **kwargs)
        print("Success!" if success else "Fail :(", file=sys.stderr)
        if (success and not large_is_easy) or (not success and large_is_easy):
            min_value = value
        else:
            max_value = value
    print(f"M{'in' if large_is_easy else 'ax'}imum handle-able {args.vary} is {min_value} ~ {max_value}")
