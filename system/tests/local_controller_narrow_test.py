import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.controller.local_controller.compass import AnalyticalCompass
from system.controller.local_controller.local_navigation import vector_navigation
from system.controller.local_controller.local_controller import LocalController, controller_rules
import numpy as np

WALL_CENTER = np.array([-1, 0])


def width_test(
    width: float,
    controller: LocalController,
    env: PybulletEnvironment,
    start_distance,
    goal_distance,
) -> bool:
    env.switch_variant(width)

    angle = (-0.75, 1.25)
    start = (angle[0], angle[1] - start_distance)
    goal = (angle[0] + goal_distance, angle[0])
    compass = AnalyticalCompass(start, goal)
    with Robot(env, base_position=start):
        success, _ = vector_navigation(env, compass, controller=controller, goal_pos=goal)
    return success

max_fail_width = 0
min_succ_width = 2

import argparse
from system.parsers import controller_parser

parser = argparse.ArgumentParser(parents=[controller_parser])
parser.add_argument('precision', nargs='?', type=float, default=0.1, help='Precision, in meters')
parser.add_argument('--start-distance', '--start', type=float, default=2.5)
parser.add_argument('--goal-distance', '--goal', type=float, default=1.75)
parser.add_argument('--starting-value', '-s', type=float)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

controller = LocalController(
    on_reset_goal=[controller_rules.TurnToGoal()],
    transform_goal_vector=[
        controller_rules.ObstacleAvoidance(ray_length=args.ray_length, follow_walls=args.follow_walls, tactile_cone=np.radians(args.tactile_cone)),
    ],
    hooks=[controller_rules.StuckDetector()],
)

with PybulletEnvironment(env_model="obstacle_map_0", visualize=args.visualize, contains_robot=False) as env:
    while(min_succ_width - max_fail_width > args.precision):
        if args.starting_value is not None:
            width = args.starting_value
            args.starting_value = None
        else:
            width = (min_succ_width + max_fail_width) / 2

        print(f"Trying {width=}...", file=sys.stderr)
        success = width_test(width, controller, env, start_distance=args.start_distance, goal_distance=args.goal_distance)
        print("Success!" if success else "Fail :(", file=sys.stderr)
        if success:
            min_succ_width = width
        else:
            max_fail_width = width
    print(f"Minimum handle-able width is {max_fail_width} ~ {min_succ_width}")
