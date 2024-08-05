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
) -> bool:
    env.switch_variant(width)

    start = (-1.0, -1.33)
    goal = (1.0, 1.4)
    compass = AnalyticalCompass(start, goal)
    with Robot(env, base_position=start):
        success, _ = vector_navigation(env, compass, controller=controller, goal_pos=goal)
    return success

min_fail_width = -2
max_succ_width = 2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('precision', nargs='?', type=float, default=0.1, help='Precision, in meters')
parser.add_argument('--ray-length', default=1, type=float)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

controller = LocalController(
    on_reset_goal=[controller_rules.TurnToGoal()],
    transform_goal_vector=[
        controller_rules.ObstacleAvoidance(ray_length=args.ray_length),
    ],
    hooks=[controller_rules.StuckDetector()],
)

with PybulletEnvironment(env_model="obstacle_map_0", visualize=args.visualize, contains_robot=False) as env:
    while(max_succ_width - min_fail_width > args.precision):
        width = (max_succ_width + min_fail_width) / 2
        print(f"Trying {width=}...", file=sys.stderr)
        success = width_test(width, controller, env)
        print("Success!" if success else "Fail :(", file=sys.stderr)
        if success:
            max_succ_width = width
        else:
            min_fail_width = width
    print(f"Minimum handle-able width is {min_fail_width} ~ {max_succ_width}")
