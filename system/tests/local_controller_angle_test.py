import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.controller.local_controller.compass import AnalyticalCompass
from system.controller.local_controller.local_navigation import vector_navigation
from system.controller.local_controller.local_controller import LocalController, controller_rules
import numpy as np
from system.types import Angle

WALL_CENTER = np.array([-1, 0])

def angle_test(
    angle: Angle,
    controller: LocalController,
    env: PybulletEnvironment,
) -> bool:
    goal_direction = np.array([ -np.sin(angle), -np.cos(angle) ])
    goal = WALL_CENTER + 3*goal_direction
    start = WALL_CENTER - 2*goal_direction
    compass = AnalyticalCompass(start, goal)
    with Robot(env, base_position=start, base_orientation=-angle-np.radians(90)):
        success, _ = vector_navigation(env, compass, controller=controller)
    return success

min_fail_angle = np.radians(90)
max_succ_angle = np.radians(0)

visualize = False
precision = np.radians(5)
controller = LocalController(
    on_reset_goal=[],
    transform_goal_vector=[controller_rules.ObstacleAvoidance(follow_walls=True)],
    hooks=[controller_rules.StuckDetector()],
)

with PybulletEnvironment(env_model="obstacle_map_2", visualize=visualize, contains_robot=False) as env:
    while(min_fail_angle - max_succ_angle > precision):
        angle = (max_succ_angle + min_fail_angle) / 2
        print(f"Trying angle={np.degrees(angle)}...")
        success = angle_test(angle, controller, env)
        print("Success!" if success else "Fail :(")
        if success:
            max_succ_angle = angle
        else:
            min_fail_angle = angle
