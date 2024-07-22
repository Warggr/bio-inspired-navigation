import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.controller.local_controller.compass import Compass
from system.controller.local_controller.local_navigation import vector_navigation
from system.controller.local_controller.local_controller import LocalController, controller_rules
import numpy as np
from system.types import Angle, Vector2D
from typing import Optional

WALL_CENTER = np.array([-1, 0])

class DirectionalCompass(Compass[float]):
    """
    Rather than navigating to a certain point, navigates towards a certain line,
    i.e. always provides the normal vector to that line as a goal vector.
    """
    arrival_threshold = 0.01

    def zvalue(self, p: Vector2D):
        return np.dot(self.goal_vector, p)

    def __init__(self, angle: Angle, goal_offset: float, zero: Vector2D = np.array([0, 0])):
        """
        angle: the angle of the line
        offset: the distance of this line to the origin
        zero: an alternative origin
        """
        self.goal_vector = np.array([ -np.sin(angle), -np.cos(angle) ])
        zero_offset = self.zvalue(zero)
        self.goal = goal_offset - zero_offset
        self.position: Optional[float] = None
    def calculate_goal_vector(self) -> Vector2D:
        return self.goal_vector * (self.goal - self.position)
    def parse(pc: 'PlaceInfo') -> float:
        return self.zvalue(pc.pos)
    def update_position(self, robot: 'Robot'):
        self.position = self.zvalue(robot.position)
    def reset_position(self, new_position: float):
        self.position = new_position
    def reset_goal(self, new_goal: float):
        self.goal = new_goal

def angle_test(
    angle: Angle,
    controller: LocalController,
    env: PybulletEnvironment,
) -> bool:
    goal_direction = np.array([ -np.sin(angle), -np.cos(angle) ])
    start = WALL_CENTER - 2*goal_direction
    compass = DirectionalCompass(angle, goal_offset=2.5, zero=WALL_CENTER)
    compass.reset_position(compass.zvalue(start))
    with Robot(env, base_position=start, base_orientation=-angle-np.radians(90)):
        success, _ = vector_navigation(env, compass, controller=controller)
    return success

min_fail_angle = np.radians(90)
max_succ_angle = np.radians(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('precision', nargs='?', type=float, default=5, help='Precision, in degrees')
parser.add_argument('--ray-length', default=1, type=float)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

precision = np.radians(args.precision)
controller = LocalController(
    on_reset_goal=[],
    transform_goal_vector=[
        controller_rules.ObstacleAvoidance(ray_length=args.ray_length),
    ],
    hooks=[controller_rules.StuckDetector()],
)

with PybulletEnvironment(env_model="obstacle_map_2", visualize=args.visualize, contains_robot=False) as env:
    while(min_fail_angle - max_succ_angle > precision):
        angle = (max_succ_angle + min_fail_angle) / 2
        print(f"Trying angle={np.degrees(angle)}...", file=sys.stderr)
        success = angle_test(angle, controller, env)
        print("Success!" if success else "Fail :(", file=sys.stderr)
        if success:
            max_succ_angle = angle
        else:
            min_fail_angle = angle
    print(f"Maximum handle-able angle is {np.degrees(max_succ_angle)} ~ {np.degrees(min_fail_angle)}")
