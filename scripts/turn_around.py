# coding: utf-8
from math import e
from system.controller.local_controller.local_controller import LocalController, controller_rules
from system.controller.local_controller.local_navigation import vector_navigation
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.local_controller.compass import Compass
from system.utils import normalize
import numpy as np

angle = np.radians(29.99)
visualize = False

class TurningCompass(Compass):
    def __init__(self):
        self.last_heading = None
        self.starting_heading = None
        self.turning_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        self.second_half_turn = False
    def update_position(self, robot):
        self.last_heading = normalize(robot.heading_vector())
        robot.env.add_debug_line(robot.position, np.array(robot.position) + self.last_heading, color=(1, 0, 0), width=3)
        robot.env.add_debug_line(robot.position, np.array(robot.position) + self.calculate_goal_vector(), color=(0, 0, 0), width=3)

    def reset_position(self, heading):
        self.starting_heading = normalize(heading)
        self.last_heading = self.starting_heading
        self.second_half_turn = False
    def reset_goal(self, heading):
        raise NotImplementedError("This will reach the goal when the agent has accomplished a full turn and not before")
    @property
    def arrival_threshold(self):
        if self.last_heading is None:
            return -1
        else:
            return 0.1
    def parse(self, pc: 'PlaceInfo'):
        return pc.angle
    def calculate_goal_vector(self):
        norm = np.dot(self.last_heading, self.starting_heading)
        print(f"{norm=}")
        if norm < -0.9 and self.second_half_turn is False:
            self.second_half_turn = True
        if self.second_half_turn is False:
            norm = 2.0
        else:
            norm = 1 - norm # norm will go from -1 to 1 -> map this to 2 -> 0
        return np.matmul(self.turning_matrix, np.array(self.last_heading)) * norm


if __name__ == "__main__":
    from system.plotting.plotResults import plotTrajectoryInEnvironment

    controller = LocalController()
    t = TurningCompass()

    with PybulletEnvironment("plane", visualize=visualize) as env:
        t.reset_position(env.robot.heading_vector())
        vector_navigation(env, compass=t, controller=controller)

        coordinates: list[np.ndarray] = env.robot.data_collector.xy_coordinates
    coordinates = np.stack(coordinates)
    min_, max_ = np.min(coordinates, axis=0), np.max(coordinates, axis=0)
    print(min_, max_)
    diameters = max_ - min_
    radii = diameters / 2
    mean_radius = sum(radii) / 2
    print(mean_radius)
    plotTrajectoryInEnvironment(env_model="plane", xy_coordinates=coordinates)
