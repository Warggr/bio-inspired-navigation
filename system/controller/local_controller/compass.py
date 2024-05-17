from system.types import Vector2D
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from typing import Optional

class Compass(ABC):
    def __init__(self, goal_pos : Optional[Vector2D] = None, stuck_threshold : int = 200):
        """
        Creates a Compass that gives a goal position to the agent.
        arrival_threshold -- threshold for goal_vector length that signals arrival at goal
        goal_pos          -- position of the goal (can be set to None, or reset later).
                             used for analytical goal vector calculation and plotting
        stuck_threshold   -- threshold for considering the agent as stuck

        """
        self.goal_pos = goal_pos
        self.stuck_threshold = stuck_threshold
        self.nr_ofsteps = 0 # keeps track of number of steps taken with current decoder (used for switching between pod and linlook decoder)

    @abstractproperty
    @property
    def arrival_threshold(self):
        ...

    @abstractmethod
    def calculate_goal_vector(self, robotPosition : Vector2D) -> Vector2D:
        ...

    def update(self, robot : 'Robot') -> bool:
        '''
        Updates the Compass position to the newest position of the Robot.
        Returns True if the goal was reached, False otherwise
        '''
        goal_vector = self.calculate_goal_vector(robot.position)

        if np.linalg.norm(goal_vector) < self.arrival_threshold:
            return True

        stop = self.stuck_threshold
        self.nr_ofsteps += 1
        #if self.buffer + stop < len(self.data_collector.data) and stop < len(self.data_collector.data):
        if self.nr_ofsteps > stop:
            if np.linalg.norm(self.goal_pos - robot.position) < 0.1:
                raise robot.Stuck()

        # Still going
        return False

    def step(self, robot : 'Robot', *args, **kwargs) -> bool:
        goal_vector = self.calculate_goal_vector(robot.position)
        if np.linalg.norm(goal_vector) == 0:
            return False
        robot.navigation_step(goal_vector, *args, **kwargs)
        return self.update(robot)

    @staticmethod
    def factory(mode, *args, **kwargs) -> 'Compass':
        if mode == "analytical":
            return AnalyticalCompass()
        else:
            from system.controller.local_controller.local_navigation import GcCompass
            return GcCompass.factory(mode, *args, **kwargs)


class AnalyticalCompass(Compass):
    """ Uses a precise goal vector. """

    arrival_threshold = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(stuck_threshold=100, *args, **kwargs)

    def calculate_goal_vector(self, robotPosition : Vector2D) -> Vector2D:
        goal_vector = [self.goal_pos[0] - robotPosition[0], self.goal_pos[1] - robotPosition[1]]

        return goal_vector
