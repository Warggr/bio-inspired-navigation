from system.types import Vector2D
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from typing import Optional

class Compass(ABC):
    """
    The Compass class tracks the current position and goal position, and allows to compute goal vectors.
    """

    def __init__(self, goal_pos : Optional[Vector2D] = None):
        """
        Creates a Compass that gives a goal position to the agent.
        arrival_threshold -- threshold for goal_vector length that signals arrival at goal
        goal_pos          -- position of the goal (can be set to None, or reset later).
                             used for analytical goal vector calculation and plotting
        stuck_threshold   -- threshold for considering the agent as stuck

        """
        self.goal_pos = goal_pos

    @abstractproperty
    @property
    def arrival_threshold(self):
        ...

    @abstractmethod
    def calculate_goal_vector(self) -> Vector2D:
        """ Computes the goal vector based on the stored current and goal position """
        ...

    @abstractmethod
    def update_position(self, robot : 'Robot'):
        """ Updates the stored position """
        ...

    def reset(self, new_goal : Vector2D):
        """ Sets a new goal position """
        print("Resetting goal_pos to", new_goal)
        self.goal_pos = new_goal

    def reached_goal(self) -> bool:
        '''
        Updates the Compass position to the newest position of the Robot.
        Returns True if the goal was reached, False otherwise
        '''
        goal_vector = self.calculate_goal_vector()

        return np.linalg.norm(goal_vector) < self.arrival_threshold

    def step(self, robot : 'Robot', *args, **kwargs) -> bool:
        goal_vector = self.calculate_goal_vector()
        if np.linalg.norm(goal_vector) == 0:
            return True
        robot.navigation_step(goal_vector, *args, **kwargs)
        self.update_position(robot)
        return self.reached_goal()

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

    def __init__(self, start_pos : Optional[Vector2D] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_pos : Vector2D = start_pos

    def update_position(self, robot : 'Robot'):
        self.current_pos = robot.position

    def calculate_goal_vector(self) -> Vector2D:
        goal_vector = [self.goal_pos[0] - self.current_pos[0], self.goal_pos[1] - self.current_pos[1]]

        return goal_vector
