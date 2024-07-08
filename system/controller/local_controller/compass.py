from system.types import Vector2D
from abc import ABC, abstractmethod, abstractmethod
import numpy as np
from typing import Optional, Generic, TypeVar

PositionRepresentation = TypeVar('PositionRepresentation')

class Compass(ABC, Generic[PositionRepresentation]):
    """
    The Compass class tracks the current position and goal position, and allows to compute goal vectors.
    """

    @staticmethod
    @abstractmethod
    def parse(pc: 'PlaceInfo') -> PositionRepresentation:
        ...

    @property
    @abstractmethod
    def arrival_threshold(self) -> float:
        """ threshold for goal_vector length that signals arrival at goal """
        ...

    @abstractmethod
    def calculate_goal_vector(self) -> Vector2D:
        """ Computes the goal vector based on the stored current and goal position """
        ...

    @abstractmethod
    def update_position(self, robot: 'Robot'):
        """ Updates the stored position """
        ...

    @abstractmethod
    def reset_position(self, new_position: PositionRepresentation):
        ...

    @abstractmethod
    def reset_goal(self, new_goal: PositionRepresentation):
        """ Sets a new goal position """
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
        if robot.buildDataSet:
            assert len(robot.data_collector.images) > 0
        self.update_position(robot)
        return self.reached_goal()

    @staticmethod
    def factory(mode, *args, **kwargs) -> 'Compass':
        if mode == "analytical":
            return AnalyticalCompass()
        else:
            from system.controller.local_controller.local_navigation import GcCompass
            return GcCompass.factory(mode, *args, **kwargs)


class AnalyticalCompass(Compass[Vector2D]):
    """ Uses a precise goal vector. """

    arrival_threshold = 0.1

    @staticmethod
    def parse(pc: 'PlaceInfo'):
        return pc.pos

    def __init__(self, start_pos: Optional[Vector2D] = None, goal_pos: Optional[Vector2D] = None):
        self.goal_pos = goal_pos
        self.current_pos = start_pos

    def update_position(self, robot : 'Robot'):
        self.current_pos = robot.position

    def calculate_goal_vector(self) -> Vector2D:
        return np.array(self.goal_pos) - np.array(self.current_pos)

    def reset_position(self, new_position: Vector2D):
        self.current_pos = new_position

    def reset_goal(self, new_goal: Vector2D):
        self.goal_pos = new_goal
