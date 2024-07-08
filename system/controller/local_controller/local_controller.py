import numpy as np
from system.utils import normalize
from system.controller.simulation.math_utils import vectors_in_one_direction, intersect

from abc import ABC
from system.types import Vector2D
from typing import List, Callable

ResetGoalHook = Callable[[Vector2D, 'Robot'], None]
TransformGoalHook = Callable[[Vector2D, 'Robot'], Vector2D]

class Hook:
    def on_reset_goal(self, new_goal: Vector2D, robot: 'Robot'):
        pass
    def transform_goal_vector(self, goal_vector: Vector2D, robot: 'Robot') -> Vector2D:
        return goal_vector

class LocalController(ABC):
    """
    Class that performs navigation.
    """
    def __init__(
        self,
        on_reset_goal : List[ResetGoalHook] = [],
        transform_goal_vector : List[TransformGoalHook] = [],
        hooks: list[Hook] = [],
    ):
        self.on_reset_goal = on_reset_goal + [hook.on_reset_goal for hook in hooks]
        self.transform_goal_vector = transform_goal_vector + [hook.transform_goal_vector for hook in hooks]

    @staticmethod
    def default(obstacles=True):
        return LocalController(on_reset_goal=[ TurnToGoal() ], transform_goal_vector=([ObstacleAvoidance()] if obstacles else []), hooks=[StuckDetector()])

    def reset_goal(self, new_goal : Vector2D, robot: 'Robot'):
        for hook in self.on_reset_goal:
            hook(new_goal, robot)

    def step(self, goal_vector: Vector2D, robot: 'Robot'):
        for hook in self.transform_goal_vector:
            goal_vector = hook(goal_vector, robot)

        robot.env.add_debug_line(robot.position, np.array(robot.position) + goal_vector, color=(0, 0, 1), width=3)
        robot.navigation_step(goal_vector)


class ObstacleAvoidance:
    def __init__(
        self,
        combine = 1.5,
        num_ray_dir = 21,
        tactile_cone = 120,
        ray_length = 1,
        follow_walls = False,
    ):
        self.combine = combine
        self.follow_walls = follow_walls
        self.lidar_kwargs = dict(tactile_cone=tactile_cone, num_ray_dir=num_ray_dir, ray_length=ray_length)

    def __call__(self, goal_vector : Vector2D, robot: 'Robot') -> Vector2D:
        lidar = robot.env.lidar(**self.lidar_kwargs, blind_spot_cone=0, agent_pos_orn=robot.lidar_sensor_position)
        point, obstacle_vector = robot.calculate_obstacle_vector(lidar)
        #print(f"{self.position=}, {obstacle_vector=}, {goal_vector=}")
        robot.env.add_debug_line(robot.position, np.array(robot.position) + obstacle_vector, color=(1, 0, 0), width=3)
        robot.env.add_debug_line(robot.position, np.array(robot.position) + goal_vector, color=(0, 0, 0), width=3)

        normed_goal_vector = normalize(goal_vector)

        # combine goal and obstacle vector
        multiple = 1 if vectors_in_one_direction(normed_goal_vector, obstacle_vector) else -1
        if not intersect(robot.position, normed_goal_vector, point, obstacle_vector * multiple):
            multiple = 0
            print("multiple is 0 (no intersection)", end='\r')
        else:
            print("   ~", end='\r')
        combine = self.combine
        if self.follow_walls and multiple == 1 and np.linalg.norm(obstacle_vector) > 3: # norm is 1.5 / min(distances), i.e. if norm > 3 <=> distance < 0.5
            combine = 0
        goal_vector = normed_goal_vector * combine + obstacle_vector * multiple
        return goal_vector


class ObstacleBackoff:
    def __init__(self, backoff_on_distance = 0.4, backoff_off_distance = 0.4):
        self.backoff_on_distance = backoff_on_distance
        self.backoff_off_distance = backoff_off_distance
        self.backing_off = False
        assert self.backoff_on_distance <= self.backoff_off_distance
    def __call__(self, goal_vector: Vector2D, robot: 'Robot') -> Vector2D:
        distances = robot.env.straight_lidar(
            radius=0.18,
            num_ray_dir=3,
            agent_pos_orn=robot.lidar_sensor_position,
            ray_length=(self.backoff_off_distance if self.backing_off else self.backoff_on_distance),
            draw_debug_lines=True
        )
        if any(distance != -1 for distance in distances):
            self.backing_off = True
        else:
            self.backing_off = False
        if self.backing_off:
            return - np.array(goal_vector)
        else:
            return goal_vector


class RobotStuck(Exception):
    pass

class StuckDetector(Hook):
    def __init__(self, stuck_threshold = 200):
        self.stuck_threshold = stuck_threshold
        self.nr_ofsteps = 0
        self.previous_position = None
    def on_reset_goal(self, new_goal, robot):
        self.nr_ofsteps = 0
    def transform_goal_vector(self, goal_vector: Vector2D, robot: 'Robot') -> Vector2D:
        if self.previous_position is not None and np.linalg.norm(robot.position - self.previous_position) < 0.001:
            self.nr_ofsteps += 1
            if self.nr_ofsteps >= self.stuck_threshold:
                raise RobotStuck()
        else:
            self.nr_ofsteps = 0

        self.previous_position = robot.position
        return goal_vector

class TurnToGoal:
    def __init__(self, tolerance = 0.05):
        self.tolerance = tolerance
    def __call__(self, goal_vector : Vector2D, robot: 'Robot'):
        ray_length = 0.25
        while True:
            collision_data, _ = robot.env.lidar(
                tactile_cone=np.radians(180),
                num_ray_dir=20,
                blind_spot_cone=0,
                agent_pos_orn=robot.lidar_sensor_position,
                ray_length=ray_length,
                draw_debug_lines=True
            )
            if all(distance == -1 for distance in collision_data.distances):
                break
            robot.navigation_step(goal_vector=-np.array(robot.heading_vector()))
        while True:
            collision_data, _ = robot.env.lidar(
                tactile_cone=np.radians(360),
                num_ray_dir=20,
                blind_spot_cone=np.radians(180),
                agent_pos_orn=robot.lidar_sensor_position,
                ray_length=ray_length,
                draw_debug_lines=True
            )
            if all(distance == -1 for distance in collision_data.distances):
                break
            robot.navigation_step(goal_vector=np.array(robot.heading_vector()))

        robot.turn_to_goal(goal_vector, tolerance=self.tolerance)
