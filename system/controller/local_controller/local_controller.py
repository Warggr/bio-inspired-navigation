import numpy as np
from system.controller.simulation.pybullet_environment import Robot, RobotStuck
from system.utils import normalize
from system.controller.simulation.math_utils import vectors_in_one_direction, intersect, compute_angle
from system.debug import DEBUG

from abc import ABC
from system.types import Vector2D, Angle, AllowedMapName
from typing import Callable

ResetGoalHook = Callable[[Vector2D, Robot], None]
TransformGoalHook = Callable[[Vector2D, Robot], Vector2D|tuple[Vector2D, dict]]


class Hook:
    def on_reset_goal(self, new_goal: Vector2D, robot: Robot):
        pass
    def transform_goal_vector(self, goal_vector: Vector2D, robot: Robot) -> Vector2D:
        return goal_vector


class LocalController(ABC):
    """
    Class that performs navigation.
    """
    def __init__(
        self,
        on_reset_goal: list[ResetGoalHook] = [],
        transform_goal_vector: list[TransformGoalHook] = [],
        hooks: list[Hook] = [],
    ):
        self.on_reset_goal = on_reset_goal + [hook.on_reset_goal for hook in hooks]
        self.transform_goal_vector = transform_goal_vector + [hook.transform_goal_vector for hook in hooks]

    @staticmethod
    def default(obstacles=True):
        return LocalController(on_reset_goal=[TurnToGoal()], transform_goal_vector=([ObstacleAvoidance()] if obstacles else []), hooks=[StuckDetector()])

    def reset_goal(self, new_goal: Vector2D, robot: Robot):
        for hook in self.on_reset_goal:
            hook(new_goal, robot)

    def step(self, goal_vector: Vector2D, robot: Robot):
        all_kwargs = {}
        for hook in self.transform_goal_vector:
            match hook(goal_vector, robot):
                case (goal_vector, kwargs):
                    pass
                case goal_vector:
                    kwargs = {}
            all_kwargs = { **all_kwargs, **kwargs } # TODO Pierre: this is ugly

        #robot.env.add_debug_line(robot.position, np.array(robot.position) + goal_vector, color=(0, 0, 1), width=3)
        if np.linalg.norm(goal_vector) != 0:
            robot.navigation_step(goal_vector, **all_kwargs)
        else:
            print('Warning: <0, 0> goal vector -> hope this doesn\'t happen too often')


class FajenObstacleAvoidance:
    """
    A biologically inspired local navigation algorithm (inspired by humans).
    See: Fajen, B.R., Warren, W.H., Temizer, S. et al. A Dynamical Model of Visually-Guided Steering, Obstacle Avoidance, and Route Selection. International Journal of Computer Vision 54, 13–34 (2003). https://doi.org/10.1023/A:1023701300169
    """
    def __init__(self, b = 3.25, kg = 7.5, c1 = 0.40, c2 = 0.20, ko = 198, c3 = 6.5, c4 = 0.8):
        self.b = b
        self.kg = kg
        self.c1 = c1
        self.c2 = c2
        self.ko = ko
        self.c3 = c3
        self.c4 = c4
        self.angle_velocity = 0

    def __call__(self, goal_vector: Vector2D, robot: Robot) -> Vector2D:
        dX = -self.b * self.angle_velocity - self.kg * (compute_angle(robot.heading_vector(), goal_vector)) * (np.exp(-self.c1 * np.linalg.norm(goal_vector)) + self.c2)
        lidar, _ = robot.env.lidar(tactile_cone=120, num_ray_dir=13, ray_length=2)
        for angle, distance in zip(lidar.angles, lidar.distances):
            if distance != -1:
                dX += self.ko * (angle) * (np.exp(-self.c3 * abs(angle))) * np.exp(-self.c4 * distance)
        self.angle_velocity += dX * robot.env.dt
        angle = robot.position_and_angle[1] + self.angle_velocity * robot.env.dt
        return np.array([ np.cos(angle), np.sin(angle) ])


class ObstacleAvoidance:
    def __init__(
        self,
        combine = 1.5,
        num_ray_dir = 21,
        tactile_cone = np.radians(120),
        ray_length = 1,
        follow_walls = False,
    ):
        self.combine = combine
        self.follow_walls = follow_walls
        self.lidar_kwargs = dict(tactile_cone=tactile_cone, num_ray_dir=num_ray_dir, ray_length=ray_length)
        self.debug = 'obstacles' in DEBUG

    def __call__(self, goal_vector: Vector2D, robot: Robot) -> Vector2D:
        lidar = robot.env.lidar(**self.lidar_kwargs, blind_spot_cone=0, agent_pos_orn=robot.lidar_sensor_position)
        point, obstacle_vector = robot.calculate_obstacle_vector(lidar)
        #print(f"{self.position=}, {obstacle_vector=}, {goal_vector=}")
        if self.debug:
            robot.env.add_debug_line(robot.position, np.array(robot.position) + obstacle_vector, color=(1, 0, 0), width=3)
            robot.env.add_debug_line(robot.position, np.array(robot.position) + goal_vector, color=(0, 0, 0), width=3)

        goal_vector_norm = np.linalg.norm(goal_vector)
        if np.dot(goal_vector, np.array(point[:2]) - np.array(robot.position)) / goal_vector_norm >= goal_vector_norm:
            # the goal is closer than the obstacle
            return goal_vector

        normed_goal_vector = normalize(goal_vector)

        # combine goal and obstacle vector
        multiple = 1 if vectors_in_one_direction(normed_goal_vector, obstacle_vector) else -1
        if not intersect(robot.position, normed_goal_vector, point, obstacle_vector * multiple):
            multiple = 0
        combine = self.combine
        if self.follow_walls and multiple == 1 and np.linalg.norm(obstacle_vector) > 3: # norm is 1.5 / min(distances), i.e. if norm > 3 <=> distance < 0.5
            combine = 0
        goal_vector = normed_goal_vector * combine + obstacle_vector * multiple
        if self.debug:
            robot.env.add_debug_line(robot.position, np.array(robot.position) + goal_vector, color=(0, 0, 1), width=3)
        #return normalize(goal_vector)
        return goal_vector


class ObstacleBackoff:
    def __init__(self, backoff_on_distance = 0.4, backoff_off_distance = 0.4):
        self.backoff_on_distance = backoff_on_distance
        self.backoff_off_distance = backoff_off_distance
        self.backing_off = False
        assert self.backoff_on_distance <= self.backoff_off_distance
    def __call__(self, goal_vector: Vector2D, robot: Robot) -> Vector2D|tuple[Vector2D,dict]:
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
            return - np.array(goal_vector), {'allow_backwards': True}
        else:
            return goal_vector


class StuckDetector(Hook):
    def __init__(self, stuck_threshold = 200):
        self.stuck_threshold = stuck_threshold
        self.nr_ofsteps = 0
        self.previous_position = None
    def on_reset_goal(self, new_goal, robot):
        self.nr_ofsteps = 0
    def transform_goal_vector(self, goal_vector: Vector2D, robot: Robot) -> Vector2D:
        if self.previous_position is not None and np.linalg.norm(robot.position - self.previous_position) < 0.001:
            self.nr_ofsteps += 1
            if self.nr_ofsteps >= self.stuck_threshold:
                raise RobotStuck()
        else:
            self.nr_ofsteps = 0

        self.previous_position = robot.position
        return goal_vector


class TurnWhenNecessary:
    def __call__(self, goal_vector: Vector2D, robot: Robot):
        current_heading = robot.heading_vector()
        diff_angle = compute_angle(current_heading, goal_vector)
        if abs(diff_angle) > np.radians(90) and np.linalg.norm(goal_vector) > 0.5:
            robot.turn_to_goal(goal_vector)
        if abs(diff_angle) > np.radians(90):
            print(np.linalg.norm(goal_vector))
        return goal_vector


class TurnToGoal:
    def __init__(self, tolerance = 0.05):
        self.tolerance = tolerance
    def __call__(self, goal_vector: Vector2D, robot: Robot):
        ray_length = 0.28
        """
        for angle in map(np.radians, [-45, 45, 135, -135]):
            for _ in range(100):
                agent_pos, agent_heading = robot.lidar_sensor_position
                world_angle = angle + agent_heading
                backoff_vector = -np.array([ np.cos(world_angle), np.sin(world_angle) ])
                collision_data, _ = robot.env.lidar(
                    tactile_cone=np.radians(90),
                    num_ray_dir=20,
                    blind_spot_cone=0,
                    agent_pos_orn=(agent_pos, world_angle),
                    ray_length=ray_length,
                    draw_debug_lines=True
                )
                if all(distance == -1 for distance in collision_data.distances):
                    break
                robot.env.add_debug_line(robot.position, np.array(robot.position) + backoff_vector, (0, 1, 0))
                robot.navigation_step(goal_vector=backoff_vector, allow_backwards=True)
            else: # no break encountered, i.e. timeout
                raise RobotStuck()
        """

        try:
            robot.turn_to_goal(goal_vector, tolerance=self.tolerance, report_stuck=True)
            return
        except RobotStuck:
            pass
        try:
            for _ in range(40): # 40 is, empirically, the number of steps needed to get away from an obstacle
                robot.navigation_step(-np.array(robot.heading_vector()), allow_backwards=True)
            goal_vector += 0.16 * np.array(robot.heading_vector()) # 0.16 is, also empirically, the distance it goes during that time
            robot.turn_to_goal(goal_vector, tolerance=self.tolerance, report_stuck=True)
            return
        except RobotStuck:
            pass
        try:
            for _ in range(40):
                robot.navigation_step(+np.array(robot.heading_vector()))
            goal_vector -= 0.16 * np.array(robot.heading_vector())
            robot.turn_to_goal(goal_vector, tolerance=self.tolerance, report_stuck=True)
            return
        except RobotStuck:
            pass
        raise RobotStuck()


class OptimalController(Hook):
    def __init__(self, env_model: AllowedMapName, allowed_angle = np.radians(60)):
        from system.controller.simulation.environment.map_occupancy import MapLayout
        self.map = MapLayout(env_model)
        self.allowed_angle = allowed_angle

    def transform_goal_vector(self, goal_vector: Vector2D, robot: Robot) -> Vector2D:
        start = np.array(robot.position)
        i = 1.0
        for i in range(10, 0, -1):
            goal = start + (i/10) * np.array(goal_vector)
            if self.map.suitable_position_for_robot(start) and self.map.suitable_position_for_robot(goal):
                try:
                    current_pos, next_waypoint, *_ignored = self.map.find_path(start, goal)
                    # assert current_pos == start
                    break
                except ValueError:
                    continue
            else:
                continue
        else: #not suitable position found
            return goal_vector

        new_goal_vector = np.array(next_waypoint) - start
        new_goal_vector /= np.linalg.norm(new_goal_vector)
        angle_to_goal = np.arccos(np.dot(new_goal_vector, goal_vector) / np.linalg.norm(goal_vector) / np.linalg.norm(new_goal_vector))
        if angle_to_goal <= self.allowed_angle:
            return new_goal_vector
        else:
            return goal_vector


class controller_rules: # namespace
    ObstacleAvoidance = ObstacleAvoidance
    FajenObstacleAvoidance = FajenObstacleAvoidance
    ObstacleBackoff = ObstacleBackoff
    StuckDetector = StuckDetector
    TurnToGoal = TurnToGoal
    TurnWhenNecessary = TurnWhenNecessary
    OptimalController = OptimalController
