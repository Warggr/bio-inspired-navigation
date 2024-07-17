""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""

''' Egocentric Ray Detection from:
***************************************************************************************
*    Title: "Biologically Plausible Spatial Navigation Based on Border Cells"
*    Author: "Camillo Heye"
*    Date: 28.08.2021
*    Availability: https://drive.google.com/file/d/1RvmLd5Ee8wzNFMbqK-7jG427M8KlG4R0/view
*
***************************************************************************************
'''
''' Camera and Object Loading from:
***************************************************************************************
*    Title: "Simulate Images for ML in PyBullet — The Quick & Easy Way"
*    Author: "Mason McGough"
*    Date: 19.08.2019
*    Availability: https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
*
***************************************************************************************
'''
''' Keyboard Movement from:
***************************************************************************************
*    Title: "pyBulletIntro"
*    Author: "Ramin Assadollahi"
*    Date: 16.05.2021
*    Availability: https://github.com/assadollahi/pyBulletIntro
*
***************************************************************************************
'''
import pybullet as p
import time
import os
import sys
import pybullet_data
import numpy as np
import math
import itertools
from typing import List, Optional, Any, Tuple, Callable, Self
from random import Random

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import system.plotting.plotResults as plot
from system.bio_model.grid_cell_model import GridCellNetwork

from system.controller.simulation.math_utils import vectors_in_one_direction, intersect, compute_angle
from system.types import types, LidarReading, Vector2D
from system.controller.simulation.environment_config import environment_dimensions

try:
    itertools.batched
except AttributeError:
    from system.polyfill import batched
    itertools.batched = batched

def closest_subsegment(values : List[float]) -> Tuple[int, int]:
    values = np.array(values)
    if not np.any(values >= 0):
        return -1, -1
    minimal_positive = np.min(values[np.where(values >= 0)])
    index = np.where(values == minimal_positive)[0][0]
    start_index = index
    while start_index > 0 and values[start_index - 1] >= 0:
        start_index -= 1
    end_index = index
    while end_index < len(values) - 1 and values[end_index + 1] >= 0:
        end_index += 1
    return start_index, end_index

DIRNAME = os.path.dirname(__file__)
DEBUG = os.getenv('DEBUG', False)

def resource_path(*filepath):
    return os.path.realpath(os.path.join(DIRNAME, "environment", *filepath))

WALL_TEXTURE_PATH = resource_path("textures", "walls")

class PybulletEnvironment:
    """This class deals with everything pybullet or environment (obstacles) related"""

    WHISKER_LENGTH = 1
    ROBOT_Z_POS = 0.02 # see p3dx model

    def __init__(self,
        env_model: str = "Savinov_val3",
        dt: float = 1e-2,
        visualize=False,
        realtime=False,
        build_data_set=False,
        start: Optional[types.Vector2D] = None,
        orientation: types.Angle = np.pi/2,
        frame_limit=5000,
        contains_robot=True,
        wall_kwargs={},
    ):
        """ Create environment.

        arguments:
        env_model   -- layout of obstacles in the environment 
                    (choices: "plane", "Savinov_val2", "Savinov_val3", "Savinov_test7")
        dt          -- timestep for simulation
        mode        -- choose goal vector calculation (choices: "analytical", "keyboard", "pod", "linear_lookahead")
        visualize   -- opens JAVA application to see agent in environment
        buildDataSet-- camera images are only taken when this is true
        start       -- the agent's [x,y] starting position (default [0,0])
        orientation -- the agent's starting orientation (default np.pi/2 (faces North))
        """

        try:
            p.disconnect()
        except:
            pass

        self.visualize = visualize  # to open JAVA application

        if os.getenv('BULLET_SHMEM_SERVER', False):
            clientId = p.connect(p.SHARED_MEMORY)
        elif self.visualize:
            clientId = p.connect(p.GUI)
        else:
            clientId = p.connect(p.DIRECT)
        # if we're connected to a different client, we need to pass that clientId as an argument to basically all pybullet functions
        assert clientId == 0, f"{clientId=}"

        if realtime:
            assert self.visualize, "realtime without visualization does not make sense"
            p.setRealTimeSimulation(1)

        self.env_model = env_model
        self.arena_size = 15

        base_position = [0, 0.05]  # [0, 0.05] ensures that it actually starts at origin

        self.planeID = None
        self.mazeID: list[int] = []

        # environment choices
        if env_model == "Savinov_val3":
            base_position = [-2, 0.05]
            p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[-2, -0.35, 5.0])
            # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[-0.55, -0.35, 5.0])
        elif env_model == "Savinov_val2":
            base_position = [0, 3.05]
            p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[0.55, -0.35, 5.0])
        elif env_model == "Savinov_test7":
            base_position = [-1, 0.05]
            p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[-1.55, -0.35, 5.0])
        elif env_model == "plane":
            p.resetDebugVisualizerCamera(cameraDistance=4.5, cameraYaw=0, cameraPitch=-70,
                                         cameraTargetPosition=[0, 0, 0])
            urdfRootPath = pybullet_data.getDataPath()
            self.planeID = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"))
        elif "obstacle" in env_model:
            plane = resource_path(self.env_model, "plane.urdf")
            self.planeID = p.loadURDF(plane)
        elif env_model.startswith("linear_sunburst"):
            base_position = [5.5, 0.55]
            if env_model == "linear_sunburst":
                doors_option = "plane"
            else:
                _, doors_option = env_model.split('.') # "plane" for default, "plane_doors", "plane_doors_individual"
            self.planeID = p.loadURDF(resource_path("linear_sunburst", doors_option + ".urdf"))
        else:
            raise ValueError("No matching env_model found.")

        base_position = start if start is not None else base_position

        if "Savinov" in env_model:
            # load the plane and maze with desired textures
            # self.mazeID = self.__load_obj(resource_path(self.env_model, "mesh.obj"), wall_texture)
            self.planeID = self.__load_obj(resource_path(self.env_model, "plane100.obj"), resource_path("textures", "green_floor.png"))
            all_wall_kwargs = { 'textures': resource_path("textures", "walls", "yellow_wall.png") }
            all_wall_kwargs.update(wall_kwargs)
            self.mazeID = self.__load_walls(resource_path(self.env_model), **all_wall_kwargs)

        p.setGravity(0, 0, -9.81)

        self.dt = dt
        p.setTimeStep(self.dt)

        # load agent
        if contains_robot:
            self.robot = Robot(self, base_position, orientation, frame_limit=frame_limit, build_data_set=build_data_set)
            # check if agent touches maze -> invalid start position
            if not env_model == "plane" and not "obstacle" in env_model and self.detect_maze_agent_contact():
                raise ValueError("Invalid start position. Agent and maze overlap.")
        else:
            self.robot = None

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, traceback):
        if self.robot is not None:
            self.robot.delete()
        if self.planeID is not None:
            p.removeBody(self.planeID)
        for wallID in self.mazeID:
            p.removeBody(wallID)
        p.removeAllUserDebugItems()
        p.disconnect()

    def switch_visualization(self, visualize) -> Self:
        if visualize == self.visualize:
            return self
        if self.robot is not None:
            pos = self.robot.position_and_angle
            robot_kwargs = { 'contains_robot': True, 'start': pos[0], 'orientation': pos[1], 'build_data_set': self.robot.buildDataSet }
        else:
            robot_kwargs = { 'contains_robot': False }
        self.__exit__(None, None, None)
        new_env = PybulletEnvironment(
            env_model=self.env_model, dt=self.dt, visualize=visualize,
            realtime=False, **robot_kwargs
        )
        if self.robot is not None:
            new_robot = new_env.robot
            self.robot.env = new_env
            self.robot.ID = new_robot.ID
        return new_env

    @property
    def dimensions(self):
        return environment_dimensions(self.env_model)

    def __load_walls(self, model_folder, textures : str | Callable[[int], str] | list[str], nb_batches = None) -> list[int]:
        MAXIMUM_BATCH_SIZE = 16 # bullet doesn't accept multibodies with >16 bodies

        wall_dir = os.path.join(model_folder, "walls")
        wall_files = os.listdir(wall_dir)
        wall_files = map(lambda filename: os.path.join(wall_dir, filename), wall_files)
        wall_files = filter(lambda filepath : os.path.isfile(filepath) and filepath[-4:] == '.obj', wall_files)
        wall_files = list(wall_files)

        if nb_batches is None:
            if type(textures) is list:
                nb_batches = len(textures)
                batches = np.array_split(wall_files, nb_batches)
                if len(batches[0]) > MAXIMUM_BATCH_SIZE:
                    batches : List[List[str]] = list(list(itertools.batched(big_batch, MAXIMUM_BATCH_SIZE)) for big_batch in batches)
                    textures = [ [ textures[i] ] * len(batches[i]) for i in range(len(batches)) ]
                    batches, textures = itertools.chain.from_iterable(batches), itertools.chain.from_iterable(textures)
            elif callable(textures): # assume the user wants to set each wall individually
                batches = [ [wall] for wall in wall_files ]
                textures = [ textures(i) for i in range(len(wall_files)) ]
            elif type(textures) is str:
                batches = list(itertools.batched(wall_files, n=MAXIMUM_BATCH_SIZE))
                textures = [ textures for _ in batches ]

        loaded_textures = {}
        def load_texture(texture_name : str):
            if texture_name not in loaded_textures:
                try:
                    loaded_textures[texture_name] = p.loadTexture(os.path.join(WALL_TEXTURE_PATH, texture_name))
                except Exception:
                    print("Couldn't load", texture_name, f"({os.path.join(WALL_TEXTURE_PATH, texture_name)})")
                    raise
            return loaded_textures[texture_name]

        textures = map(load_texture, textures)

        walls=[]
        for batch, textureId in zip(batches, textures):
            batch = list(batch) # tuples cause a segfault in PyBullet
            visualShapeId = p.createVisualShapeArray(
                shapeTypes=[p.GEOM_MESH for _ in batch],
                fileNames=batch,
            )
            collisionShapeId = p.createCollisionShapeArray(
                shapeTypes=[p.GEOM_MESH for _ in batch],
                fileNames=batch,
            )
            multiBodyId = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collisionShapeId,
                baseVisualShapeIndex=visualShapeId,
                basePosition=[0, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
            p.changeVisualShape(multiBodyId, -1, textureUniqueId=textureId)
            walls.append(multiBodyId)

        return walls


    def __load_obj(self, objectFilename : str, texture : str) -> int:
        """load object files with specified texture into the environment"""

        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=objectFilename,
            rgbaColor=None,
            meshScale=[1, 1, 1])

        collisionShapeId = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=objectFilename,
            meshScale=[1, 1, 1])

        multiBodyId = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

        textureId = p.loadTexture(texture)
        p.changeVisualShape(multiBodyId, -1, textureUniqueId=textureId)
        return multiBodyId

    def camera(self, agent_pos_orn: Optional[types.PositionAndOrientation] = None) -> types.Image:
        """ simulates a camera mounted on the robot, creating images """

        distance = 100000
        img_w, img_h = 64, 64

        if agent_pos_orn is not None:
            agent_pos, agent_orn = agent_pos_orn
            agent_pos = (agent_pos[0], agent_pos[1], PybulletEnvironment.ROBOT_Z_POS)
            yaw = agent_orn
        else:
            agent_pos, agent_orn_quaternion = \
                p.getBasePositionAndOrientation(self.robot.ID)

            yaw = p.getEulerFromQuaternion(agent_orn_quaternion)[-1]

        xA, yA, zA = agent_pos
        zA = zA + 0.3  # make the camera a little higher than the robot

        # Put the camera in front of the robot to simulate eyes
        xA = xA + math.cos(yaw) * 0.2
        yA = yA + math.sin(yaw) * 0.2

        # compute focusing point of the camera
        xB = xA + math.cos(yaw) * distance
        yB = yA + math.sin(yaw) * distance
        zB = zA

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[xA, yA, zA],
            cameraTargetPosition=[xB, yB, zB],
            cameraUpVector=[0, 0, 1.0]
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=120, aspect=1.5, nearVal=0.02, farVal=3.5)

        _height, _width, rgb_img, _data1, _data2 = p.getCameraImage(img_w, img_h,
                               view_matrix,
                               projection_matrix, shadow=True,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)

        return rgb_img

    def step(self):
        p.stepSimulation()
        if self.visualize:
            time.sleep(self.dt / 5)

    def step_forever(self, slowdown = 1):
        while True:
            p.stepSimulation()
            if self.visualize:
                time.sleep(self.dt * slowdown / 5)

    def detect_maze_agent_contact(self):
        """ true, if the robot is in contact with the maze """
        return any(
            bool(p.getContactPoints(self.robot.ID, wallId))
            for wallId in self.mazeID
        )

    def end_simulation(self):
        p.disconnect()

    def add_debug_line(self, start, end, color, width=1, height=None):
        """ add line into visualization """
        if not DEBUG:
            return
        if len(start) == 2 or height:
            height = height or 1
            start = [*start, height]; end = [*end, height]
        if self.visualize:
            p.addUserDebugLine(start, end, color, width)

    def straight_lidar(
        self,
        agent_pos_orn : Optional[types.PositionAndOrientation] = None,
        ray_length = WHISKER_LENGTH,
        draw_debug_lines = False,
        radius: float = 0.25,
        num_ray_dir: float = 3,
    ) -> list[float]:
        if agent_pos_orn:
            rayFromPoint, euler_angle = agent_pos_orn
            if len(rayFromPoint) == 2:
                rayFromPoint = list(rayFromPoint) + [ PybulletEnvironment.ROBOT_Z_POS + 0.1 ] # TODO: make sure this is high enough not to hit the robot, if there is one
        else:
            rayFromPoint, euler_angle = self.robot.lidar_sensor_position
        rayFromPoint = np.array(rayFromPoint)

        rayFrom, rayTo = [], []
        for distance in np.linspace(start=-radius, stop=radius, num=num_ray_dir):
            rayFromI = rayFromPoint + distance * np.array([ -np.sin(euler_angle), np.cos(euler_angle), 0 ])
            rayToI = rayFromI + ray_length * np.array([ np.cos(euler_angle), np.sin(euler_angle), 0 ])
            rayFrom.append(rayFromI); rayTo.append(rayToI)

        rayHitColor = [1, 0, 0]
        rayMissColor = [1, 1, 1]

        ray_return = []
        results = p.rayTestBatch(rayFrom, rayTo, numThreads=0)  # get intersections with obstacles
        for start, end, hit in zip(rayFrom, rayTo, results):
            hit_object_uid = hit[0]

            if hit_object_uid < 0:
                if draw_debug_lines:
                    self.add_debug_line(start, end, (0, 0, 0))
                ray_return.append(-1)
            else:
                if self.robot:
                    assert hit_object_uid != self.robot.ID
                hitPosition = hit[3]
                if draw_debug_lines:
                    self.add_debug_line(start, end, rayHitColor)
                distance = math.sqrt((hitPosition[0] - start[0]) ** 2 + (hitPosition[1] - start[1]) ** 2)
                assert 0 <= distance and distance <= 1
                ray_return.append(distance)
        return ray_return

    def lidar(
        self,
        agent_pos_orn: Optional[types.PositionAndOrientation] = None,
        ray_length = WHISKER_LENGTH,
        draw_debug_lines = False,
        **angle_args
    ) -> Tuple[LidarReading, List[Vector2D]]:
        """
        returns the egocentric distance to obstacles in num_ray_dir directions

        returns: (distances, angles, hitpoints)
        """

        if self.visualize and draw_debug_lines:
            p.removeAllUserDebugItems()  # removes raylines

        ray_return = []
        rayFrom = []
        rayTo = []
        rayHitColor = [1, 0, 0]
        rayMissColor = [1, 1, 1]

        if agent_pos_orn:
            rayFromPoint, euler_angle = agent_pos_orn
            if len(rayFromPoint) == 2:
                ROBOT_HEIGHT = 0.25 # TODO: precise, non-guessed value
                MARGIN = 0.1
                rayFromPoint = list(rayFromPoint) + [ PybulletEnvironment.ROBOT_Z_POS + ROBOT_HEIGHT + MARGIN ]
        else:
            rayFromPoint, euler_angle = self.robot.lidar_sensor_position

        ray_angles = list(LidarReading.angle_range(start_angle=euler_angle, **angle_args))

        for angle in ray_angles:
            rayTo.append([
                rayFromPoint[0] + ray_length * math.cos(angle),
                rayFromPoint[1] + ray_length * math.sin(angle),
                rayFromPoint[2]
            ])
            rayFrom.append(rayFromPoint)

        results = p.rayTestBatch(rayFrom, rayTo, numThreads=0)  # get intersections with obstacles
        for start, end, hit in zip(rayFrom, rayTo, results):
            hit_object_uid = hit[0]

            if hit_object_uid < 0:
                if draw_debug_lines:
                    self.add_debug_line(start, end, (0, 0, 0))
                ray_return.append(-1)
            else:
                if self.robot:
                    assert hit_object_uid != self.robot.ID
                hitPosition = hit[3]
                if draw_debug_lines:
                    self.add_debug_line(start, end, rayHitColor)
                distance = math.sqrt((hitPosition[0] - start[0]) ** 2 + (hitPosition[1] - start[1]) ** 2)
                assert 0 <= distance and distance <= ray_length
                ray_return.append(distance)

        return LidarReading(ray_return, ray_angles), [it[3] for it in results]


class Robot:
    def __init__(
        self,
        env: PybulletEnvironment,
        base_position : types.Vector2D,
        base_orientation : types.Angle = np.pi/2,
        max_speed=5.5,
        frame_limit=5000,
        build_data_set : bool = True,
        compass: Optional['Compass'] = None,
    ):
        """
        arguments:
            max_speed - determines speed at which agent travels: max_speed = 5.5 -> actual speed of ~0.5 m/s
            build_data_set - when true create camera images
        """
        base_position = [ base_position[0], base_position[1], PybulletEnvironment.ROBOT_Z_POS ]
        base_orientation = p.getQuaternionFromEuler([0, 0, base_orientation])
        filename = os.path.join(DIRNAME,"p3dx","urdf","pioneer3dx.urdf")
        self.ID = p.loadURDF(filename, basePosition=base_position, baseOrientation=base_orientation)
        self.env = env
        self.buildDataSet = build_data_set

        self.max_speed = max_speed

        self.buffer = 0  # buffer for checking if agent got stuck, discards timesteps spent turning towards the goal
        self.data_collector = DatasetCollector(frame_limit=frame_limit, collectImages=build_data_set)

        goal_vector: Vector2D = compass.calculate_goal_vector() if compass else np.array([0., 0.])
        self.save_snapshot(goal_vector=goal_vector)  # save initial configuration

        self.mapping = 1.5  # see local_navigation experiments

        self.navigation_hooks : List[Callable[[Vector2D], None]] = []

    def __enter__(self):
        assert self.env.robot is None
        self.env.robot = self
        return self

    def __exit__(self, ex_type, ex_value, traceback):
        self.env.robot = None
        self.delete()

    def navigation_step(self, goal_vector: Vector2D, allow_backwards=False):
        """ One navigation step for the agent. Moves towards goal_vector.
        """

        if self.env.visualize:
            p.removeAllUserDebugItems()

        gains = self.compute_gains(goal_vector, allow_backwards=allow_backwards)
        assert not np.any(np.isnan(gains))

        self._step(gains, goal_vector)
        for hook in self.navigation_hooks:
            hook(self.xy_speed)

    @property
    def position_and_angle(self) -> Tuple[types.Vector2D, types.Angle]:
        # Possible improvement: check if retrieving last value from data_collector is faster
        position, angle = p.getBasePositionAndOrientation(self.ID)
        assert position[2] > -10 #== PybulletEnvironment.ROBOT_Z_POS
        position = np.array([position[0], position[1]])
        angle = p.getEulerFromQuaternion(angle)[2]
        return position, angle

    @property
    def position(self) -> types.Vector2D:
        return self.position_and_angle[0]

    @property
    def xy_speed(self) -> types.Vector2D:
        linear_v, _ = p.getBaseVelocity(self.ID)
        return linear_v[0], linear_v[1]

    def heading_vector(self) -> types.Vector2D:
        _, current_angle = self.position_and_angle
        return [np.cos(current_angle), np.sin(current_angle)]

    @property
    def lidar_sensor_position(self) -> types.PositionAndOrientation:
        linkWorldPosition, linkWorldOrientation, *_ignored_data = p.getLinkState(self.ID, 0) # position of chassis compared to base
        linkWorldPosition = list(linkWorldPosition)
        linkWorldPosition[2] += PybulletEnvironment.ROBOT_Z_POS # ROBOT_Z_POS is position of base
        linkWorldPosition[2] += 0.1 # safety margin; TODO this could be removed once we put the correct 3D orientation
        # TODO maybe the orientation needs to be turned?
        angle = p.getEulerFromQuaternion(linkWorldOrientation)[2]
        return tuple(linkWorldPosition), angle

    def save_snapshot(self, goal_vector : Optional[types.Vector2D]):
        position, angle = self.position_and_angle
        linear_v = self.xy_speed
        self.data_collector.append(position, angle, linear_v, goal_vector)
        if self.buildDataSet:
            self.data_collector.images.append(self.env.camera((position, angle)))

    def compute_gains(self, goal_vector, allow_backwards=False) -> Tuple[float, float]:
        """ computes the motor gains resulting from (inhibited) goal vector
        Arguments:
        :param allow_backwards: Whether to allow backwards movement.
                                If set to False, if the goal vector points behind the robot,
                                the robot will start a U-Turn.
        """
        current_heading = self.heading_vector()
        diff_angle = compute_angle(current_heading, goal_vector)
        assert not np.isnan(diff_angle), (current_heading, goal_vector)

        gain = min(np.linalg.norm(goal_vector) * 5, 1)
        assert gain is not None

        # If close to the goal do not move
        if gain < 0.5:
            gain = 0

        backwards = False
        if abs(diff_angle) > math.radians(90) and allow_backwards:
            backwards = True
            diff_angle = -diff_angle + np.pi if diff_angle > 0 else -diff_angle - np.pi

        # threshold for turning: turning too sharply is not biologically accurate
        if abs(diff_angle) > math.radians(30):
            diff_angle = math.copysign(math.radians(30), diff_angle)
            v_right = self.max_speed * diff_angle * 2 * gain / np.pi
            v_left = - v_right
        else:
            # For biologically inspired movement: only adjust course slightly
            # TODO Johanna: Future Work: This restricts robot movement too severely
            v_left = self.max_speed * (1 - diff_angle / np.pi * 2) * gain
            v_right = self.max_speed * (1 + diff_angle / np.pi * 2) * gain

        if backwards:
            [v_left, v_right] = [-v_left, -v_right]
        assert not np.isnan(v_left) and not np.isnan(v_right)
        return (v_left, v_right)

    def keyboard_simulation(self):
        """ Control the agent with your keyboard. SPACE ends the simulation."""
        turn, forward = 0, 0
        while True:
            """ simulates a timestep with keyboard controlled movement """
            keys = p.getKeyboardEvents()
            for k, v in keys.items():

                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    turn = -0.5
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED):
                    turn = 0
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    turn = 0.5
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED):
                    turn = 0

                if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    forward = 1
                if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED):
                    forward = 0
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    forward = -1
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED):
                    forward = 0
                if k == p.B3G_SPACE and (v & p.KEY_WAS_TRIGGERED):
                    self.calculate_obstacle_vector()
                if k == p.B3G_BACKSPACE and (v & p.KEY_WAS_TRIGGERED):
                    raise KeyboardInterrupt()

            v_left = (forward - turn) * self.max_speed
            v_right = (forward + turn) * self.max_speed
            gains = [v_left, v_right]
            self._step(gains, None)
            self.env.camera()

    def _step(self, gains: tuple[float, float], current_goal_vector: Optional[Vector2D]):
        # change speed
        p.setJointMotorControlArray(bodyUniqueId=self.ID,
                            jointIndices=[4, 6],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocities=gains,
                            forces=[10, 10])
        self.env.step()
        self.save_snapshot(current_goal_vector)

    def calculate_simple_normal_vector(self, lidar_data: Optional[tuple[LidarReading, list[Vector2D]]] = None) -> Vector2D:
        if lidar_data is None:
            lidar_data = self.env.lidar(tactile_cone=120, num_ray_dir=21, blind_spot_cone=0, agent_pos_orn=self.lidar_sensor_position)
        lidar, hit_points = lidar_data
        hit_points = np.array(hit_points)
        hit_points = hit_points[np.array(lidar.distances) != -1]

        normal_vector : Vector2D = np.array([0.0, 0.0], dtype=float)
        if len(hit_points) == 0:
            return normal_vector

        height = hit_points[0, 2] # they should all have the same height anyway
        hit_points = hit_points[:, :2]
        hit_points_egocentric = hit_points - self.position

        for vector in hit_points_egocentric:
            distance = np.linalg.norm(vector)
            normal_vector += vector * (-1) / distance**2

        self.env.add_debug_line([*self.position, height], [*(self.position + normal_vector), height], (0, 0, 1))
        return normal_vector

    def calculate_obstacle_vector(self, lidar_data : Optional[Tuple[LidarReading, List[Vector2D]]] = None) -> Tuple[Vector2D, Vector2D]:
        """
        Calculates the obstacle_vector from the ray distances.

        Returns:
            p: the first point on the obstacle line
            v: the direction vector of the obstacle
        """
        if lidar_data is None:
            lidar_data = self.env.lidar(tactile_cone=120, num_ray_dir=21, blind_spot_cone=0, agent_pos_orn=self.lidar_sensor_position, ray_length=1)
        lidar, hit_points = lidar_data
        start_index, end_index = closest_subsegment(lidar.distances)
        hit_points = hit_points[start_index:end_index+1]

        if end_index < 0:
            return np.array([0, 0]), np.array([0.0, 0.0])

        if end_index - start_index + 1 < 5:
            middle_index = (end_index + start_index) // 2
            angle = lidar.angles[middle_index]
            direction_vector = np.array([-np.sin(angle), np.cos(angle)])
        else:
            try:
                # TODO Pierre: isn't that overkill compared to e.g. just taking the slope of two points?
                # Calculate the slope (m) of the line using linear regression
                # Step 2: Calculate a straight line using linear regression that fits the best to these points
                hit_points = np.array(hit_points)
                x_values = hit_points[:, 0]
                y_values = hit_points[:, 1]

                # For cases where x_values are constant (obstacle parallel to y-axis),
                # we can directly calculate the slope and intercept of the line.
                if np.all(abs(x_values - x_values[0]) < 0.001):
                    direction_vector = np.array([0.0, 1.0])
                else:
                    # Calculate the slope and intercept using the Least Squares Regression.
                    A = np.vstack([x_values, np.ones(len(x_values))]).T
                    slope, intercept = np.linalg.lstsq(A, y_values, rcond=None)[0]
                    direction_vector = np.array([1.0, slope])
                    direction_vector /= np.linalg.norm(direction_vector)  # Normalize the direction vector
            except (IndexError, ValueError, np.linalg.LinAlgError):
                return np.array([0.0, 0.0]), np.array([0.0, 0.0])

        if lidar[end_index] > 0:
            self_point = p.getLinkState(self.ID, 0)[0]
            start_point = self_point + np.array(
                [np.cos(lidar.angles[end_index]), np.sin(lidar.angles[end_index]), self_point[-1]]) * lidar[end_index]
            end_point = start_point - np.array([direction_vector[0], direction_vector[1], 0])
            self.env.add_debug_line(start_point, end_point, (0, 0, 0))

        assert min(lidar[start_index:end_index + 1]) != 0, lidar[start_index:end_index + 1]
        direction_vector = direction_vector * 1.5 / min(lidar[start_index:end_index + 1])
        return hit_points[0], direction_vector

    def turn_to_goal(self, goal_vector: Vector2D, tolerance: types.Angle = 0.05):
        """ Agent turns to face in goal vector direction """

        i = 0
        MAX_ITERATIONS = 1000
        diff_angle = None
        while i < MAX_ITERATIONS and (diff_angle is None or abs(diff_angle) > tolerance):
            i += 1
            normed_goal_vector = np.array(goal_vector) / np.linalg.norm(np.array(goal_vector))

            current_heading = self.heading_vector()
            diff_angle = compute_angle(current_heading, normed_goal_vector) / np.pi
            #print("Turning in place, diff_angle is", diff_angle, end="...\r")

            gain = min(np.linalg.norm(normed_goal_vector) * 5, 1)

            # If close to the goal do not move
            if gain < 0.5:
                break

            # If large difference in heading, do an actual turn
            if abs(diff_angle) > tolerance and gain > 0:
                max_speed = self.max_speed / 2
                direction = np.sign(diff_angle)
                if direction > 0:
                    v_left = max_speed * gain * -1
                else:
                    v_left = max_speed * gain
            else:
                v_left = 0

            gains = (v_left, -v_left)

            self._step(gains, goal_vector)

        # turning in place does not mean the agent is stuck
        self.buffer = len(self.data_collector.xy_coordinates)

    def delete(self):
        p.removeBody(self.ID)

class DatasetCollector:
    DataPoint = Tuple[types.Vector2D, types.Angle, types.Vector2D, float, Optional[types.Vector2D]]
    class Index: # Namespace
        POSITION = 0
        ANGLE = 1
        SPEED = 2
        SPEED_NORM = 3
        GOAL = 4

    def __init__(self, frame_limit = 5000, collectImages = True):
        self.frame_limit = frame_limit

        #self.xy_coordinates : List[Vector2D] = []  # keeps track of agent's coordinates at each time step
        #self.orientation_angle = []  # keeps track of agent's orientation at each time step
        #self.xy_speeds = []  # keeps track of agent's speed (vector) at each time step
        #self.speeds = []  # keeps track of agent's speed (value) at each time step
        #self.goal_vector_array = []  # keeps track of agent's goal vector at each time step
        self.data : List[DatasetCollector.DataPoint] = []
        self.images : List[types.Image] = [] # if buildDataSet: collect images

        if collectImages:
            self.images : list[Image] = [] 


    def append(self, position : types.Vector2D, angle : types.Angle, speed : types.Vector2D, goal_vector : Optional[types.Vector2D]):
        self.data.append((
            np.array(position),
            angle,
            speed,
            np.linalg.norm(speed),
            goal_vector,
        ))
        self.data = self.data[-self.frame_limit:]

        # TODO: self.xy_coordinates = self.xy_coordinates[-self.frame_limit*100:] -> is that an error?

    @property
    def xy_coordinates(self):
        return [ datapoint[0] for datapoint in self.data ]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_observations(self, deltaT = 3) -> List[Any]: # TODO: how is this different from a types.Image?
        # observations with context length k=10 and delta T = 3
        assert len(self.images) != 0
        return self.images
        # observations = self.images[::3][-1:]
        # return [np.transpose(observation[2], (2, 0, 1)) for observation in observations]

all_possible_textures = [ file for file in sorted(os.listdir(WALL_TEXTURE_PATH)) if file[-4:] == '.jpg' ]

if __name__ == "__main__":
    """
    Test keyboard movement an plotting in different environments. 
    Press arrow keys to move, SPACE to visualize egocentric rays with obstacle detection and BACKSPACE to exit.

    Available environments:
    - plane
    - obstacle_map_0
    - obstacle_map_1
    - obstacle_map_2
    - obstacle_map_3
    - Savinov_test7
    - Savinov_val2
    - Savinov_val3
    """
    try:
        env_model = sys.argv[1]
    except IndexError:
        env_model = "Savinov_val3"

    random = Random(0)
    with PybulletEnvironment(env_model, visualize=True, realtime=True, start=(-0.5, 0),
        wall_kwargs={
            'textures': lambda i: random.choice(all_possible_textures)
        }
    ) as env:
        try:
            env.robot.keyboard_simulation()
        except KeyboardInterrupt:
            pass

        # plot the agent's trajectory in the environment
        plot.plotTrajectoryInEnvironment(env)
