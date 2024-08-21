""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""

import os
import sys

from system.controller.reachability_estimator.types import PlaceInfo

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from system.bio_model.cognitive_map import CognitiveMapInterface
from system.bio_model.place_cell_model import PlaceCell, PlaceCellNetwork

from system.controller.local_controller.decoder.linear_lookahead_no_rewards import *
from system.controller.local_controller.decoder.phase_offset_detector import PhaseOffsetDetectorNetwork
from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.controller.local_controller.compass import Compass, AnalyticalCompass
from system.controller.local_controller.local_controller import LocalController, RobotStuck
import system.controller.local_controller.local_controller as ctrl_rules
from system.types import Spikings, WaypointInfo, types, Vector2D
from system.utils import normalize

import system.plotting.plotResults as plot
import numpy as np
from typing import Callable, Iterable, Optional, Sequence, Any, Generic, TypeVar
from system.debug import PLOTTING, DEBUG

plotting = ('localctrl' in PLOTTING)  # if True: plot everything
debug = ('localctrl' in DEBUG)  # if True: print debug output


def print_debug(*params):
    """ output only when in debug mode """
    if debug:
        print(*params)


class GoalVectorCache(Compass):
    @staticmethod
    def parse(pc: 'PlaceCell'):
        return GcCompass.parse(pc)

    @property
    def arrival_threshold(self):
        return self.impl.arrival_threshold

    def reset_goal(self, new_goal: Spikings):
        self.impl.reset_goal(new_goal)
        self.goal_vector = None

    def __init__(self, impl: Compass[Spikings], update_fraction=0.5):
        # intentionally don't call super().__init__() because we don't really need it
        self.impl = impl
        self.update_fraction = update_fraction  # how often the goal vector is recalculated
        self.goal_vector = None  # egocentric goal vector after last update
        self.distance_to_goal_original = None  # distance to goal after last recalculation

    def calculate_goal_vector(self) -> Vector2D:
        if self.goal_vector is None:
            self.goal_vector = self.impl.calculate_goal_vector()
            self.distance_to_goal_original = np.linalg.norm(self.goal_vector)

        return self.goal_vector

    def reset_position(self, new_position: Spikings):
        return self.impl.reset_position(new_position)

    def update_position(self, robot: Robot):
        distance_to_goal = np.linalg.norm(self.goal_vector)  # current length of goal vector

        if (
                self.distance_to_goal_original > 0.3 and
                distance_to_goal / self.distance_to_goal_original < self.update_fraction
        ):
            # Vector-based navigation and agent has traversed a large portion of the goal vector
            # Discarding current (now inaccurate) goal vector so it gets recomputed next time calculate_goal_vector() is called
            self.goal_vector = None

            # adding a turn here could make the local controller more robust
            # robot.turn_to_goal()
        else:
            self.goal_vector = self.goal_vector - np.array(robot.xy_speed) * robot.env.dt
        self.impl.update_position(robot)

class GcCompass(Compass[Spikings]):
    """ Uses decoded grid cell spikings as a goal vector. """

    @staticmethod
    def parse(pc: 'PlaceInfo'):
        return pc.spikings

    def __init__(self, gc_network: GridCellNetwork):
        self.gc_network = gc_network

    def reset_position(self, new_position: Spikings):
        self.gc_network.set_as_current_state(new_position)

    def reset_goal(self, new_goal: Spikings):
        self.gc_network.set_as_target_state(new_goal)

    def update_position(self, robot: 'Robot'):
        pass # TODO Pierre: ensure that the GCNetwork is updated separately

    @staticmethod
    def factory(mode, gc_network: GridCellNetwork, *args,
        pod_network: Optional[PhaseOffsetDetectorNetwork] = None,
        arena_size: Optional[float] = None,
        **kwargs
    ):
        if mode == "pod":
            return GoalVectorCache(PodGcCompass(pod_network, gc_network, *args, **kwargs))
        if mode == "linear_lookahead":
            return GoalVectorCache(LinearLookaheadGcCompass(arena_size, gc_network, *args, **kwargs))
        elif mode == "combo":
            return ComboGcCompass(gc_network, pod_network, *args, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected one of: analytical, pod, linear_lookahead, combo")


class PodGcCompass(GcCompass):
    arrival_threshold = 0.5

    def __init__(self, pod_network: Optional[PhaseOffsetDetectorNetwork] = None, *args, **kwargs):
        if pod_network is None:  # TODO (Pierre): this is ugly
            pod_network = PhaseOffsetDetectorNetwork(16, 9, 40)
        super().__init__(*args, **kwargs)
        self.pod_network = pod_network

    def calculate_goal_vector(self):
        """For Vector-based navigation, computes goal vector with one grid cell decoder"""
        return self.pod_network.compute_goal_vector(self.gc_network.gc_modules)    


class LinearLookaheadGcCompass(GcCompass):
    arrival_threshold = 0.02
    def __init__(self, arena_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arena_size = arena_size

    def calculate_goal_vector(self):
        goal_vector, goal_info = perform_look_ahead_2xnr(self.gc_network, self.arena_size)
        return goal_vector


class ComboGcCompass(GcCompass):
    def __init__(self, gc_network: GridCellNetwork, pod_network: Optional[PhaseOffsetDetectorNetwork] = None, *args, **kwargs):
        super().__init__(gc_network, *args, **kwargs)
        # self.gc_network = gc_network # already done by the super().__init__
        compass = PodGcCompass(pod_network, gc_network, *args, **kwargs)
        self.pod_network = compass.pod_network
        compass = GoalVectorCache(compass)
        self.impl = compass

    def reset_goal(self, new_goal: Spikings):
        super().reset_goal(new_goal)
        if type(self.impl) == LinearLookaheadGcCompass:
            self.impl = PodGcCompass(pod_network=self.pod_network, gc_network=self.gc_network, goal_pos=new_goal)
            self.impl = GoalVectorCache(self.impl)
        else:
            self.impl.reset_goal(new_goal)

    def calculate_goal_vector(self, *args, **kwargs):
        return self.impl.calculate_goal_vector(*args, **kwargs)

    @property
    def arrival_threshold(self):
        return LinearLookaheadGcCompass.arrival_threshold

    def update_position(self, robot: Robot):
        self.impl.update_position(robot)
        goal_reached = self.impl.reached_goal()
        if goal_reached and type(self.impl) == PodGcCompass:
            # switch from pod to linear lookahead
            self.impl = LinearLookaheadGcCompass(arena_size=robot.env.arena_size, gc_network=self.gc_network, goal_pos=self.impl.goal_pos)
            self.impl = GoalVectorCache(self.impl)
            # robot.turn_to_goal(goal_vector)


def create_gc_spiking(start: Vector2D, goal: Vector2D, gc_network_at_start: Optional[GridCellNetwork]=None, plotting=plotting) -> types.Spikings:
    """ 
    Agent navigates from start to goal accross a plane without any obstacles, using the analyticallly 
    calculated goal vector to genereate the grid cell spikings necessary for the decoders. During actual
    navigation this would have happened in the exploration phase.
    """

    # Grid-Cell Initialization
    dt = 1e-2
    if gc_network_at_start is None:
        gc_network = GridCellNetwork(from_data=True, dt=dt)
    else:
        assert gc_network_at_start.dt == dt
        gc_network_at_start.reset_s_virtual()
        gc_network = gc_network_at_start

    compass = AnalyticalCompass(start_pos=start, goal_pos=goal)
    robot_position = np.array(start, dtype=float)
    history: list[Vector2D] = [robot_position]

    if compass.reached_goal():
        assert False, "Positions are too close to each other!"

    i = 0
    while not compass.reached_goal():
        i += 1
        if i == 5000:
            raise AssertionError("Agent should not get caught in a loop in an empty plane.")

        goal_vector = compass.calculate_goal_vector()
        if np.linalg.norm(goal_vector) == 0:
            break
        ROBOT_MAX_SPEED = 0.5 # This is empirically the maximum speed of the robot, but I don't know why
        # (pybullet_environment.py says 5.5, but this value probably gets multiplied with something somewhere)
        xy_speed = ROBOT_MAX_SPEED * normalize(goal_vector)

        robot_position += xy_speed * dt
        history.append(np.array(robot_position))
        compass.current_pos = robot_position
        gc_network.track_movement(xy_speed, virtual=True)

    if plotting: plot.plotTrajectoryInEnvironment(env_model="plane", xy_coordinates=history)
    return gc_network.consolidate_gc_spiking(virtual=True)

from deprecation import deprecated

@deprecated()
def setup_gc_network(dt) -> GridCellNetwork:
    """ Initialize the grid cell newtork """
    # Grid-Cell Initialization

    # note that if gc modules are created from data n and M are overwritten
    gc_network = GridCellNetwork(dt, from_data=True)

    return gc_network


def vector_navigation(
    env: PybulletEnvironment, compass: Compass, gc_network: Optional[GridCellNetwork] = None, controller: Optional[LocalController] = None, target_gc_spiking=None,
    step_limit=float('inf'), plot_it=plotting,
    collect_data_freq=False, collect_data_reachable=False, collect_nr_steps=False, exploration_phase=False,
    pc_network: Optional[PlaceCellNetwork] = None, cognitive_map: Optional[CognitiveMapInterface] = None,
    goal_pos: Optional[Vector2D] = None,
    add_nodes = True,
) -> tuple[bool, list[WaypointInfo]|tuple|int|PlaceCell|None]:
    """
    Agent navigates towards goal.

    arguments:
    env                    --  running PybulletEnvironment
    compass                --  A Compass pointing to the goal
    controller             --  An algorithm for local control and obstacle avoidance
    gc_network             --  grid cell network used for navigation (pod, linear_lookahead, combo)
                               or grid cell spiking generation (analytical)
    gc_spiking             --  grid cell spikings at the goal (pod, linear_lookahead, combo)
    step_limit             --  navigation stops after step_limit amount of steps (default infinity)
    plot_it                --  if true: plot the navigation (default false)
    collect_data_freq      -- return necessary data for trajectory generation
    collect_data_reachable -- return necessary data for reachability dataset generation
    exploration_phase      -- track movement for cognitive map and place cell model (this is a misnomer and also used in the navigation phase)
    pc_network             -- place cell network
    cognitive_map          -- cognitive map object
    goal_pos               --  The true location of the goal - for plotting & reporting

    Returns: (depending on the arguments):
    goal_reached : bool, data : List[WaypointInfo] if collect_data_freq
    goal_reached : bool, ???  if collect_data_reachable
    goal_reached : bool, nr_steps : int if collect_nr_steps
    goal_reached : bool, last_pc : PlaceCell|None if not add_nodes
    goal_reached : bool, last_pc : PlaceCell else
    """

    data: list[WaypointInfo] = []
    robot = env.robot
    assert robot is not None

    if gc_network is not None:
        def on_nav_step(robot: Robot):
            gc_network.track_movement(robot.xy_speed)
        robot.navigation_hooks.append(on_nav_step)

    if controller is None:
        controller = LocalController.default()

    # TODO Pierre: do this before the call
    if gc_network and (target_gc_spiking is not None):
        print('Warning: deprecated: please set the target_state beforehand')
        gc_network.set_as_target_state(target_gc_spiking)

    n = 0  # time steps
    goal_vector = compass.calculate_goal_vector()
    try:
        controller.reset_goal(goal_vector, robot)
    except RobotStuck:
        # a possible reason is e.g. TurnToGoal failing -> we can safely ignore that
        pass

    if collect_data_reachable:
        sample_after_turn = (robot.data_collector[-1][0], robot.data_collector[-1][1])
        first_goal_vector = goal_vector

    goal_reached = False
    end_state = ""  # for plotting
    last_pc = None
    last_observation = None
    while n < step_limit and not goal_reached:
        try:
            goal_vector = compass.calculate_goal_vector()
            if np.linalg.norm(goal_vector) == 0:
                goal_reached = True
            else:
                controller.step(goal_vector, robot)
                compass.update_position(robot)
                goal_reached = compass.reached_goal()
        except RobotStuck:
            break

        # TODO: feature suggestion: attach "nav step hooks" to the robot
        # so that we could run gc_network.track_movement and e.g. buildDataSet automatically

        if pc_network is not None and cognitive_map is not None:
            observations = robot.data_collector.get_observations()
            assert len(observations) != 0
            last_observation = observations[-1]
            firing_values, created_new_pc = pc_network.track_movement(
                PlaceInfo(
                    *robot.position_and_angle,
                    spikings=gc_network.consolidate_gc_spiking(),
                    img=last_observation, lidar=robot.env.lidar()[0],
                ),
                creation_allowed=exploration_phase,
            )

            lidar = env.lidar()
            assert lidar is not None
            mapped_pc = cognitive_map.track_vector_movement(
                firing_values, created_new_pc, pc_network.place_cells[-1], lidar=lidar,
                exploration_phase=exploration_phase, pc_network=pc_network)
            if mapped_pc is not None:
                last_pc = mapped_pc

        if collect_data_freq and n % collect_data_freq == 0:
            # collect grid cell spikings for trajectory generation
            spiking = gc_network.consolidate_gc_spiking().flatten()
            data.append((*robot.position_and_angle, spiking))

        n += 1

    if goal_reached:
        end_state = f"Agent reached the goal. Perceived distance: {np.linalg.norm(goal_vector)}."
        if goal_pos is not None:
            end_state += f"Actual distance: {np.linalg.norm(np.array(goal_pos) - robot.position)}."
    else:
        end_state = "Agent got stuck"

    if plot_it:
        plot.plotTrajectoryInEnvironment(env, title=end_state, end=goal_pos)

    if collect_data_freq:
        return goal_reached, data
    if collect_data_reachable:
        return goal_reached, [sample_after_turn, first_goal_vector]
    if collect_nr_steps:
        return goal_reached, n

    if gc_network is not None:
        robot.navigation_hooks.remove(on_nav_step)

    if not last_pc and add_nodes and pc_network:
        if last_observation is None:
            last_observation = robot.env.camera()
        position, angle = robot.position_and_angle
        pc_network.create_new_pc(PlaceInfo(spikings=gc_network.consolidate_gc_spiking(), img=last_observation, pos=position, angle=angle, lidar=robot.env.lidar()[0]))
        last_pc = pc_network.place_cells[-1]
    return goal_reached, last_pc


T = TypeVar('T')

from bisect import bisect_left

class ChainSequence(Generic[T]):
    def __init__(self, *sequences: list[Sequence[T]]):
        self.sequences = sequences
        self.cumsum = np.cumsum([len(seq) for seq in self.sequences])
    def __len__(self):
        return self.cumsum[-1]
    def __getitem__(self, idx) -> T:
        seq_idx = bisect_left(self.cumsum, idx+1)
        seq_offset = idx - self.cumsum[seq_idx]
        return self.sequences[seq_idx][seq_offset]

import random

class RandomTaker(Generic[T]):
    def __init__(self, dataset: Sequence[T], seed: Optional[int] = None):
        self.dataset = dataset
        self.rng = random.Random()
        self.rng.seed(seed)
    def __iter__(self): return self
    def __next__(self):
        return self.rng.choice(self.dataset)

from tqdm import tqdm

def randomPointsTest(
    controller: LocalController,
    Compass: Callable[[Vector2D, Vector2D], Compass],
    points: Iterable[tuple[Vector2D, Vector2D]],
    visualize=False, plot_it=False
):
    successes = 0
    points = iter(points)
    with PybulletEnvironment(env_model="Savinov_val3", visualize=visualize, contains_robot=False) as env:
        for i in (bar := tqdm(range(100))):
            start, goal = next(points)
            compass = Compass(start, goal)
            with Robot(env, base_position=start) as robot:
                goal_reached, _ = vector_navigation(env, compass, gc_network=None, plot_it=plot_it, controller=controller, step_limit=1000, goal_pos=goal)
                successes += goal_reached
            bar.set_description(f"{successes} ({successes/(i+1):.1%}) success")


if __name__ == "__main__":
    """ Test the local controller's ability of vector navigation with obstacle avoidance. """

    from argparse import ArgumentParser
    main_parser = ArgumentParser()
    experiments = main_parser.add_subparsers(dest='experiment', required=True)

    none_parser = experiments.add_parser('none')
    """
    Available decoders:
    - pod: phase-offset decoder
    - linear_lookahead: linear lookahead decoder
    - analytical: precise calculation of goal vector with information that is not biologically available to the agent
    - combo: uses pod until < 0.5 m from the goal, then switches to linear_lookahead for higher precision 
    The navigation uses the faster pod decoder until the agent thinks it has reached its goal, 
    then switches to slower linear lookahead for increased accuracy.

    Change the start and goal position and environment model, as needed.
    """
    none_parser.add_argument('decoder', choices=['pod', 'linear_lookahead', 'analytical', 'combo'])
    parse_vector = lambda st: [float(i) for i in st.split(',')]
    none_parser.add_argument('start', type=parse_vector, nargs='?', default=[-6, -0.5], help='Start position in the form x,y')
    none_parser.add_argument('goal', type=parse_vector, nargs='?', default=[-8, -0.5], help='Goal position in the form x,y')

    vector_nav_parser = experiments.add_parser('vector_navigation')
    vector_nav_parser.add_argument('decoder', choices=['pod', 'linear_lookahead', 'combo'], help='decoder')
    vector_nav_parser.add_argument('simplicity', choices=['simple', 'generate_gc'])
    vector_nav_parser.add_argument('-d', '--goal-distance', help='distance to the goal', type=int, default=15)
    """
    Test the local controller with different decoders

    Ctrl-F to see where to adjust the following parameters
    1) CHOOSE THE DECODER YOU WANT TO TEST
    2) CHOOSE THE DISTANCE TO THE GOAL
    3) CHOOSE WHETHER TO PERFORM
        3A) A SIMPLE RETURN TO START
        3B) GENERATING THE GOAL SPIKINGS, NAVIGATING TO THE GOAL, THEN RETURN TO START
    4) ADJUST THE NAME FOR SAVING YOUR RESULTS

    """

    obstacle_avoidance_parser = experiments.add_parser('obstacle_avoidance')
    obstacle_avoidance_parser.add_argument('--all', help='Execute all combinations of test parameters', action='store_true')
    """ 
    Test the obstacle avoidance system

    Ctrl-F to see where to adjust the following parameters
    1) CHOOSE WHETHER TO TEST WITH ANALYTICAL OR BIO-INSPIRED GOAL VECTOR CALCULATION
    2) ADJUST TEST PARAMETER RANGES
        2A) test a range of parameter values in different combinations              
        2B) choose a few combinations to test
    """
    random_nav_parser = experiments.add_parser('random_nav')
    random_nav_parser.add_argument('--seed', type=int, default=None)

    main_parser.add_argument('--visualize', action='store_true')
    args = main_parser.parse_args()

    if args.experiment == 'none':
        env_model = "Savinov_val3"

        dt = 1e-2
        # initialize grid cell network and create target spiking
        gc_network = GridCellNetwork(from_data=True, dt=dt)
        target_spiking = create_gc_spiking(args.start, args.goal)

        compass = Compass.factory(mode=args.decoder, gc_network=gc_network, goal_pos=args.goal)

        with PybulletEnvironment(env_model, dt=dt, start=args.start, visualize=args.visualize) as env:
            vector_navigation(env, compass, gc_network, target_gc_spiking=target_spiking, step_limit=float('inf'),
                          plot_it=False, exploration_phase=False)

    elif args.experiment == "vector_navigation":
        import time

        nr_trials = 1
        env_model = "plane"
        dt = 1e-2

        error_array = []
        actual_error_array = []
        actual_error_goal_array = []
        time_array = []

        for i in range(0, nr_trials):
            # initialize grid cell network and create target spiking
            gc_network = GridCellNetwork(from_data=True, dt=dt)

            start: Vector2D = np.array([0.0, 0.0])

            """Picks a location at circular edge of environment"""
            distance = args.goal_distance
            angle = np.random.uniform(0, 2 * np.pi)
            goal = start + np.array([np.cos(angle), np.sin(angle)]) * distance

            with PybulletEnvironment(env_model, dt=dt, start=start, visualize=args.visualize) as env:
                model = args.decoder
                # changes the update fraction and arrival threshold according to the chosen model
                compass: Compass
                if model == "pod":
                    compass = PodGcCompass(gc_network=gc_network)
                    compass.arrival_threshold = 0.2
                elif model == "linear_lookahead":
                    compass = LinearLookaheadGcCompass(gc_network=gc_network, arena_size=env.arena_size)
                    update_fraction = 0.2
                elif model == "combo":
                    compass = ComboGcCompass(gc_network=gc_network)
                else: assert False, f"unrecognized model {model}"

                # 3) CHOOSE WHETHER TO PERFORM

                # 3A) A SIMPLE RETURN TO START
                if args.simplicity == 'simple':
                    """ navigate ~ 15 m away from the start position """
                    target_spiking = gc_network.consolidate_gc_spiking()
                    vector_navigation(env, AnalyticalCompass(start_pos=start, goal_pos=goal), gc_network)
                    start_time = time.time()
                    gc_network.set_as_target_state(target_spiking)
                    vector_navigation(env, compass, gc_network, step_limit=8000, plot_it=False)
                    trial_time = time.time() - start_time
                    """------------------------------------------------------------------------------------------"""
                else:
                    # 3B) GENERATING THE GOAL SPIKINGS, NAVIGATING TO THE GOAL, THEN RETURN TO START
                    """ alternatively: generate spiking at goal then navigate there before returning to the start """
                    start_spiking = gc_network.consolidate_gc_spiking()
                    target_spiking = create_gc_spiking(start, goal)

                    start_time = time.time()
                    gc_network.set_as_target_state(target_spiking)
                    vector_navigation(env, compass, gc_network, step_limit=8000, plot_it=False)
                    actual_error_goal = np.linalg.norm(env.robot.position - goal)
                    actual_error_goal_array.append(actual_error_goal)
                    # TODO Pierre: env.nr_ofsteps = 0
                    compass.reset_goal(new_goal=(start if type(compass)==AnalyticalCompass else start_spiking))
                    vector_navigation(env, compass, gc_network, step_limit=8000, plot_it=False)

                    trial_time = time.time() - start_time
                    """------------------------------------------------------------------------------------------"""
                final_position = env.robot.position

            # Decoding Error
            error = np.linalg.norm((final_position + compass.calculate_goal_vector()) - goal)
            error_array.append(error)

            # Navigation Error
            actual_error = np.linalg.norm(final_position - goal)
            actual_error_array.append(actual_error)

            time_array.append(trial_time)
            print(trial_time)

            progress_str = "Progress: " + str(int((i + 1) * 100 / nr_trials)) + "% | Latest error: " + str(error)
            print(progress_str)

        # Directly plot and print the errors (distance between goal and actual end position)
        # error_plot(error_array)
        # print(error_array)

        # 4) ADJUST THE NAME FOR SAVING YOUR RESULTS
        # Save the data of all trials in a dedicated folder
        name = "test"
        directory = "experiments/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = "experiments/" + name
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Decoding Error (Return to start)
        np.save("experiments/" + name + "/error_array", error_array)
        # Navigation Error (Return to start)
        np.save("experiments/" + name + "/actual_error_array", actual_error_array)
        # Navigation Error of navigating to the goal in case of 3B
        np.save("experiments/" + name + "/actual_error_goal_array", actual_error_goal_array)
        # Time Cost
        np.save("experiments/" + name + "/time_array", time_array)

    elif args.experiment == "obstacle_avoidance":

        def three_trials(model, working_combinations, num_ray_dir, cone, mapping, combine):
            nr_steps = 0
            for trial in range(4):
                if trial == 0:
                    start, goal = [-1, -2], [1.5, 1.25]
                elif trial == 1:
                    start, goal = [-1, -2], [0, 2]
                elif trial == 2:
                    start, goal = [0, -2], [-2, 2]
                elif trial == 3:
                    start, goal = [-2.5, -2], [-1, -1]

                env_model = f"obstacle_map_{trial}"

                # initialize grid cell network and create target spiking
                if model == "combo":
                    gc_network = GridCellNetwork(from_data=True, dt=1e-2)
                    target_spiking = create_gc_spiking(start, goal)
                else:
                    gc_network = None
                    target_spiking = None

                compass = AnalyticalCompass(start_pos=start, goal_pos=goal)
                with PybulletEnvironment(env_model, start=start, visualize=args.visualize) as env:
                    controller = LocalController(
                        transform_goal_vector=([] if trial == 0 else [ctrl_rules.ObstacleAvoidance()]),
                        on_reset_goal=[ctrl_rules.TurnToGoal()],
                        hooks=[ctrl_rules.StuckDetector()],
                    )
                    env.mapping = mapping

                    over, nr_steps_this_trial = vector_navigation(env, compass, collect_nr_steps=True, gc_network=gc_network, controller=controller, target_gc_spiking=target_spiking,
                                            plot_it=True, step_limit=500)
                    # assert over == 1
                    print(trial, over, mapping, combine, num_ray_dir, cone)

                    nr_steps += nr_steps_this_trial

            # save all combinations that passed all three tests and how many time steps the agent took in total
            working_combinations.append((nr_ofrays, cone, mapping, combine, nr_steps))

            print(working_combinations)
            nr_steps_list = [sub[4] for sub in working_combinations]
            min_val = min(nr_steps_list)
            index = nr_steps_list.index(min_val)
            print("combination with fewest steps: ", working_combinations[index])


        # 1) CHOOSE WHETHER TO TEST WITH ANALYTICAL OR BIO-INSPIRED GOAL VECTOR CALCULATION
        model = "combo"  # "combo"

        working_combinations = []
        # 2) ADJUST TEST PARAMETER RANGES
        if args.all:
            # 2A) test a range of parameter values in different combinations
            for nr_ofrays in [21]:
                for cone in [120]:
                    num_ray_dir = int(nr_ofrays // (360 / cone))
                    for mapping in [1.5]:
                        for combine in [1.5]:
                            three_trials(model, working_combinations, nr_ofrays, cone, mapping, combine)
        else:
            # 2B) choose a few combinations to test
            combinations = [(21, 120, 1.5, 1.5)]
            for c in combinations:
                nr_ofrays, cone, mapping, combine = c
                num_ray_dir = int(nr_ofrays // (360 / cone))
                three_trials(model, working_combinations, nr_ofrays, cone, mapping, combine)

        directory = "experiments/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save("experiments/working_combinations", working_combinations)

    elif args.experiment == "random_nav":
        controller = LocalController(
            transform_goal_vector=[ctrl_rules.ObstacleAvoidance()],
            on_reset_goal=[ctrl_rules.TurnToGoal()],
            hooks=[ctrl_rules.StuckDetector()],
        )

        from system.controller.reachability_estimator.data_generation.dataset import TrajectoriesDataset, get_path
        from system.controller.simulation.environment.map_occupancy import random_points
        from itertools import chain

        dataset = TrajectoriesDataset([os.path.join(get_path(), "data", "trajectories", "trajectories.hd5")], env_cache=None)
        samepath_points = dataset.subset(map_name="Savinov_val3", seed=args.seed)

        rng = random.Random()
        samepath_points = [ rng.choice(samepath_points) for _ in range(50) ]
        random_points = [ random_points(env_model="Savinov_val3", rng=rng) for _ in range(50) ]
        points = list(chain.from_iterable(zip(samepath_points, random_points))) # zip the two lists so the progress is more accurate (not all good points are shown at the beginning)

        #if args.seed:
        #    random.seed(args.seed)
        randomPointsTest(controller, AnalyticalCompass, points, visualize=args.visualize, plot_it=False)
