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

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from system.bio_model.cognitive_map import CognitiveMapInterface
from system.bio_model.place_cell_model import PlaceCellNetwork

from system.controller.local_controller.decoder.linear_lookahead_no_rewards import *
from system.controller.local_controller.decoder.phase_offset_detector import PhaseOffsetDetectorNetwork
from system.controller.simulation.math_utils import Vector2D
from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.controller.local_controller.compass import Compass, AnalyticalCompass
from system.types import WaypointInfo, types

import system.plotting.plotResults as plot
import numpy as np
from abc import abstractmethod
from typing import Optional, List

plotting = True  # if True: plot everything
debug = os.getenv('DEBUG', False)  # if True: print debug output


def print_debug(*params):
    """ output only when in debug mode """
    if debug:
        print(*params)


class GoalVectorCache(Compass):
    @property
    def arrival_threshold(self):
        return self.impl.arrival_threshold

    @property
    def goal_pos(self):
        return self.impl.goal_pos

    def reset(self, new_goal : Vector2D):
        self.impl.reset(new_goal)
        self.goal_vector = None

    def __init__(self, impl : Compass, update_fraction = 0.5):
        # intentionally don't call super().__init__() because we don't really need it
        self.nr_ofsteps = 0 # we just need this line
        self.impl = impl
        self.update_fraction = update_fraction # how often the goal vector is recalculated
        self.goal_vector = None  # egocentric goal vector after last update
        self.distance_to_goal_original = None # distance to goal after last recalculation

    def calculate_goal_vector(self) -> Vector2D:
        if self.goal_vector is None:
            self.goal_vector = self.impl.calculate_goal_vector()
            self.distance_to_goal_original = np.linalg.norm(self.goal_vector)

        return self.goal_vector

    def update_position(self, robot : Robot):
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

class GcCompass(Compass):
    """ Uses decoded grid cell spikings as a goal vector. """

    def __init__(self, gc_network : GridCellNetwork, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gc_network = gc_network

    def update_position(self, robot : 'Robot'):
        pass # TODO Pierre: ensure that the GCNetwork is updated separately

    @staticmethod
    def factory(mode, gc_network : GridCellNetwork, *args,
        pod_network : Optional[PhaseOffsetDetectorNetwork] = None,
        arena_size: Optional[float] = None,
        **kwargs
    ):
        if mode == "pod":
            return GoalVectorCache( PodGcCompass(pod_network, gc_network, *args, **kwargs) )
        if mode == "linear_lookahead":
            return GoalVectorCache( LinearLookaheadGcCompass(arena_size, gc_network, *args, **kwargs) )
        elif mode == "combo":
            return ComboGcCompass(pod_network, gc_network, *args, **kwargs)
        else:
            return ValueError(f"Unknown mode: {mode}. Expected one of: analytical, pod, linear_lookahead, combo")


class PodGcCompass(GcCompass):
    arrival_threshold = 0.5

    def __init__(self, pod_network : 'PhaseOffsetDetectorNetwork', *args, **kwargs):
        if pod_network is None: # TODO Pierre this is ugly
            pod_network = PhaseOffsetDetectorNetwork(16, 9, 40)
        super().__init__(*args, **kwargs)
        self.pod_network = pod_network

    def calculate_goal_vector(self):
        """For Vector-based navigation, computes goal vector with one grid cell decoder"""
        return self.pod_network.compute_goal_vector(self.gc_network.gc_modules)    


class LinearLookaheadGcCompass(GcCompass):
    arrival_threshold = 0.2
    def __init__(self, arena_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arena_size = arena_size

    def calculate_goal_vector(self):
        return perform_look_ahead_2xnr(self.gc_network, self.arena_size)


class ComboGcCompass(GcCompass):
    def __init__(self, pod_network : 'PhaseOffsetDetectorNetwork', gc_network : GridCellNetwork, *args, **kwargs):
        super().__init__(gc_network, *args, **kwargs)
        # self.gc_network = gc_network # already done by the super().__init__
        self.pod_network = pod_network

        compass = PodGcCompass(pod_network, gc_network, *args, **kwargs)
        compass = GoalVectorCache(compass)
        self.impl = compass

    def reset(self, new_goal : Vector2D):
        super().reset(new_goal)
        if type(self.impl) == LinearLookaheadGcCompass:
            self.impl = PodGcCompass(pod_network=self.pod_network, gc_network=self.impl.gc_network, goal_pos=new_pos)
            self.impl = GoalVectorCache(self.impl)
        else:
            self.impl.reset(new_goal)

    def calculate_goal_vector(self, *args, **kwargs):
        # TODO this often returns 0,0 for some reason
        return self.impl.calculate_goal_vector(*args, **kwargs)

    @property
    def arrival_threshold(self):
        return self.impl.arrival_threshold

    def update_position(self, robot : Robot):
        self.impl.update_position(robot)
        goal_reached = self.impl.reached_goal()
        if goal_reached and type(self.impl) == PodGcCompass:
            # switch from pod to linear lookahead
            self.impl = LinearLookaheadGcCompass(gc_network=self.gc_network, goal_pos=self.impl.goal_pos)
            self.impl = GoalVectorCache(self.impl)
            self.impl.update_position(robot)
            goal_vector = self.impl.calculate_goal_vector()
            robot.env.step_forever(4)
            robot.turn_to_goal(goal_vector)
            return False # Continue with the LL
        return goal_reached


def create_gc_spiking(start : Vector2D, goal : Vector2D) -> types.Spikings:
    """ 
    Agent navigates from start to goal accross a plane without any obstacles, using the analyticallly 
    calculated goal vector to genereate the grid cell spikings necessary for the decoders. During actual
    navigation this would have happened in the exploration phase.
    """

    with PybulletEnvironment("plane", start=start) as env:
        robot = env.robot

        # Grid-Cell Initialization
        gc_network = setup_gc_network(env.dt)

        compass = AnalyticalCompass(start_pos=robot.position, goal_pos=goal)
        robot.turn_to_goal(compass.calculate_goal_vector())

        i = 0
        while True:
            i += 1
            if i == 5000:
                raise AssertionError("Agent should not get caught in a loop in an empty plane.")

            goal_vector = compass.calculate_goal_vector()
            if np.linalg.norm(goal_vector) == 0:
                break
            robot.navigation_step(goal_vector, obstacles=False)
            compass.update_position(robot.position)
            gc_network.track_movement(robot.xy_speed)
            reached_goal = compass.reached_goal()

            if reached_goal:
                if plotting: plot.plotTrajectoryInEnvironment(env)
                return gc_network.consolidate_gc_spiking()


def setup_gc_network(dt) -> GridCellNetwork:
    """ Initialize the grid cell newtork """
    # Grid-Cell Initialization
    M = 6  # 6 for default, number of modules
    n = 40  # 40 for default, size of sheet -> nr of neurons is squared
    gmin = 0.2  # 0.2 for default, maximum arena size, 0.5 -> ~10m | 0.05 -> ~105m
    gmax = 2.4  # 2.4 for default, determines resolution, dont pick to high (>2.4 at speed = 0.5m/s)

    # note that if gc modules are created from data n and M are overwritten
    gc_network = GridCellNetwork(n, M, dt, gmin, gmax=gmax, from_data=True)

    return gc_network


def vector_navigation(env : PybulletEnvironment, compass: Compass, gc_network : GridCellNetwork, target_gc_spiking=None,
    step_limit=float('inf'), plot_it=False,
                      collect_data_freq=False, collect_data_reachable=False, exploration_phase=False,
    pc_network: PlaceCellNetwork = None, cognitive_map: CognitiveMapInterface = None,
    *nav_args, **nav_kwargs
):
    """
    Agent navigates towards goal.

    arguments:
    env                    --  running PybulletEnvironment
    compass                --  A Compass pointing to the goal
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
    All remaining arguments are passed to the Robot.navigation_step() function

    Returns: (depending on the arguments):
    goal_reached : bool, data : List[WaypointInfo] if collect_data_freq
    goal_reached : bool, ???  if collect_data_reachable
    goal_reached : bool, last_pc : PlaceCell else
    """

    data : List[WaypointInfo] = []
    robot = env.robot

    if gc_network and (target_gc_spiking is not None):
        gc_network.set_as_target_state(target_gc_spiking)

    robot.nr_ofsteps = 0
    goal_vector = compass.calculate_goal_vector()
    robot.turn_to_goal(goal_vector)

    if collect_data_reachable:
        sample_after_turn = (robot.data_collector[-1][0], robot.data_collector[-1][1])
        first_goal_vector = goal_vector

    n = 0  # time steps
    goal_reached = False
    end_state = ""  # for plotting
    last_pc = None
    assert compass.goal_pos is not None
    while n < step_limit and not goal_reached:
        if True: #try:
            goal_reached = compass.step(robot, *nav_args, **nav_kwargs)
            gc_network.track_movement(robot.xy_speed)
        #except compass.RobotStuck:
        #    break

        if pc_network is not None and cognitive_map is not None:
            observations = robot.data_collector.get_observations()
            [firing_values, created_new_pc] = pc_network.track_movement(gc_network, observations,
                                                                        robot.position, exploration_phase)

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
        #if n % 100 == 0:
            #print("Vector nav step", n)

    assert compass.goal_pos is not None
    if goal_reached:
        end_state = "Agent reached the goal. Actual distance: " + str(
                np.linalg.norm(np.array(compass.goal_pos) - robot.position)) + "."
    else:
        end_state = "Agent got stuck"

    if plot_it:
        plot.plotTrajectoryInEnvironment(env, title=end_state)

    if collect_data_freq:
        return goal_reached, data
    if collect_data_reachable:
        return goal_reached, [sample_after_turn, first_goal_vector]

    if not last_pc and not exploration_phase and pc_network:
        pc_network.create_new_pc(gc_network.consolidate_gc_spiking(), observations, env.xy_coordinates[-1])
        last_pc = pc_network.place_cells[-1]
    return goal_reached, last_pc


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
    """ 
    Test the obstacle avoidance system

    Ctrl-F to see where to adjust the following parameters
    1) CHOOSE WHETHER TO TEST WITH ANALYTICAL OR BIO-INSPIRED GOAL VECTOR CALCULATION
    2) ADJUST TEST PARAMETER RANGES
        2A) test a range of parameter values in different combinations              
        2B) choose a few combinations to test
    """
    main_parser.add_argument('--visualize', action='store_true')
    args = main_parser.parse_args()

    if args.experiment == 'none':
        env_model = "Savinov_val3"

        dt = 1e-2
        # initialize grid cell network and create target spiking
        gc_network = setup_gc_network(dt)
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
            gc_network = setup_gc_network(dt)

            # 1) CHOOSE THE DECODER YOU WANT TO TEST
            model = "pod"
            # model = "linear_lookahead"
            # model = "combo"

            env = PybulletEnvironment(env_model, dt=dt, start=[0, 0])

            # changes the update fraction and arrival threshold according to the chosen model
            if model == "pod":
                env.pod_arrival_threshold = 0.2
            elif model == "linear_lookahead":
                update_fraction = 0.2

            """Picks a location at circular edge of environment"""
            # 2) CHOOSE THE DISTANCE TO THE GOAL
            distance = 15  # goal distance
            angle = np.random.uniform(0, 2 * np.pi)
            goal = env.xy_coordinates[0] + np.array([np.cos(angle), np.sin(angle)]) * distance

            start = np.array([0, 0])

            # 3) CHOOSE WHETHER TO PERFORM

            # 3A) A SIMPLE RETURN TO START
            simple = True
            if simple:
                """ navigate ~ 15 m away from the start position """
                target_spiking = gc_network.consolidate_gc_spiking()
                vector_navigation(env, goal, gc_network, model="analytical")
                start_time = time.time()
                vector_navigation(env, list(start), gc_network, target_gc_spiking=target_spiking, model=model, step_limit=8000,
                                  plot_it=False)
                trial_time = time.time() - start_time
                """------------------------------------------------------------------------------------------"""
            else:
                # 3B) GENERATING THE GOAL SPIKINGS, NAVIGATING TO THE GOAL, THEN RETURN TO START
                """ alternatively: generate spiking at goal then navigate there before returning to the start """
                start_spiking = gc_network.consolidate_gc_spiking()
                target_spiking = create_gc_spiking(start, goal)
                env = PybulletEnvironment(env_model, model, start=list(start))

                if model == "pod":
                    env.pod_arrival_threshold = 0.2
                elif model == "linear_lookahead":
                    update_fraction = 0.2

                start_time = time.time()
                vector_navigation(env, list(goal), gc_network, target_gc_spiking=target_spiking, model=model, step_limit=8000,
                                  plot_it=False)
                actual_error_goal = np.linalg.norm(env.xy_coordinates[-1] - env.goal_pos)
                actual_error_goal_array.append(actual_error_goal)
                env.nr_ofsteps = 0
                vector_navigation(env, list(start), gc_network, target_gc_spiking=start_spiking, model=model, step_limit=8000,
                                  plot_it=False)

                trial_time = time.time() - start_time
                """------------------------------------------------------------------------------------------"""

            # Decoding Error
            error = np.linalg.norm((env.xy_coordinates[-1] + env.goal_vector) - env.goal_pos)
            error_array.append(error)

            # Navigation Error
            actual_error = np.linalg.norm(env.xy_coordinates[-1] - env.goal_pos)
            actual_error_array.append(actual_error)

            time_array.append(trial_time)
            print(trial_time)

            env.end_simulation()

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
                    gc_network = setup_gc_network(1e-2)
                    target_spiking = create_gc_spiking(start, goal)
                else:
                    gc_network = None
                    target_spiking = None

                env = PybulletEnvironment(env_model, "analytical", start=start)

                env.mapping = mapping
                env.combine = combine
                env.num_ray_dir = num_ray_dir
                env.tactile_cone = cone

                over, _ = vector_navigation(env, goal, gc_network=gc_network, target_gc_spiking=target_spiking, model=model,
                                            plot_it=True, step_limit=10000, obstacles=(True if trial == 0 else False))
                # assert over == 1
                print(trial, over, mapping, combine, num_ray_dir, cone)

                nr_steps += env.nr_ofsteps

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
        all = False
        if all:
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
