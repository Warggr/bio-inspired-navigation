""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
import numpy as np

import sys
import os
from typing import Optional

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.local_controller.decoder.phase_offset_detector import PhaseOffsetDetectorNetwork

from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.bio_model.cognitive_map import LifelongCognitiveMap, CognitiveMapInterface
from system.bio_model.place_cell_model import PlaceCellNetwork, PlaceCell
from system.controller.local_controller.compass import Compass, AnalyticalCompass
from system.controller.local_controller.local_controller import LocalController, ObstacleAvoidance, StuckDetector, ObstacleBackoff, TurnToGoal
from system.controller.local_controller.local_navigation import vector_navigation, setup_gc_network, PodGcCompass, LinearLookaheadGcCompass
import system.plotting.plotResults as plot
from system.types import Vector2D

# if True plot results
plotting = True
DEBUG = os.getenv('DEBUG', '').split('&')

class TopologicalNavigation(object):
    def __init__(self, env_model: str,
                 pc_network: PlaceCellNetwork, cognitive_map: CognitiveMapInterface,
                 #gc_network: GridCellNetwork,
        compass: Compass,
    ):
        """
        Handles interactions between local controller and cognitive_map to navigate the environment.
        Performs topological navigation

        arguments:
        env_model: str -- name of the environment model
        pc_network: PlaceCellNetwork -- place cell network
        cognitive_map: CognitiveMapInterface -- cognitive map object
        gc_network: GridCellNetwork -- grid cell network
        """
        self.pc_network = pc_network
        self.cognitive_map = cognitive_map
        #self.gc_network = gc_network
        self.env_model = env_model
        self.path_length_limit = 30  # max number of topological navigation steps
        self.step_limit = 2000  # max number of vector navigation steps
        self.compass = compass

    def navigate(self, start_ind: int, goal_ind: int, gc_network: GridCellNetwork,
        cognitive_map_filename: str = None,
        env: Optional[PybulletEnvironment] = None,
        robot: Optional[Robot] = None,
        *nav_args, **nav_kwargs,
    ):
        """ Navigates the agent through the environment with topological navigation.

        arguments:
        start_ind: int              -- index of start node in the cognitive map
        goal_ind: int               -- index of goal node in the cognitive map
        cognitive_map_filename: str -- name of file to save the cognitive map to

        returns:
        bool -- whether the goal was reached
        int  -- index of start node
        int  -- index of goal node
        """

        start = list(self.cognitive_map.node_network.nodes)[start_ind]
        goal = list(self.cognitive_map.node_network.nodes)[goal_ind]

        # Plan a topological path through the environment,
        # if no such path exists choose random start and goal until a path is found
        path = self.cognitive_map.find_path(start, goal)
        if not path:
            j = 0
            while path is None and j < 10:
                node = np.random.choice(list(self.cognitive_map.node_network.nodes))
                path = self.cognitive_map.find_path(node, goal)
                j += 1
            if path is None:
                path = [goal]
            path = [start] + path

        for i, p in enumerate(path):
            print("path_index", i, list(self.cognitive_map.node_network.nodes).index(p))

        src_pos = list(path[0].env_coordinates)

        if env is None:
            if robot is None:
                env = PybulletEnvironment(self.env_model, build_data_set=True, start=src_pos)
            else:
                env = robot.env
        if robot is None:
            robot = env.robot

        assert env.env_model == self.env_model

        # TODO Pierre: make env required, use "with Robot"
        if env.robot is None:
            env.robot = Robot(env=env, base_position=src_pos, build_data_set=True)
        else:
            #assert np.linalg.norm(np.array(env.robot.position) - np.array(src_pos)) <= 0.5, "robot provided but not at start position"
            pass

        if plotting:
            plot.plotTrajectoryInEnvironment(env, cognitive_map=self.cognitive_map, path=path)

        #self.gc_network.set_as_current_state(path[0].gc_connections)
        last_pc = path[0]
        i = 0
        curr_path_length = 0
        while i + 1 < len(path) and curr_path_length < self.path_length_limit:
            goal = path[i + 1]
            self.compass.reset_goal(new_goal=self.compass.parse(goal))
            goal_spiking = path[i + 1].gc_connections
            if gc_network:
                gc_network.set_as_target_state(goal_spiking)
            controller = LocalController(on_reset_goal=[TurnToGoal()], transform_goal_vector=[ObstacleAvoidance(follow_walls=True)], hooks=[StuckDetector()]) # ObstacleBackoff(backoff_on_distance=0.25, backoff_off_distance=0.35)
            stop, pc = vector_navigation(env, self.compass, gc_network=gc_network,
                                         controller=controller, exploration_phase=False, pc_network=self.pc_network,
                                         cognitive_map=self.cognitive_map, plot_it=False,
                                         step_limit=self.step_limit, goal_pos=goal.pos, *nav_args, **nav_kwargs)
            self.cognitive_map.postprocess_vector_navigation(node_p=path[i], node_q=path[i + 1],
                                                             observation_p=last_pc, observation_q=pc, success=stop == 1)

            curr_path_length += 1
            if stop != 1:
                last_pc, new_path = self.locate_node(self.compass, pc, goal)
                if not last_pc:
                    last_pc = path[i]

                # if no path to the goal exists, try to find a random node that has valid path to the goal
                # if none is found, go straight to the goal
                if new_path is None:
                    j = 0
                    while new_path is None and j < 10:
                        node = np.random.choice(list(self.cognitive_map.node_network.nodes))
                        new_path = self.cognitive_map.find_path(node, goal)
                        j += 1
                    if new_path is None:
                        new_path = [path[i]] + [goal]
                    new_path = [path[i]] + new_path

                path[i:] = new_path
                if plotting:
                    plot.plotTrajectoryInEnvironment(env, cognitive_map=self.cognitive_map, path=path)
            else:
                last_pc = pc
                i += 1
            if i == len(path) - 1:
                break

        if curr_path_length >= self.path_length_limit:
            print("LIMIT WAS REACHED STOPPING HERE")

        if plotting:
            plot.plotTrajectoryInEnvironment(env, goal=False, start=start.env_coordinates, end=goal.env_coordinates)
            plot.plotTrajectoryInEnvironment(env, goal=False, cognitive_map=self.cognitive_map,
                                             start=path[0].env_coordinates, end=path[-1].env_coordinates)

        self.cognitive_map.postprocess_topological_navigation()
        if cognitive_map_filename is not None:
            self.cognitive_map.save(filename=cognitive_map_filename)
        return curr_path_length < self.path_length_limit

    def locate_node(self, compass : Compass, pc: PlaceCell, goal: PlaceCell):
        """
        Maps a location of the given place cell to the node in the graph.
        Among multiple close nodes prioritize the one that has a valid path to the goal.

        arguments:
        compass: Compass   -- A Compass that will be reset to get distance to the goal.
        pc: PlaceCell      -- a place cell to be located
        goal: PlaceCell    -- goal node

        returns:
        PlaceCell          -- mapped node in the graph or the given place cell if no node was found
        [PlaceCell] | None -- path to the goal if exists
        """
        closest_node = None
        # TODO: assert compass.current_pos == pc
        #compass.reset_position(compass.parse(pc))
        for node in self.cognitive_map.node_network.nodes:
            compass.reset_goal(new_goal=compass.parse(node))
            goal_vector = compass.calculate_goal_vector()
            if np.linalg.norm(goal_vector) < compass.arrival_threshold: # TODO: maybe we could find the *closest* node?
                closest_node = node
                new_path = self.cognitive_map.find_path(node, goal)
                if new_path:
                    return node, new_path
        return closest_node or pc, None


if __name__ == "__main__":
    """
    Test navigation through the maze.
    The exploration should be completed before running this script. 
    """
    from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory
    from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random index generation.') # for reproducibility / debugging purposes
    args = parser.parse_args()

    re_type = "neural_network"
    re_weights_file = "re_mse_weights.50"
    map_file = "after_exploration.gpickle"
    map_file_after_lifelong_learning = "after_lifelong_learning.gpickle"
    env_model = "Savinov_val3"
    model = "combo"
    input_config = SampleConfig(grid_cell_spikings=True)

    re = reachability_estimator_factory(re_type, weights_file=re_weights_file, config=input_config)
    pc_network = PlaceCellNetwork(from_data=True, reach_estimator=re)
    cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from=map_file, debug=('cogmap' in DEBUG))
    assert len(cognitive_map.node_network.nodes) > 1
    gc_network = setup_gc_network(1e-2)
    pod = PhaseOffsetDetectorNetwork(16, 9, 40)

    tj = TopologicalNavigation(env_model, model, pc_network, cognitive_map, gc_network, pod)

    with PybulletEnvironment(env_model, build_data_set=True, visualize=args.visualize, contains_robot=False) as env:
        # TODO Pierre figure out how to remove unnecessary xy_coordinates
        plot.plotTrajectoryInEnvironment(env, xy_coordinates=[0,0], goal=False, cognitive_map=tj.cognitive_map, trajectory=False)

        if not args.seed:
            args.seed = np.random.default_rng().integers(low=0, high=0b100000000)
            print("Using seed", args.seed)
        random = np.random.default_rng(seed=args.seed)

        successful = 0
        for navigation_i in range(100):
            start_index, goal_index = None, None
            while start_index == goal_index:
                start_index, goal_index = random.integers(0, len(tj.cognitive_map.node_network.nodes), size=2)
                assert len(tj.cognitive_map.node_network.nodes) > 1

            success = tj.navigate(start_ind=start_index, goal_ind=goal_index, cognitive_map_filename=map_file_after_lifelong_learning, env=env)
            if success:
                successful += 1
            tj.cognitive_map.draw()
            print(f"Navigation {navigation_i} finished")
            plot.plotTrajectoryInEnvironment(env, goal=False, cognitive_map=tj.cognitive_map, trajectory=False)

    print(f"{successful} successful navigations")
    print("Navigation finished")
