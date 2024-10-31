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

from typing import Optional, Protocol, Literal

from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.local_controller.decoder.phase_offset_detector import PhaseOffsetDetectorNetwork

from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.bio_model.cognitive_map import LifelongCognitiveMap, CognitiveMapInterface
from system.bio_model.place_cell_model import PlaceCellNetwork, PlaceCell
from system.controller.local_controller.compass import Compass
from system.controller.local_controller.local_controller import LocalController, ObstacleAvoidance, StuckDetector, ObstacleBackoff, TurnToGoal
from system.controller.local_controller.local_navigation import vector_navigation
import system.plotting.plotResults as plot
from system.types import AllowedMapName
import argparse

from system.debug import DEBUG, PLOTTING
plotting = 'topo' in PLOTTING # if True plot results

def _printable_path(path: list[int]) -> str:
    """ Helper function to print a path of node indices"""
    return ','.join((str(i) for i in path))

class TopoNavStepHook(Protocol):
    def __call__(self, i: int, *, endpoints: tuple[PlaceCell, PlaceCell], endpoint_indices: tuple[int, int], success: bool):
        ...

class TopologicalNavigation:
    def __init__(
        self, env_model: AllowedMapName,
        pc_network: Optional[PlaceCellNetwork], cognitive_map: CognitiveMapInterface,
        compass: Compass,
        log = False,
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
        self.env_model = env_model
        self.step_limit = 2000  # max number of vector navigation steps
        self.compass = compass
        self.log = log
        self.step_hooks: list[TopoNavStepHook] = []

    def navigate(self, start: PlaceCell, goal: PlaceCell, gc_network: GridCellNetwork,
        controller: Optional[LocalController] = None,
        cognitive_map_filename: Optional[str] = None,
        env: Optional[PybulletEnvironment] = None,
        robot: Optional[Robot] = None,
        head: Optional[int] = None,
        path_length_limit: int|Literal[float('inf')] = 30,
        *nav_args, **nav_kwargs,
    ):
        """ Navigates the agent through the environment with topological navigation.

        arguments:
        start_ind: int              -- start node in the cognitive map
        goal_ind: int               -- goal node in the cognitive map
        cognitive_map_filename: str -- name of file to save the cognitive map to
        head: int                   -- only perform the first n steps

        returns:
        bool -- whether the goal was reached
        int  -- index of start node
        int  -- index of goal node
        """


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

        path_indices = [self.cognitive_map._place_cell_number(p) for p in path]
        if self.log:
            print('Path:', _printable_path(path_indices))

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
            # only reset the compass to the (center of the) place cell if the robot was just created and is actually at the center
            self.compass.reset_position(self.compass.parse(start))
        else:
            #assert np.linalg.norm(np.array(env.robot.position) - np.array(src_pos)) <= 0.5, "robot provided but not at start position"
            pass

        if plotting:
            plot.plotTrajectoryInEnvironment(env, cognitive_map=self.cognitive_map, path=path)

        if controller is None:
            controller = LocalController(
                on_reset_goal=[TurnToGoal()],
                transform_goal_vector=[ObstacleAvoidance()],
                hooks=[StuckDetector()],
            )

        if self.log:
            if 'hooks' not in nav_kwargs:
                nav_kwargs['hooks'] = []
            def print_position(i, robot):
                if (i+1) % 100 == 0:
                    print('Robot position:', robot.position)
            nav_kwargs['hooks'].append(print_position)

        #self.gc_network.set_as_current_state(path[0].gc_connections)
        last_pc = path[0]
        i = 0
        curr_path_length = 0
        while i + 1 < len(path) and curr_path_length < path_length_limit:
            self.compass.reset_goal(new_goal=self.compass.parse(path[i+1]))
            goal_spiking = path[i+1].gc_connections
            if gc_network:
                gc_network.set_as_target_state(goal_spiking)
            success, pc = vector_navigation(env, self.compass, gc_network=gc_network,
                                         controller=controller, exploration_phase=False, pc_network=self.pc_network,
                                         cognitive_map=self.cognitive_map, plot_it=plotting,
                                         step_limit=self.step_limit, goal_pos=path[i+1].pos, *nav_args, **nav_kwargs)
            self.cognitive_map.postprocess_vector_navigation(node_p=path[i], node_q=path[i + 1],
                                                             observation_p=last_pc, observation_q=pc, success=success)

            for hook in self.step_hooks:
                hook(curr_path_length, success=success, endpoints=(path[i], path[i+1]), endpoint_indices=(path_indices[i], path_indices[i+1]))

            if self.log:
                print(f'Vector navigation: goal={path_indices[i+1]}, {success=}')
            curr_path_length += 1
            if not success:
                last_pc, new_path = self.locate_node(self.compass, pc, goal)
                if self.log:
                    if not last_pc:
                        print('Last PC: status=not found/assuming current place cell still correct')
                    else:
                        print('Last PC: status=found,', f'#={self.cognitive_map._place_cell_number(last_pc)}')
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
                        new_path = [goal]
                        if len(path) == 2:
                            # if the previous path was already just start <-> goal
                            # (which can happen if the goal is not connected to anything)
                            # and it failed once, don't just try the same thing again
                            # instead, try to put some node in the middle
                            new_path = [np.random.choice(list(self.cognitive_map.node_network.nodes))] + [goal]
                    new_path = [last_pc] + new_path

                path[i:] = new_path
                if self.log:
                    path_indices[i:] = [self.cognitive_map._place_cell_number(p) for p in new_path]
                    print(f'Recomputed path: so_far={_printable_path(path_indices[:i])}, continue={_printable_path(path_indices[i:])}')
                if plotting:
                    plot.plotTrajectoryInEnvironment(env, cognitive_map=self.cognitive_map, path=path)
            else:
                last_pc = pc
                i += 1
            if i == len(path) - 1:
                break
            elif i == head:
                break

        if plotting:
            plot.plotTrajectoryInEnvironment(env, goal=False, start=start.env_coordinates, end=goal.env_coordinates)
            plot.plotTrajectoryInEnvironment(env, goal=False, cognitive_map=self.cognitive_map,
                                             start=path[0].env_coordinates, end=path[-1].env_coordinates)

        self.cognitive_map.postprocess_topological_navigation()
        if cognitive_map_filename is not None:
            self.cognitive_map.save(filename=cognitive_map_filename)
        if curr_path_length >= path_length_limit:
            print(f"LIMIT WAS REACHED STOPPING HERE: remaining_path={_printable_path(path_indices[i:])}")
            return False
        return True

    def locate_node(self, compass: Compass, pc: PlaceCell, goal: PlaceCell):
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


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('env_model', choices=['Savinov_val3', 'linear_sunburst'], default='Savinov_val3')
parser.add_argument('--env-variant', '--variant', help='Environment model variant')
parser.add_argument('map_file', nargs='?', default='after_exploration.gpickle')
parser.add_argument('--compass', choices=['analytical', 'pod', 'linear_lookahead', 'combo'], default='combo')
parser.add_argument('--log', help='Log everything to stdout', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--re-type', help='Type of the reachability estimator used for connecting nodes', default='neural_network(re_mse_weights.50)')
parser.add_argument('--pc-creation-re', help='Use an alternative reachability estimator for deciding when to create new nodes')
parser.add_argument('--max-path-length', '-m', help='Maximimum path length after which topological navigation will be aborted; number or "inf"', type=lambda m: float('inf') if m == 'inf' else int(m), default=30)


if __name__ == "__main__":
    """
    Test navigation through the maze.
    The exploration should be completed before running this script. 
    """
    from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory

    parser = argparse.ArgumentParser(parents=[parser])
    parser.add_argument('map_file_after_lifelong_learning', nargs='?', default='after_lifelong_learning.gpickle')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random index generation.')  # for reproducibility / debugging purposes
    parser.add_argument('--num-topo-nav', '-n', help='number of topological navigations', type=int, default=100)
    parser.add_argument('--log-metrics', action='store_true')
    parser.add_argument('--load', help='Load and replay log file, then continue navigation')
    args = parser.parse_args()

    map_file, map_file_after_lifelong_learning = args.map_file, args.map_file_after_lifelong_learning
    if args.env_model != 'Savinov_val3':
        if not map_file.startswith(args.env_model + '.'):
            map_file = args.env_model + '.' + map_file
        if not map_file_after_lifelong_learning.startswith(args.env_model + '.'):
            map_file_after_lifelong_learning = args.env_model + '.' + map_file_after_lifelong_learning

    re = reachability_estimator_factory(args.re_type, env_model=args.env_model)
    #pc_network = PlaceCellNetwork(from_data=True, map_name=args.env_model)
    cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from=map_file, debug=('cogmap' in DEBUG or args.log))

    if args.load:
        print('Loading log file', args.load)
        saved_debug_state = cognitive_map.debug; cognitive_map.debug = False
        with open(args.load) as log_file:
            start_step = cognitive_map.retrace_logs(map(lambda line: line.strip(), log_file.readlines()))
        cognitive_map.debug = saved_debug_state
    else:
        start_step = 0

    if args.pc_creation_re is not None:
        pc_creation_re = reachability_estimator_factory(args.pc_creation_re, env_model=args.env_model)
    else:
        pc_creation_re = re
    pc_network = cognitive_map.get_place_cell_network()
    assert len(cognitive_map.node_network.nodes) > 1
    gc_network = GridCellNetwork(from_data=True, dt=1e-2)
    pod = PhaseOffsetDetectorNetwork(16, 9, 40)
    compass = Compass.factory(args.compass, gc_network=gc_network, pod_network=pod)

    tj = TopologicalNavigation(args.env_model, pc_network, cognitive_map, compass, log=args.log)

    if args.log_metrics:
        from system.tests.map import unobstructed_lines, mean_distance_between_nodes, scalar_coverage, agreement
        def log_metrics(i, endpoints, endpoint_indices, success):
            print('Metrics:')
            for fun in (unobstructed_lines, mean_distance_between_nodes, scalar_coverage, agreement):
                print(fun.__name__, ':', fun(cognitive_map, args.env_model))
        tj.step_hooks.append(log_metrics)

    with PybulletEnvironment(args.env_model, variant=args.env_variant, build_data_set=True, visualize=args.visualize, contains_robot=False) as env:
        if plotting:
            from system.plotting.plotHelper import environment_plot
            ax = environment_plot(args.env_model, args.env_variant)
            plot.plotCognitiveMap(ax, cognitive_map)

        if not args.seed:
            args.seed = np.random.default_rng().integers(low=0, high=0b100000000)
            print("Using seed", args.seed)
        random = np.random.default_rng(seed=args.seed)

        place_cells = list(tj.cognitive_map.node_network.nodes)

        successful = 0
        nb_correct_place_cells = len(place_cells) # if a large number of useless place cells are added to the cognitive map,
        # we want to navigate only between the original place cells. Else we would "dilute" places that are difficult to reach
        for navigation_i in range(args.num_topo_nav):
            start_index, goal_index = None, None
            while start_index == goal_index:
                start_index, goal_index = random.integers(0, nb_correct_place_cells, size=2)
                assert len(tj.cognitive_map.node_network.nodes) > 1

            start, goal = place_cells[start_index], place_cells[goal_index]

            compass.reset_position(compass.parse(start))
            success = tj.navigate(start, goal, cognitive_map_filename=map_file_after_lifelong_learning, env=env, gc_network=gc_network)
            if success:
                successful += 1
            #tj.cognitive_map.draw()
            print(f"Navigation {navigation_i} finished")
            if plotting:
                ax = environment_plot(env.env_model, args.env_variant)
                plot.plotCognitiveMap(ax, cognitive_map=tj.cognitive_map)

    print(f"{successful} successful navigations")
    print("Navigation finished")
