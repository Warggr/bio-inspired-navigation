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

from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot, wall_colors_by_description
from system.bio_model.cognitive_map import LifelongCognitiveMap, CognitiveMapInterface
from system.bio_model.place_cell_model import PlaceCellNetwork, PlaceCell, PlaceInfo
from system.controller.local_controller.compass import Compass
from system.controller.local_controller.local_controller import LocalController, ObstacleAvoidance, StuckDetector, ObstacleBackoff, TurnToGoal
from system.controller.local_controller.local_navigation import vector_navigation
import system.plotting.plotResults as plot
from system.types import AllowedMapName
import argparse
import pickle

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

        i: int # the index of the last found PC in the path
        if self.log:
            if 'hooks' not in nav_kwargs:
                nav_kwargs['hooks'] = []
            def print_position(i, robot):
                if (i+1) % 100 == 0:
                    print('[proprioception] Robot position:', robot.position)
            nav_kwargs['hooks'].append(print_position)
            if self.log and 'drift' in DEBUG:
                from system.controller.local_controller.position_estimation import DoubleCompass
                from system.controller.local_controller.local_navigation import PodGcCompass
                starting_position_compass = DoubleCompass(PodGcCompass())
                starting_position_compass.reset_position((gc_network.consolidate_gc_spiking(), start))

                def print_drift(i, robot):
                    if (i + 1) % 100 == 0:
                        current_position = (gc_network.consolidate_gc_spiking(), robot.position)
                        starting_position_compass.reset_goal(current_position)
                        current_compass = DoubleCompass(PodGcCompass())
                        current_compass.reset_position(current_position)
                        current_compass.reset_goal(path[i+1])
                        error_from_start = starting_position_compass.error()
                        error_to_goal = current_compass.error()
                        estimated_position_from_start = starting_position_compass.calculate_estimated_position()
                        estimated_position_from_goal = current_compass.calculate_estimated_position()
                        print(f'[drift] Drift: ' +
                            f'{estimated_position_from_start=}, {estimated_position_from_goal=},' +
                            f'{error_from_start=}, {error_to_goal=}'
                        )
                nav_kwargs['hooks'].append(print_drift)

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
                print(f'[navigation] Vector navigation: goal={path_indices[i+1]}, {success=}')
            curr_path_length += 1
            if not success:
                last_pc, new_path = self.locate_node(self.compass, pc, goal)
                if self.log:
                    if not last_pc:
                        print('[navigation] Last PC: status=not found/assuming current place cell still correct')
                    else:
                        print('[navigation] Last PC: status=found,', f'#={self.cognitive_map._place_cell_number(last_pc)}')
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
                    print(f'[navigation] Recomputed path: so_far={_printable_path(path_indices[:i])}, continue={_printable_path(path_indices[i:])}')
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
        if curr_path_length >= path_length_limit:
            print(f"[navigation] LIMIT WAS REACHED STOPPING HERE: remaining_path={_printable_path(path_indices[i:])}")
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
parser.add_argument('env_model', choices=AllowedMapName.options, default='Savinov_val3')
parser.add_argument('--env-variant', '--variant', help='Environment model variant')
parser.add_argument('map_file')
parser.add_argument('--compass', choices=['analytical', 'pod', 'linear_lookahead', 'combo'], default='combo')
parser.add_argument('--log', help='Log everything to stdout', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--re-type', help='Type of the reachability estimator used for connecting nodes', default='neural_network(re_mse_weights.50)')
parser.add_argument('--pc-creation-re', help='Use an alternative reachability estimator for deciding when to create new nodes')
parser.add_argument('--max-path-length', '-m', help='Maximimum path length after which topological navigation will be aborted; number or "inf"', type=lambda m: float('inf') if m == 'inf' else int(m), default=30)
parser.add_argument('--dump-all-steps', help='(folder to which to dump each step)')


if __name__ == "__main__":
    """
    Test navigation through the maze.
    The exploration should be completed before running this script. 
    """
    import os
    from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory
    from system.parsers import controller_parser, controller_creator

    class DontSave: pass # Sentinel value

    parser = argparse.ArgumentParser(parents=[parser, controller_parser])
    parser.add_argument('-o', dest='map_file_after_lifelong_learning', nargs='?', default=DontSave, help='Save the map at end of navigation into a file')
    parser.add_argument('--log-metrics', action='store_true')
    parser.add_argument('--load', help='Load and replay log file, then continue navigation')
    parser.add_argument('--wall-colors', choices=['1color', '3colors', 'patterns'], default='1color')
    subparsers = parser.add_subparsers(dest='mode')

    random_parser = subparsers.add_parser('random', help='Perform multiple navigations between random nodes')
    random_parser.add_argument('--seed', type=int, help='Seed for random index generation.')  # for reproducibility / debugging purposes
    random_parser.add_argument('--num-topo-nav', '-n', help='number of topological navigations', type=int, default=100)

    one_traj_parser = subparsers.add_parser('path', help='(Try to) Navigate from a start to a goal node')
    one_traj_parser.add_argument('navigations', nargs='?',
                                 type=lambda outer: map(lambda inner: map(int, inner.split(',')), outer.split(';')),
                                 default=((73, 21),),
                                 help='Start, intermediary nodes, and goal, separated by a comma.'
                                     +' Multiple navigations can be separated by a semicolon.'
                                     +' Negative indices (-1 for the last node) are allowed.'
                                 )
    one_traj_parser.add_argument('--head', type=int, help='Only perform navigation to the first n nodes')
    one_traj_parser.add_argument('--restore', type=int, help='Restore from dumped step')

    args = parser.parse_args()

    map_file, map_file_after_lifelong_learning = args.map_file, args.map_file_after_lifelong_learning
    if map_file_after_lifelong_learning is None:
        map_file_after_lifelong_learning = 'after_lifelong_learning.gpickle'
    if map_file_after_lifelong_learning == DontSave:
        map_file_after_lifelong_learning = None
    if args.env_model != 'Savinov_val3':
        if not map_file.startswith(args.env_model + '.'):
            map_file = args.env_model + '.' + map_file
        if map_file_after_lifelong_learning is not None and not map_file_after_lifelong_learning.startswith(args.env_model + '.'):
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
    pc_network = cognitive_map.get_place_cell_network(pc_creation_re)

    gc_network = GridCellNetwork(from_data=True, dt=1e-2)
    pod = PhaseOffsetDetectorNetwork(16, 9, 40)
    compass = Compass.factory(args.compass, gc_network=gc_network, pod_network=pod)

    controller = controller_creator(args, env_model=args.env_model)

    class Counter:
        def __init__(self): self.counter = 0
        def __call__(self, *args, **kwargs): self.counter += 1
        def reset(self): self.counter = 0

    vector_step_counter = Counter()
    controller.on_reset_goal.append(vector_step_counter)

    if args.position_estimation:
        raise NotImplementedError()
        """
        from system.controller.local_controller.position_estimation import PositionEstimation

        controller.transform_goal_vector.append(
            PositionEstimation(pc_network, gc_network, re, compass, current_position=start_pc)
        )
        """

    tj = TopologicalNavigation(args.env_model, pc_network=pc_network, cognitive_map=cognitive_map, compass=compass, log=args.log)

    if args.log_metrics:
        from system.tests.map import unobstructed_lines, mean_distance_between_nodes, scalar_coverage, agreement
        def log_metrics(i, endpoints, endpoint_indices, success):
            print('Metrics:')
            for fun in (unobstructed_lines, mean_distance_between_nodes, scalar_coverage, agreement):
                print(fun.__name__, ':', fun(cognitive_map, args.env_model))
        tj.step_hooks.append(log_metrics)

    if args.dump_all_steps:
        def hook(i, success: bool, endpoints: tuple['PlaceCell', 'PlaceCell'], endpoint_indices: tuple[int, int]):
            if not success:
                return
            filename = os.path.join(args.dump_all_steps, 'step{i}.pkl')
            data = {
                'i': i,
                'env': env._dumps_state(),
                'gc_network': gc_network.consolidate_gc_spiking(),
                'current_pc_index': endpoint_indices[1],
            }
            with open(filename, 'wb') as file:
                pickle.dump(data, file)
        tj.step_hooks.append(hook)

    env_kwargs = {'wall_kwargs': wall_colors_by_description(args.wall_colors)}

    place_cells: list[PlaceCell] = list(tj.cognitive_map.node_network.nodes)
    assert len(place_cells) > 1
    navigations: list[list[int]]
    # Will be used in the following way (pseudocode):
    # for points in navigations:
    #     reset robot to points[0]
    #     for point in points[1:]:
    #         navigate_to_point()
    if args.mode == 'random':
        if not args.seed:
            args.seed = np.random.default_rng().integers(low=0, high=0b100000000)
            print("Using seed", args.seed)
        random = np.random.default_rng(seed=args.seed)
        navigations = []
        for _ in range(args.num_topo_nav):
            start_index, goal_index = None, None
            while start_index == goal_index:
                start_index, goal_index = random.integers(0, len(place_cells)-1, size=2)
            navigations.append([start_index, goal_index])
    elif args.mode == 'path':
        navigations = args.navigations
    else:
        raise AssertionError

    if args.mode == 'path' and args.head:
        assert len(navigations) == 1 and len(navigations[0]) == 1, "`head` only makes sense for one navigation"

    with PybulletEnvironment(args.env_model, variant=args.env_variant, build_data_set=True, visualize=args.visualize, contains_robot=False, **env_kwargs) as env:
        if plotting:
            from system.plotting.plotHelper import environment_plot
            ax = environment_plot(args.env_model, args.env_variant)
            plot.plotCognitiveMap(ax, cognitive_map)

        successful = 0
        for i, continuous_navigation in enumerate(navigations):
            start_index, *goals = continuous_navigation
            start = place_cells[start_index]

            tj.compass.reset_position(compass.parse(start))
            gc_network.set_as_current_state(start.spikings)

            if args.mode == 'path' and args.restore and i == 0:
                filename = f'logs/step{args.restore}.pkl'
                with open(filename, 'rb') as file:
                    data = pickle.load(file)
                start_index = data['current_pc_index']
                start = place_cells[start_index]
                robot = Robot.loads(env, data['env'], compass=compass)
                start_position = PlaceInfo(*robot.position_and_angle, spikings=data['gc_network'], img=None, lidar=None)
            else:
                robot = Robot(env, start.pos, start.angle, compass=compass)
                start_position = start

            with robot:
                for j, goal_index in enumerate(goals):
                    goal = place_cells[goal_index]
                    vector_step_counter.reset()
                    success = tj.navigate(start, goal,
                        gc_network=gc_network, controller=controller,
                        head=args.head, path_length_limit=args.max_path_length,
                        cognitive_map_filename=map_file_after_lifelong_learning,
                        env=env,
                    )
                    if success:
                        print(f"Success! simulation time: {env.t}. Navigation steps: {vector_step_counter.counter}")
                    else:
                        print("Fail :(")
                        break
                else: # no break
                    successful += 1
                #tj.cognitive_map.draw()
                if len(navigations) != 1:
                    print(f"Navigation {i} finished")
                if plotting:
                    ax = environment_plot(env.env_model, args.env_variant)
                    plot.plotCognitiveMap(ax, cognitive_map=tj.cognitive_map)

    if len(navigations) > 1:
        print(f"{successful} successful navigations")
