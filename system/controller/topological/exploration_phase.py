""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.local_controller.local_navigation import vector_navigation
from system.controller.local_controller.local_controller import LocalController, controller_rules
from system.controller.local_controller.compass import AnalyticalCompass
from system.bio_model.place_cell_model import PlaceCellNetwork
from system.bio_model.cognitive_map import LifelongCognitiveMap, CognitiveMapInterface
import system.plotting.plotResults as plot
from system.types import AllowedMapName, Vector2D
from system.debug import DEBUG, PLOTTING

plotting = 'exploration' in PLOTTING  # if True: plot paths


class TooManyPlaceCells(Exception):
    def __init__(self, progress: float):
        self.progress = progress

def waypoint_movement(
    path: list[Vector2D], env_model: AllowedMapName, gc_network: GridCellNetwork, pc_network: PlaceCellNetwork,
    cognitive_map: CognitiveMapInterface,
    visualize=False,
):
    """ Navigates the agent on the given path and builds the cognitive map.
        The local controller navigates the path analytically and updates the pc_network and the cognitive_map.

    arguments:
    path: [PlaceCell]                    -- path to follow
    env_model: str                       -- environment model
    gc_network: GridCellNetwork          -- grid cell network
    pc_network: PlaceCellNetwork         -- place cell network
    cognitive_map: CognitiveMapInterface -- cognitive map object
    mode: str                            -- mode goal vector detection, possible values:
                                            ['pod', 'linear_lookahead', 'combo']
    """

    map_layout = MapLayout(env_model)
    goals = []
    for i in range(len(path) - 1):
        new_wp = map_layout.find_path(path[i], path[i + 1])
        if new_wp is None:
            raise ValueError("No path found!")
        goals += new_wp
        if plotting:
            map_layout.draw_map_path(path[i], path[i + 1], i)

    if plotting:
        map_layout.draw_path(goals)

    controller = LocalController(
        on_reset_goal=[controller_rules.TurnToGoal()],
        transform_goal_vector=[], # not using ObstacleAvoidance because the waypoints are not in obstacles anyway
        hooks=[controller_rules.StuckDetector()],
    )
    #controller.transform_goal_vector.append(controller_rules.TurnWhenNecessary())

    with PybulletEnvironment(env_model, dt=gc_network.dt, build_data_set=True, start=path[0], visualize=visualize) as env:

        from tqdm import tqdm

        compass = AnalyticalCompass(start_pos=path[0])
        for i, goal in enumerate(tqdm(goals)):
            try:
                compass.reset_goal(goal)
                goal_reached, last_pc = vector_navigation(env, compass, gc_network, step_limit=5000,
                            plot_it=plotting, exploration_phase=True, pc_network=pc_network, cognitive_map=cognitive_map, controller=controller)
                assert goal_reached
            except AssertionError:
                print(f"At {','.join(map(str, env.robot.position))} -> {','.join(map(str, goals[i]))}")
                raise
            except CognitiveMapInterface.TooManyPlaceCells:
                raise TooManyPlaceCells(progress=i/len(goals))
            if plotting and (i + 1) % 100 == 0:
                cognitive_map.draw()
                plot.plotTrajectoryInEnvironment(env, goal=False, cognitive_map=cognitive_map, trajectory=False)

    return pc_network, cognitive_map


if __name__ == "__main__":
    """
    Create a cognitive map by exploring the environment. 
    Agent follows a hard-coded path to explore the environment and build the cognitive map. 
    """
    from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--env-model", default="Savinov_val3", choices=["Savinov_val3", "linear_sunburst", 'plane', 'final_layout'])
    parser.add_argument('--re', dest='re_type', default='spikings')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--mini', help='Use only a few trajectories', action='store_true')
    modes = parser.add_subparsers(dest='subcommand')
    parser.add_argument('--output-filename', "-o", help='The file to save the network to', nargs='?', default='after_exploration.gpickle')

    binary_param_search = modes.add_parser('npc', help='Target a number of place cells')
    binary_param_search.add_argument('-n', dest='desired_nb_of_place_cells', help='desired number of place cells', type=int, default=30)
    binary_param_search.add_argument('-t', '--threshold-same-hint', help='The first value that will be tried as the sameness threshold', type=float)
    binary_param_search.add_argument('--lower-bound', '->', type=float, default=0.2)
    binary_param_search.add_argument('--upper-bound', '-<', type=float, default=1.0)

    fixed_threshold = modes.add_parser('threshold', help='Set a threshold_same')
    fixed_threshold.add_argument('threshold_same', type=float)

    args = parser.parse_args()

    if args.env_model == "Savinov_val3":
        goals = [
            [-2, 0], [-6, -2.5], [-4, 0.5], [-6.5, 0.5], [-7.5, -2.5], [-2, -1.5], [1, -1.5],
            [0.5, 1.5], [2.5, -1.5], [1.5, 0], [5, -1.5], [4.5, -0.5], [-0.5, 0], [-8.5, 3],
            [-8.5, -4], [-7.5, -3.5], [1.5, -3.5], [-6, -2.5]
        ]
        if args.mini:
            goals = goals[7:11]
    elif args.env_model == "final_layout":
        from numpy import loadtxt
        from system.controller.simulation.pybullet_environment import resource_path
        goals = loadtxt(resource_path(args.env_model, 'path.csv'), delimiter=',')
        assert not args.mini, "No mini path defined yet"
    elif args.env_model == "linear_sunburst":
        goals = [
             [5.5, 4.5],
             [1.5, 4.5],
             [9.5, 4.5],
             [9.5, 7.5],
             [10.5, 7.5],
             [10.5, 10],
             [8.5, 10],
             [8.5, 7.5],
             [6.5, 7.5],
             [6.5, 10],
             [4.5, 10],
             [4.5, 7.5],
             [2.5, 7.5],
             [2.5, 10],
             [0.5, 10],
             [0.5, 7.5],
             [2.5, 7.5],
             [2.5, 10],
             [4.5, 10],
             [4.5, 7.5],
             [6.5, 7.5],
             [6.5, 10],
             [8.5, 10],
             [8.5, 7.5],
             [9.5, 7.5],
        ]
        if args.mini:
            goals = goals[2:8] + [goals[3]]
    elif args.env_model == 'plane':
        from numpy import random
        rng = random.RandomState(1)
        goals = rng.normal(size=(20, 2))
    else:
        raise ValueError(f"Unsupported map: {args.env_model}")

    if args.env_model != "Savinov_val3" and not args.output_filename.startswith(args.env_model + '.'):
        args.output_filename = args.env_model + '.' + args.output_filename

    gc_network = GridCellNetwork(from_data=True)

    re = reachability_estimator_factory(args.re_type, debug=('plan' in DEBUG), env_model=args.env_model)

    def create_cogmap(threshold, max_capacity=200):
        re.threshold_same = threshold
        pc_network = PlaceCellNetwork(reach_estimator=re)
        cognitive_map = LifelongCognitiveMap(reachability_estimator=re, max_capacity=max_capacity, metadata={'threshold': re.threshold_same, 're': str(re)}, debug=('cogmap' in DEBUG))
        pc_network, cognitive_map = waypoint_movement(goals, args.env_model, gc_network, pc_network, cognitive_map, visualize=args.visualize)
        cognitive_map.postprocess_topological_navigation()
        return cognitive_map, pc_network

    if args.subcommand == 'npc':
        too_strict_threshold = args.upper_bound
        too_lax_threshold = args.lower_bound
        while True:
            if args.threshold_same_hint is not None:
                threshold_same = args.threshold_same_hint
                args.threshold_same_hint = None
            else:
                threshold_same = (too_lax_threshold + too_strict_threshold) / 2

            print(f"Trying threshold {threshold_same}...")
            try:
                cognitive_map, pc_network = create_cogmap(threshold=threshold_same, max_capacity=2*args.desired_nb_of_place_cells)
            except TooManyPlaceCells as err:
                too_strict_threshold = threshold_same
                print("Too high!")
                continue
            if len(cognitive_map.node_network.nodes) < args.desired_nb_of_place_cells / 2:
                too_lax_threshold = threshold_same
                print("Too low!")
                continue
            assert len(cognitive_map.node_network.nodes) > 1
            break
    elif args.subcommand == 'threshold':
        cognitive_map, pc_network = create_cogmap(threshold=args.threshold_same)
    else:
        raise ValueError('Unrecognized subcommand')

    pc_network.save_pc_network(filename=(f'-{args.env_model}' if args.env_model != 'Savinov_val3' else ''))
    cognitive_map.save(filename=args.output_filename)

    if plotting:
        cognitive_map.draw()
        with PybulletEnvironment(args.env_model, dt=gc_network.dt, build_data_set=True) as env:
            plot.plotTrajectoryInEnvironment(env, goal=False, cognitive_map=cognitive_map, trajectory=False)
