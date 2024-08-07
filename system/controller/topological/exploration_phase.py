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
import os

PLOTTING = os.getenv('PLOTTING', '').split('&')
plotting = 'exploration' in PLOTTING  # if True: plot paths
debug = True  # if True: print debug output



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
            #print_debug(f"new waypoint with coordinates {goal}.", f'{i / len(goals) * 100} % completed.')
            try:
                compass.reset_goal(goal)
                goal_reached, last_pc = vector_navigation(env, compass, gc_network, step_limit=5000,
                            plot_it=plotting, exploration_phase=True, pc_network=pc_network, cognitive_map=cognitive_map, controller=controller)
                assert goal_reached
            except AssertionError:
                print(f"At {','.join(map(str, env.robot.position))} -> {','.join(map(str, goals[i]))}")
                raise
            except PlaceCellNetwork.TooManyPlaceCells:
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
    from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_model", default="Savinov_val3", choices=["Savinov_val3", "linear_sunburst"])
    parser.add_argument('-n', dest='desired_nb_of_place_cells', help='desired number of place cells', type=int, default=30)
    parser.add_argument('-t', '--threshold-same-hint', help='The first value that will be tried as the sameness threshold', type=float)
    parser.add_argument('--re', dest='re_type', default='neural_network', choices=['neural_network', 'view_overlap'])
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    re_weights_file = "re_mse_weights.50"

    if args.env_model == "Savinov_val3":
        goals = [
            [-2, 0], [-6, -2.5], [-4, 0.5], [-6.5, 0.5], [-7.5, -2.5], [-2, -1.5], [1, -1.5],
            [0.5, 1.5], [2.5, -1.5], [1.5, 0], [5, -1.5], [4.5, -0.5], [-0.5, 0], [-8.5, 3],
            [-8.5, -4], [-7.5, -3.5], [1.5, -3.5], [-6, -2.5]
        ]
        cognitive_map_filename = "after_exploration.gpickle"
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
        cognitive_map_filename = "linear_sunburst.after_exploration.gpickle"
    else:
        raise ValueError(f"Unsupported map: {args.env_model}")

    gc_network = GridCellNetwork(from_data=False)
    re = reachability_estimator_factory(args.re_type, weights_file=re_weights_file, debug=('plan' in DEBUG), config=SampleConfig(grid_cell_spikings=True))

    too_strict_threshold = 1.4
    too_lax_threshold = 0.2
    while True:
        if args.threshold_same_hint is not None:
            re.threshold_same = args.threshold_same_hint
            args.threshold_same_hint = None
        else:
            re.threshold_same = (too_lax_threshold + too_strict_threshold) / 2
        pc_network = PlaceCellNetwork(reach_estimator=re, max_capacity=2*args.desired_nb_of_place_cells)
        cognitive_map = LifelongCognitiveMap(reachability_estimator=re, metadata={'threshold': re.threshold_same})

        print(f"Trying threshold {re.threshold_same}...")
        try:
            pc_network, cognitive_map = waypoint_movement(goals, args.env_model, gc_network, pc_network, cognitive_map, visualize=args.visualize)
        except TooManyPlaceCells as err:
            too_strict_threshold = re.threshold_same
            print("Too high!")
            continue
        if len(pc_network.place_cells) < args.desired_nb_of_place_cells / 2:
            too_lax_threshold = re.threshold_same
            print("Too low!")
            continue
        cognitive_map.postprocess_topological_navigation()
        assert len(cognitive_map.node_network.nodes) > 1
        break

    pc_network.save_pc_network(filename=(f'-{args.env_model}' if args.env_model != 'Savinov_val3' else ''))
    cognitive_map.save(filename=cognitive_map_filename)

    if plotting:
        cognitive_map.draw()
        with PybulletEnvironment(args.env_model, dt=gc_network.dt, build_data_set=True) as env:
            plot.plotTrajectoryInEnvironment(env, goal=False, cognitive_map=cognitive_map, trajectory=False)
