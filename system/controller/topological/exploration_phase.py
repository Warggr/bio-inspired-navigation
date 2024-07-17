""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.local_controller.local_navigation import vector_navigation, setup_gc_network
from system.controller.local_controller.local_controller import LocalController, controller_rules
from system.controller.local_controller.compass import AnalyticalCompass
from system.bio_model.place_cell_model import PlaceCellNetwork
from system.bio_model.cognitive_map import LifelongCognitiveMap, CognitiveMapInterface
import system.plotting.plotResults as plot
from system.types import Vector2D

plotting = True  # if True: plot paths
debug = True  # if True: print debug output
dt = 1e-2


def print_debug(*params):
    """ output only when in debug mode """
    if debug:
        print(*params)


def waypoint_movement(path: list[Vector2D], env_model: str, gc_network: GridCellNetwork, pc_network: PlaceCellNetwork,
                      cognitive_map: CognitiveMapInterface):
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

    with PybulletEnvironment(env_model, dt=dt, build_data_set=True, start=path[0]) as env:

        from tqdm import tqdm

        for i, goal in enumerate(tqdm(goals)):
            #print_debug(f"new waypoint with coordinates {goal}.", f'{i / len(goals) * 100} % completed.')
            compass = AnalyticalCompass(start_pos=path[0], goal_pos=goal)
            vector_navigation(env, compass, gc_network, step_limit=5000,
                            plot_it=plotting, exploration_phase=True, pc_network=pc_network, cognitive_map=cognitive_map)
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
    parser.add_argument("env_model", default="Savinov_val3", choices=["Savinov_val3", "linear_sunburst_map"])
    args = parser.parse_args()

    re_type = "neural_network"
    re_weights_file = "re_mse_weights.50"

    if args.env_model == "Savinov_val3":
        goals = [
            [-2, 0], [-6, -2.5], [-4, 0.5], [-6.5, 0.5], [-7.5, -2.5], [-2, -1.5], [1, -1.5],
            [0.5, 1.5], [2.5, -1.5], [1.5, 0], [5, -1.5], [4.5, -0.5], [-0.5, 0], [-8.5, 3],
            [-8.5, -4], [-7.5, -3.5], [1.5, -3.5], [-6, -2.5]
        ]
        cognitive_map_filename = "after_exploration.gpickle"
    elif args.env_model == "linear_sunburst_map":
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

    gc_network = setup_gc_network(dt)
    re = reachability_estimator_factory(re_type, weights_file=re_weights_file, debug=debug, config=SampleConfig(grid_cell_spikings=True))
    pc_network = PlaceCellNetwork(reach_estimator=re)
    cognitive_map = LifelongCognitiveMap(reachability_estimator=re)

    pc_network, cognitive_map = waypoint_movement(goals, args.env_model, gc_network, pc_network, cognitive_map)
    cognitive_map.postprocess_topological_navigation()
    assert len(cognitive_map.node_network.nodes) > 1

    pc_network.save_pc_network()
    cognitive_map.save(filename=cognitive_map_filename)

    if plotting:
        cognitive_map.draw()
        with PybulletEnvironment(args.env_model, dt=dt, build_data_set=True) as env:
            plot.plotTrajectoryInEnvironment(env, goal=False, cognitive_map=cognitive_map, trajectory=False)
