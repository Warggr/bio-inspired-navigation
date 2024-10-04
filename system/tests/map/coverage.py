from system.bio_model.cognitive_map import CognitiveMap
from system.bio_model.place_cell_model import PlaceCell
from system.controller.reachability_estimator.reachability_estimation import ReachabilityEstimator, reachability_estimator_factory
from system.controller.simulation.environment_config import environment_dimensions
from system.controller.local_controller.local_navigation import create_gc_spiking, setup_gc_network
from system.controller.reachability_estimator.data_generation.dataset import place_info
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.types import AllowedMapName, PositionAndOrientation
from system.controller.reachability_estimator._types import PlaceInfo
from system.controller.reachability_estimator.data_generation.dataset import TrajectoriesDataset, get_path
from system.controller.local_controller.local_navigation import RandomTaker
import numpy as np
from tqdm import tqdm
import os

def get_points_for_map(map_name: AllowedMapName, total_points=100) -> list[PositionAndOrientation]:
    dataset = TrajectoriesDataset([os.path.join(get_path(), "data", "trajectories", "trajectories.hd5")], env_cache=None)._init_once()
    dataset = dataset.subset_single_point(map_name=map_name, seed=1)
    dataset = RandomTaker(dataset, seed=1)
    dataset = iter(dataset)
    return [next(dataset) for _ in range(total_points)]

def cognitive_map_coverage(
    place_cells: list[PlaceCell],
    reachability_estimator: ReachabilityEstimator,
    map_name: AllowedMapName
):
    points = get_points_for_map(map_name)
    covered = np.zeros((len(points), len(place_cells)), dtype=bool)

    #env = PybulletEnvironment(map_name, visualize=False, contains_robot=False)
    #gc_network = setup_gc_network(dt=env.dt)

    last_pos = (0, 0)
    for i, pos in enumerate(tqdm(points)):
        #angle = 0
        #spiking = create_gc_spiking(start=last_pos, goal=pos, gc_network_at_start=gc_network, plotting=False)
        place = PlaceInfo(pos[0], pos[1], spikings=None, img=None, lidar=None)

        covered[i, :] = reachability_estimator.is_same_batch(place, place_cells)

        #last_pos = pos
        #gc_network.set_as_current_state(spiking)
    return points, covered

def scalar_coverage(
    cogmap: CognitiveMap,
    env_model: AllowedMapName,
):
    re = reachability_estimator_factory('view_overlap', env_model=env_model)
    _, covered = cognitive_map_coverage(list(cogmap.node_network.nodes), re, env_model)
    coverage = np.any(covered, axis=1)
    return sum(coverage) / len(coverage)
