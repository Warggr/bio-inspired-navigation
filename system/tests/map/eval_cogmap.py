from system.bio_model.cognitive_map import CognitiveMap
from system.bio_model.place_cell_model import PlaceCell
from system.controller.reachability_estimator.reachability_estimation import ReachabilityEstimator, reachability_estimator_factory
from system.controller.simulation.environment_config import environment_dimensions
from system.controller.local_controller.local_navigation import create_gc_spiking, setup_gc_network
from system.controller.reachability_estimator.data_generation.dataset import place_info
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.types import AllowedMapName, Vector2D
import numpy as np
from tqdm import tqdm

def grid_for_map(map_name: AllowedMapName, total_points=100) -> list[Vector2D]:
    x1, x2, y1, y2 = environment_dimensions(map_name)
    x, y = x2-x1, y2-y1
    x_nb = int(np.sqrt(total_points * x/y))
    y_nb = int(total_points / x_nb)
    x, y = np.linspace(x1, x2, x_nb+1), np.linspace(y1, y2, y_nb+1) # borders of unit cells
    x, y = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()

    free: list[Vector2D] = []
    layout = MapLayout(map_name)
    for (xi, yi) in zip(x, y):
        if layout.suitable_position_for_robot((xi, yi)):
            free.append((xi, yi))
    return free

def cognitive_map_coverage(
    place_cells: list[PlaceCell],
    reachability_estimator: ReachabilityEstimator,
    map_name: AllowedMapName
):
    points = grid_for_map(map_name=map_name)
    covered = np.zeros((len(points), len(place_cells)), dtype=bool)

    env = PybulletEnvironment(map_name, visualize=False, contains_robot=False)
    gc_network = setup_gc_network(dt=env.dt)

    last_pos = (0, 0)
    for i, pos in enumerate(tqdm(points)):
        angle = 0
        spiking = create_gc_spiking(start=last_pos, goal=pos, gc_network_at_start=gc_network, plotting=False)
        place = place_info((pos, angle, spiking), env)

        covered[i, :] = reachability_estimator.is_same_batch(place, place_cells)

        last_pos = pos
        gc_network.set_as_current_state(spiking)
    return points, covered

def scalar_coverage(
    cogmap: CognitiveMap,
    env_model: AllowedMapName,
):
    re = reachability_estimator_factory('view_overlap', env_model=env_model)
    _, covered = cognitive_map_coverage(list(cogmap.node_network.nodes), re, env_model)
    coverage = np.any(covered, axis=1)
    return sum(coverage) / len(coverage)
