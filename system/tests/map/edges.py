import sys
from typing import Any, Generator, Iterable
from system.bio_model.cognitive_map import LifelongCognitiveMap, CognitiveMap
from system.bio_model.place_cell_model import PlaceCell
from system.types import AllowedMapName
import numpy as np
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.reachability_estimator.reachability_estimation import (
    ViewOverlapReachabilityEstimator, NetworkReachabilityEstimator, SimulationReachabilityEstimator
)
import os
import itertools

def random_edges(rng: np.random.Generator, cogmap: CognitiveMap, n=100):
    indices = rng.integers(0, len(cogmap.node_network.nodes)-1, size=(100, 2))
    pcs = list(cogmap.node_network.nodes)
    for i, j in indices:
        nodei, nodej = pcs[i], pcs[j]
        yield (i, nodei), (j, nodej)

def agreement_values(
    cogmap: CognitiveMap, env: PybulletEnvironment,
    edges: Iterable[tuple[tuple[int, PlaceCell], tuple[int, PlaceCell]]]
) -> Generator[dict[str, Any], None, None]:
    res = dict(
    #    sim = SimulationReachabilityEstimator(env),
        view = ViewOverlapReachabilityEstimator(env_model=env.env_model),
        net = NetworkReachabilityEstimator.from_file('re_mse_weights.50'),
        net_2 = NetworkReachabilityEstimator.from_file('reachability_network-3colors+lidar--ego_bc+fc.25'),
        net_3 = NetworkReachabilityEstimator.from_file('reachability_network+spikings+lidar--raw_lidar+conv.25'),
    )

    for batch in itertools.batched(edges, 10):
        indices = []
        batches_i, batches_j = [], []
        for (i, nodei), (j, nodej) in edges:
            indices.append((i, j))
            batches_i.append(nodei); batches_j.append(nodej)
        batches = {}
        for key, re in res.items():
            batches[key] = re.reachability_factor_batch_mm(batches_i, batches_j)
        for k in range(len(indices)):
            i, j = indices[k]
            nodei, nodej = batches_i[k], batches_j[k]
            values = {}
            values['indices'] = (i, j)
            values['in_map'] = nodej in cogmap.node_network.adj[nodei]
            values['distance'] = np.linalg.norm(np.array(nodei.pos) - np.array(nodej.pos))
            for key, batch in batches.items():
                values[key + '_success'] = batch[k]
            yield values

if __name__ == "__main__":
    map_file = sys.argv[1]

    cogmap = LifelongCognitiveMap(reachability_estimator=None)
    try:
        cogmap.load(map_file, absolute_path=True)
    except FileNotFoundError:
        cogmap.load(map_file, absolute_path=False)
    map_file = os.path.basename(map_file)
    env_model = map_file.split('.')[0]
    if env_model not in AllowedMapName.options:
        env_model = 'Savinov_val3'

    rng = np.random.default_rng(seed=1)
    edges = random_edges(rng, cogmap, n=100)

    keys = None
    with PybulletEnvironment(env_model, contains_robot=False) as env:
        for line in agreement_values(cogmap, env, edges):
            if keys is None:
                keys = list(line.keys())
                print(','.join(keys))
            else:
                assert list(line.keys()) == keys
            print(','.join(map(str, line.values())))
