import pandas as pd
from system.controller.reachability_estimator.reachability_estimation import (
    ViewOverlapReachabilityEstimator, NetworkReachabilityEstimator, SimulationReachabilityEstimator
)
from system.controller.reachability_estimator.training.train_multiframe_dst import Hyperparameters
from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig
from system.tests.eval_cogmap import cognitive_map_coverage

from tqdm import tqdm

def cognitive_map_agreement(
    cogmap: 'CognitiveMap',
    map_name: 'AllowedMapName',
):
    view_re = ViewOverlapReachabilityEstimator(env_model=map_name)
    net_re = NetworkReachabilityEstimator.from_file('re_mse_weights.50', config=SampleConfig(grid_cell_spikings=True))
    connections = []
    for (p1, p2), attrs in tqdm(cogmap.node_network.edges.items()):
        attrs = dict(attrs)
        attrs['view_success'] = view_re.reachability_factor(p1, p2)
        attrs['net_success'] = net_re.reachability_factor(p1, p2)
        connections.append(attrs)
    df = pd.DataFrame(connections)
    return (df['view_success'] >= view_re.threshold_reachable).cov(df['net_success'] >= net_re.threshold_reachable)

from scipy.stats import variation
import networkx as nx
from itertools import pairwise

def cognitive_map_variation(
    cogmap: 'CognitiveMap',
):
    network = cogmap.node_network

    nodes = list(network.nodes)
    nodes_visited = {node: 0 for node in nodes}
    edges_visited = {edge: 0 for edge in network.edges}

    for i in range(len(nodes)):
        for j in range(i, len(nodes)):
            path = nx.shortest_path(network, source=nodes[i], target=nodes[j])
            for node in path:
                nodes_visited[node] += 1
            for n1, n2 in pairwise(path):
                edges_visited[n1, n2] += 1
    return variation(list(nodes_visited.values())), variation(list(nodes_visited.values()))

import os
import sys
import numpy as np
from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory
from system.bio_model.cognitive_map import LifelongCognitiveMap

df = {'map_name': [], 'coverage': [], 'agreement': [], 'node_variation': [], 'edge_variation': []}

for filename in os.listdir("system/bio_model/data/cognitive_map"):
    print('Processing', filename, '...', file=sys.stderr)
    match filename.split("."):
        case (env_model, _type, "gpickle"):
            pass
        case (_type, "gpickle"):
            env_model = "Savinov_val3"
        case _:
            continue
    cogmap = LifelongCognitiveMap(load_data_from=filename, reachability_estimator=None)
    re = reachability_estimator_factory('view_overlap', env_model=env_model)
    df['map_name'].append(filename)
    _, coverage = cognitive_map_coverage(cogmap.node_network.nodes.keys(), re, env_model)
    coverage = np.any(coverage, axis=1)
    df['coverage'].append(sum(coverage) / len(coverage))
    df['agreement'].append(cognitive_map_agreement(cogmap, env_model))

    node_var, edge_var = cognitive_map_variation(cogmap)
    df['node_variation'].append(node_var)
    df['edge_variation'].append(edge_var)
df = pd.DataFrame(df)
print(df.to_csv())
