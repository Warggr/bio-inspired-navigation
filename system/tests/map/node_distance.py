import numpy as np
import networkx as nx
from itertools import product

def average(li):
    return sum(li) / len(li)

def to_mst(graph):
    graph = graph.to_undirected()

    for (p1, p2), edge_data in graph.edges.items():
        edge_data['distance'] = np.linalg.norm(np.array(p1.pos) - np.array(p2.pos))

    return nx.minimum_spanning_tree(graph, weight='distance')

def mean_distance_between_nodes(cogmap, env_model):
    G = cogmap.node_network.copy()
    G.add_edges_from((a, b) for a, b in product(G.nodes, G.nodes) if a!=b)
    mst = to_mst(G)
    if len(mst.edges) == 0:
        return float('nan')
    return average([edge['distance'] for edge in mst.edges.values()])
