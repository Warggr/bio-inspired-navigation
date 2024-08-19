import sys
from typing import Callable, Any
from system.bio_model.cognitive_map import LifelongCognitiveMap, CognitiveMap
from system.types import AllowedMapName
import numpy as np
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.utils import average

def all_unobstructed_lines(cogmap: CognitiveMap, env_model: AllowedMapName):
    graph = cogmap.node_network
    map = MapLayout(env_model)

    if len(graph.edges) == 0:
        return []
    lines = np.zeros((len(graph.edges), 4))
    for i, (x1, x2) in enumerate(graph.edges):
        lines[i, 0:2] = x1.pos
        lines[i, 2:4] = x2.pos
    collides = map.no_touch_batch(lines)
    return collides

def unobstructed_lines(cogmap: CognitiveMap, env_model: AllowedMapName) -> float:
    try:
        return average(all_unobstructed_lines(cogmap, env_model))
    except ZeroDivisionError:
        return float('nan')

def plot_unobstructed_lines(cogmap: CognitiveMap, env_model: AllowedMapName):
    import matplotlib.pyplot as plt
    import matplotlib.collections as mc
    from system.plotting.plotHelper import add_environment

    collides = all_unobstructed_lines(cogmap, env_model)

    _, ax = plt.subplots()
    add_environment(ax, env_model)
    lines = [[ start.pos, stop.pos ] for start, stop in cogmap.node_network.edges ]
    colors = { True: 'g', False: 'r'}
    lc = mc.LineCollection(lines, colors=[ colors[collide] for collide in collides ], linewidths=2)
    ax.add_artist(lc)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        maps = (line.strip() for line in sys.stdin.readlines())
    elif len(sys.argv) == 3:
        maps = [sys.argv[2]]
    else:
        print("Wrong usage")
        sys.exit(1)

    from node_distance import mean_distance_between_nodes
    from eval_cogmap import scalar_coverage
    import os

    functions: dict[str, Callable[[CognitiveMap, AllowedMapName], Any]] = {
        "mean-distance": mean_distance_between_nodes,
        "coverage": scalar_coverage,
        "edges": unobstructed_lines,
    }

    for map_file in maps:
        cogmap = LifelongCognitiveMap(reachability_estimator=None)
        try:
            cogmap.load(map_file, absolute_path=True)
        except FileNotFoundError:
            cogmap.load(map_file, absolute_path=False)
        map_file = os.path.basename(map_file)
        env_model = map_file.split('.')[0]
        if env_model not in AllowedMapName.options:
            env_model = 'Savinov_val3'

        value = functions[sys.argv[1]](cogmap, env_model)
        print(value)
