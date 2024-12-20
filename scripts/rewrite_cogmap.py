import sys, os

from matplotlib import interactive
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from system.bio_model.cognitive_map import LifelongCognitiveMap
import system.plotting.plotResults as plot

map_file = "after_lifelong_learning.gpickle"
map_file_after_correction = "handcrafted.gpickle"
env_model = 'Savinov_val3'
interactive = False

edges_to_delete = set()
edges_to_keep = set()

cognitive_map = LifelongCognitiveMap(reachability_estimator=None, load_data_from=map_file)
for start, end in cognitive_map.node_network.edges:
    if (start, end) in edges_to_delete or (start, end) in edges_to_keep:
        print("Reverse edge")
        continue
    if interactive:
        plot.plotTrajectoryInEnvironment(cognitive_map=cognitive_map, path=[start, end], xy_coordinates=[start.pos], env_model=env_model)
    k = input('Enter x to delete')
    if k == 'x':
        edges_to_delete.add((start, end))
    else:
        edges_to_keep.add((start, end))

for start, end in edges_to_delete:
    try:
        cognitive_map.remove_bidirectional_edge(start, end)
    except KeyError:
        print("Could not remove edge :(")

cognitive_map.draw()
cognitive_map.save(map_file_after_correction)
