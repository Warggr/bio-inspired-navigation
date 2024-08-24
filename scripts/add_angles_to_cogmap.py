from system.bio_model.cognitive_map import LifelongCognitiveMap
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from cogmap_utils import guess_env_model
import sys

map_name = sys.argv[1]
env_model = guess_env_model(map_name)
cogmap = LifelongCognitiveMap(None, load_data_from=map_name)

with PybulletEnvironment(env_model, contains_robot=False, visualize=False) as env:
    cogmap.get_place_cell_network().add_angles_and_lidar(env)

pcs = list(cogmap.node_network.nodes)

for pc1 in cogmap.node_network.nodes:
    adj = list(cogmap.node_network.adj[pc1])
    for pc2 in adj:
        try:
            pc2.angle
        except AttributeError:
            attributes = dict(cogmap.node_network.edges[pc1, pc2])
            cogmap.node_network.remove_edge(pc1, pc2)
            pc2_but_correct = pcs[pcs.index(pc2)] #  since PlaceCells override __eq__, there can be (and there is) a PlaceCell in pcs that is corrected but is "equal" to the wrong PC
            cogmap.node_network.add_edge(pc1, pc2_but_correct, **attributs)
            print('Corrected place cell in adjacency list')

cogmap.save(map_name)
