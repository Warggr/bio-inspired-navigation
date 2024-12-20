# coding: utf-8
from system.bio_model.place_cell_model import PlaceCell, PlaceCellNetwork
from system.bio_model.place_cell_model import PlaceCell, PlaceCellNetwork
pc_network = PlaceCellNetwork(re=None, from_data=True)
pc_network = PlaceCellNetwork(reach_estimator=None, from_data=True)
pc_network
pc_network.place_cells
pc_network.place_cells[0]
pc_network.place_cells[0].lidar
from system.bio_model.cognitive_map import LifelongCognitiveMap
map_file = "cognitive_map_partial_2.gpickle"
cognitive_map = LifelongCognitiveMap(reachability_estimator=None, load_data_from=map_file, debug=True)
cognitive_map
cognitive_map.node_network
cognitive_map.node_network.nodes
cognitive_map.node_network.nodes[0]
cognitive_map.node_network.nodes.keys()
node = next(cognitive_map.node_network.nodes)
cognitive_map.node_network
note = next(iter(cognitive_map.node_network.nodes))
note
node = note
del note
node
node.lidar
dir(node)
from system.controller.simulation.pybullet_environment import PybulletEnvironment
env = PybulletEnvironment()
env.__enter__()
env.__exit__()
env.__exit__(None, None, None)
env = PybulletEnvironment(contains_robot=False).__enter__()
help(env.lidar)
node.observations
dir(node)
node.angle
node.pos
node.img
