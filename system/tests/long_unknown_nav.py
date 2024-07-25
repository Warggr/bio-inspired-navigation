import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.bio_model.cognitive_map import LifelongCognitiveMap
from system.bio_model.place_cell_model import PlaceCellNetwork
from system.controller.local_controller.compass import AnalyticalCompass
from system.controller.local_controller.local_controller import LocalController, controller_rules
from system.controller.local_controller.local_navigation import PhaseOffsetDetectorNetwork, setup_gc_network
from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory, SpikingsReachabilityEstimator
from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.topological.topological_navigation import TopologicalNavigation

re_type = 'view_overlap' #"neural_network"
re_weights_file = "re_mse_weights.50"
map_file = "after_exploration.gpickle"
env_model = "Savinov_val3"
model = "combo"
input_config = SampleConfig(grid_cell_spikings=True)
visualize = True
re = reachability_estimator_factory(re_type, weights_file=re_weights_file, config=input_config)
pc_network = PlaceCellNetwork(from_data=True, reach_estimator=SpikingsReachabilityEstimator())
cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from=map_file, debug=False)
gc_network = setup_gc_network(1e-2)
pod = PhaseOffsetDetectorNetwork(16, 9, 40)
#compass = ComboGcCompass(gc_network, pod)
#compass = LinearLookaheadGcCompass(arena_size=15, gc_network=gc_network)
#compass = GoalVectorCache(compass)

class Counter:
    def __init__(self):
        self.counter = 0
    def __call__(self, *args, **kwargs):
        self.counter += 1

vector_step_counter = Counter()
compass = AnalyticalCompass()
controller = LocalController(
    on_reset_goal=[controller_rules.TurnToGoal(), vector_step_counter],
    transform_goal_vector=[controller_rules.ObstacleAvoidance()],
    hooks=[controller_rules.StuckDetector()],
)
tj = TopologicalNavigation(env_model, pc_network, cognitive_map, compass)

cognitive_map.draw()
start, stop = 73, 21
start_pc = list(cognitive_map.node_network.nodes.keys())[start]
goal_pc = list(cognitive_map.node_network.nodes.keys())[stop]
gc_network.set_as_current_state(start_pc.spikings)
compass.reset_position(start_pc.pos)

with PybulletEnvironment(start=start_pc.pos, visualize=visualize, build_data_set=True) as env:
    success = tj.navigate(start_pc, goal_pc, gc_network=gc_network, env=env, controller=controller)
    if success:
        print(f"Success! simulation time: {env.t}. Navigation steps: {vector_step_counter.counter}")
    else:
        print("Fail :(")
