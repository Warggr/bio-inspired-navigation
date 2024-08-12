from system.bio_model.grid_cell_model import GridCellNetwork
from system.bio_model.cognitive_map import LifelongCognitiveMap
from system.bio_model.place_cell_model import PlaceCellNetwork
from system.controller.local_controller.compass import AnalyticalCompass
from system.controller.local_controller.local_controller import LocalController, controller_rules
from system.controller.local_controller.local_navigation import PhaseOffsetDetectorNetwork, setup_gc_network
from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory, SpikingsReachabilityEstimator
from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.topological.topological_navigation import TopologicalNavigation

import argparse
from system.parsers import controller_parser

parser = argparse.ArgumentParser(parents=[controller_parser])
parser.add_argument('env_model', nargs='?', default='Savinov_val3')
parser.add_argument('map_file', nargs='?', default="after_exploration.gpickle")
parser.add_argument('start_stop', nargs='?', type=lambda ab: map(int, ab.split(',')), default=(73, 21))
parser.add_argument('--env-variant', '--variant')
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

re_type = 'view_overlap' #"neural_network"
re_weights_file = "re_mse_weights.50"
if args.env_model != 'Savinov_val3' and not args.map_file.startswith(args.env_model):
    args.map_file = args.env_model + '.' + args.map_file

model = "combo"
log = True
input_config = SampleConfig(grid_cell_spikings=True)
re = reachability_estimator_factory(re_type, weights_file=re_weights_file, config=input_config, env_model=args.env_model)
pc_network = PlaceCellNetwork(from_data=True, reach_estimator=SpikingsReachabilityEstimator(), map_name=args.env_model)
cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from=args.map_file, debug=False)
gc_network = GridCellNetwork(dt=1e-2, from_data=True)
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
    transform_goal_vector=[controller_rules.ObstacleAvoidance(ray_length=args.ray_length, tactile_cone=args.tactile_cone, follow_walls=args.follow_walls)],
    hooks=[controller_rules.StuckDetector()],
)
tj = TopologicalNavigation(args.env_model, pc_network, cognitive_map, compass, log=log)

cognitive_map.draw()
start, stop = args.start_stop
start_pc = list(cognitive_map.node_network.nodes.keys())[start]
goal_pc = list(cognitive_map.node_network.nodes.keys())[stop]
gc_network.set_as_current_state(start_pc.spikings)
compass.reset_position(start_pc.pos)

with PybulletEnvironment(args.env_model, variant=args.env_variant, start=start_pc.pos, visualize=args.visualize, build_data_set=True) as env:
    # backwards compatibility: a lot of cognitive maps were saved without angles
    # TODO: put it into the cognitive_map code?
    cognitive_map.get_place_cell_network().add_angles_and_lidar(env)

    success = tj.navigate(start_pc, goal_pc, gc_network=gc_network, env=env, controller=controller)
    if success:
        print(f"Success! simulation time: {env.t}. Navigation steps: {vector_step_counter.counter}")
    else:
        print("Fail :(")
