import argparse
import pickle

from system.bio_model.cognitive_map import LifelongCognitiveMap
from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.local_controller.compass import Compass
from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig
from system.controller.reachability_estimator._types import PlaceInfo
from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.controller.topological.topological_navigation import TopologicalNavigation, parser
from system.parsers import controller_parser, controller_creator
from system.debug import DEBUG


parser = argparse.ArgumentParser(parents=[controller_parser, parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_stop', nargs='?', type=lambda ab: map(int, ab.split(',')), default=(73, 21), help='Start and stop node indices, joined by a comma. Negative indices (-1 for the last node) are allowed.')
parser.add_argument('--dump-all-steps', action='store_true')
parser.add_argument('--restore', type=int, help='Restore from dumped step')
parser.add_argument('--head', type=int, help='Only perform navigation to the first n nodes')
args = parser.parse_args()

if args.env_model != 'Savinov_val3' and not args.map_file.startswith(args.env_model):
    args.map_file = args.env_model + '.' + args.map_file

re = reachability_estimator_factory(args.re_type, env_model=args.env_model)
cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from=args.map_file, debug=('cogmap' in DEBUG or args.log))
pc_network = cognitive_map.get_place_cell_network()
gc_network = GridCellNetwork(dt=1e-2, from_data=True)
#compass = ComboGcCompass(gc_network, pod)
#compass = LinearLookaheadGcCompass(arena_size=15, gc_network=gc_network)
#compass = GoalVectorCache(compass)
compass = Compass.factory(args.compass, gc_network=gc_network)

class Counter:
    def __init__(self):
        self.counter = 0
    def __call__(self, *args, **kwargs):
        self.counter += 1


vector_step_counter = Counter()
controller = controller_creator(args, env_model=args.env_model)
controller.on_reset_goal.append(vector_step_counter)

tj = TopologicalNavigation(args.env_model, pc_network=pc_network, cognitive_map=cognitive_map, compass=compass, log=args.log)

if args.dump_all_steps:
    def hook(i, success: bool, endpoints: tuple['PlaceCell', 'PlaceCell'], endpoint_indices: tuple[int, int]):
        if not success:
            return
        filename = f'logs/step{i}.pkl'
        data = {
            'i': i,
            'env': env._dumps_state(),
            'gc_network': gc_network.consolidate_gc_spiking(),
            'current_pc_index': endpoint_indices[1],
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
    tj.step_hooks.append(hook)

#cognitive_map.draw()
start, stop = args.start_stop
goal_pc = list(cognitive_map.node_network.nodes.keys())[stop]

if args.restore:
    filename = f'logs/step{args.restore}.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    start = data['current_pc_index']
    start_pc: 'PlaceCell' = list(cognitive_map.node_network.nodes.keys())[start]
    env = PybulletEnvironment._loads_state(data['env'], visualize=args.visualize)
    start_position = PlaceInfo(*env.robot.position_and_angle, spikings=data['gc_network'], img=None, lidar=None)
else:
    start_pc: 'PlaceCell' = list(cognitive_map.node_network.nodes.keys())[start]
    env = PybulletEnvironment(args.env_model, variant=args.env_variant, start=start_pc.pos, visualize=args.visualize, build_data_set=True)
    start_position = start_pc

compass.reset_position(compass.parse(start_position))
gc_network.set_as_current_state(start_position.spikings)

if args.position_estimation:
    from system.controller.local_controller.position_estimation import PositionEstimation
    controller.transform_goal_vector.append(PositionEstimation(pc_network, gc_network, re, compass, current_position=start_pc))

with env:
    # backwards compatibility: a lot of cognitive maps were saved without angles
    # TODO: put it into the cognitive_map code?
    # cognitive_map.get_place_cell_network().add_angles_and_lidar(env)

    success = tj.navigate(start_pc, goal_pc, gc_network=gc_network, env=env, controller=controller, head=args.head)
    if success:
        print(f"Success! simulation time: {env.t}. Navigation steps: {vector_step_counter.counter}")
    else:
        print("Fail :(")
