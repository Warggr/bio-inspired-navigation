from system.controller.local_controller.compass import Compass, AnalyticalCompass
from system.controller.local_controller.local_navigation import LinearLookaheadGcCompass, PodGcCompass
from system.bio_model.grid_cell_model import GridCellNetwork
import numpy as np

class DoubleCompass(Compass):
    def __init__(self):
        self.gc_network = GridCellNetwork(from_data=True) # this points from the starting position to the current position.
        self.compasses = {
            'analytical': AnalyticalCompass(),
            'll': LinearLookaheadGcCompass(arena_size=12, gc_network=self.gc_network),
            'pod': PodGcCompass(gc_network=self.gc_network),
        }
    def calculate_goal_vector(self):
        raise NotImplementedError()
    def parse(self, pc: 'PlaceInfo'):
        return pc.pos, pc.spikings
    def update_position(self, robot):
        pass
    arrival_threshold = None
    def reset_goal(self, tup):
        pos, spikings = tup
        self.compasses['ll'].reset_goal(spikings) # should also affect the PodGcCompass
        self.compasses['analytical'].reset_goal(pos)
    def reset_position(self, tup):
        pos, spikings = tup
        self.compasses['analytical'].reset_position(pos)
        self.compasses['ll'].reset_position(spikings) # also affects the PodGcCompass, since they're both using the same GC network
    def error(self):
        errors = []
        true_position = self.compasses['analytical'].calculate_goal_vector()
        for compass in ('ll', 'pod'):
            estimated_position = self.compasses[compass].calculate_goal_vector()
            error = estimated_position - true_position
            angle_error = np.arccos(np.dot(true_position, estimated_position) / np.linalg.norm(true_position) / np.linalg.norm(estimated_position))
            errors.append((error, angle_error))
        return errors

from system.bio_model.cognitive_map import CognitiveMap
from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory

env_model = 'Savinov_val3'

cogmap = CognitiveMap(reachability_estimator=None, mode='navigation', load_data_from='after_exploration.gpickle', debug=False)
#cogmap.draw()

from system.controller.topological.topological_navigation import TopologicalNavigation
from system.controller.local_controller.compass import AnalyticalCompass

pc_network = cogmap.get_place_cell_network()
pc_network.max_capacity = len(pc_network.place_cells)
start, end = 73, 82
start, end = pc_network.place_cells[start], pc_network.place_cells[end]

#compass = AnalyticalCompass()
gc_network = GridCellNetwork(from_data=True)
compass = Compass.factory('combo', gc_network=gc_network)
tj = TopologicalNavigation(env_model, pc_network, cogmap, compass)

from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.local_controller.local_controller import LocalController

import matplotlib.pyplot as plt
from system.plotting.plotHelper import add_environment

place_cell_positions = np.array([pc.pos for pc in pc_network.place_cells])

def plot_every_gc_error(double_compass: DoubleCompass):
    fig, ax = plt.subplots()
    add_environment(ax, env_model)

    pc_firing = pc_network.compute_firing_values(gc_network=double_compass.gc_network)

    analytical_compass = double_compass.compasses['analytical']

    ax.scatter(x=[pc.pos[0] for pc in pc_network.place_cells], y=[pc.pos[1] for pc in pc_network.place_cells], c=pc_firing, label='Place cells with firing strength')
    ax.plot(*(analytical_compass.current_pos), 'bx', label='Current position')

    ax.plot(*(analytical_compass.goal_pos), 'gx', label='Next goal node')
    pod_estimated_position = double_compass.compasses['pod'].calculate_goal_vector()
    ax.plot(*(analytical_compass.current_pos + pod_estimated_position), 'yx', label='Estimated goal node')

    real_place_cell = pc_network.place_cells[np.argmin(np.linalg.norm(place_cell_positions - analytical_compass.current_pos, axis=1))]
    ax.plot(*(real_place_cell.pos), 'cx', label='Current place cell')

    pod_compass = PodGcCompass(gc_network=GridCellNetwork(from_data=True))
    pod_compass.reset_position(real_place_cell.spikings)
    pod_compass.reset_goal(double_compass.compasses['pod'].gc_network.target_spiking)
    ax.plot(*(np.array(real_place_cell.pos) + pod_compass.calculate_goal_vector()), 'rx', label='Estimated goal node with corrected spikings')

    pod_compass.reset_position(gc_network.consolidate_gc_spiking())
    # decode goal vectors from current position to every place cell on the cognitive map
    quivers = np.zeros((len(pc_network.place_cells), 4))
    for i, p in enumerate(pc_network.place_cells):
        pod_compass.reset_goal(p.spikings)

        pred_gv = pod_compass.calculate_goal_vector()
        true_gv = AnalyticalCompass(start_pos=pc_network.place_cells[0].pos, goal_pos=p.pos).calculate_goal_vector()
        error_gv = pred_gv - true_gv
        quivers[i, 0:2] = p.pos
        quivers[i, 2:4] = error_gv
    ax.quiver(quivers[:, 0], quivers[:, 1], quivers[:, 2], quivers[:, 3], label='Grid cell drift as measured by start node', width=0.005)

    fig.legend()
    #plt.show()
    global invocation_counter
    plt.savefig(f'./logs/gc_drift_{invocation_counter}.png')
    invocation_counter += 1

with PybulletEnvironment(env_model, visualize=False, start=start.pos, build_data_set=True) as env:
    robot = env.robot
    pc_network.add_angles_and_lidar(env)

    compass.reset_position(compass.parse(start))

    double_compass = DoubleCompass()
    double_compass.reset_position(double_compass.parse(start))

    gc_network = GridCellNetwork(from_data=True)
    gc_network.set_as_current_state(start.spikings)

    errors = []
    def on_reset_goal(new_goal, robot):
        double_compass.reset_position((robot.position, gc_network.consolidate_gc_spiking()))
        double_compass.reset_goal((robot.position + new_goal, gc_network.target_spiking))
        errors.append(double_compass.error())
        plot_every_gc_error(double_compass)
        print(errors[-1])

    controller = LocalController.default()
    controller.on_reset_goal.append(on_reset_goal)

    tj.navigate(start, end, gc_network, controller, env=env, robot=robot, add_nodes=False)
