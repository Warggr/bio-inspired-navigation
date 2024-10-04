if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.bio_model.place_cell_model import PlaceCell, PlaceCellNetwork
from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.local_controller.compass import AnalyticalCompass, Compass
from system.controller.local_controller.local_navigation import GcCompass, LinearLookaheadGcCompass, PodGcCompass
from system.controller.reachability_estimator.reachability_estimation import ReachabilityEstimator
from system.controller.reachability_estimator.data_generation.dataset import place_info
from system.controller.simulation.pybullet_environment import PybulletEnvironment
from system.types import Vector2D
from typing import Optional
from system.debug import PLOTTING
plotting = 'drift' in PLOTTING

if plotting:
    from system.plotting.plotHelper import add_environment
    import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_every_gc_error(from_pos, from_spikings):
    analytical_compass = AnalyticalCompass()

    analytical_compass.reset_position(from_pos)
    ll_compass.reset_position(from_spikings)

    analytical_compass.reset_goal(env.robot.position)
    ll_compass.reset_goal(gc_network.consolidate_gc_spiking())

    fig, ax = plt.subplots()
    add_environment(ax, env_model)
    plt.scatter(x=[pc.pos[0] for pc in pc_network.place_cells], y=[pc.pos[1] for pc in pc_network.place_cells], c=pc_firing, label='Place cells with firing strength')
    plt.plot(*(env.robot.position + experiment_offset), 'gx', label='Current position')

    plt.plot(*(analytical_compass.current_pos + experiment_offset), 'ro', label='Start position')
    pod_estimated_position = pod_compass.calculate_goal_vector()
    plt.plot(*(analytical_compass.current_pos + pod_estimated_position + experiment_offset), 'yx', label='Estimated current position')

    # decode goal vectors from current position to every place cell on the cognitive map
    quivers = np.zeros((len(pc_network.place_cells), 4))
    for i, p in enumerate(tqdm(pc_network.place_cells)):
        pod_compass.reset_goal(p.spikings)
    
        pred_gv = pod_compass.calculate_goal_vector()
        true_gv = AnalyticalCompass(start_pos=pc_network.place_cells[0].pos, goal_pos=p.pos).calculate_goal_vector()
        error_gv = pred_gv - true_gv
        quivers[i, 0:2] = p.pos
        quivers[i, 2:4] = error_gv
    plt.quiver(quivers[:, 0], quivers[:, 1], quivers[:, 2], quivers[:, 3], label='Grid cell drift as measured by start node')
    
    fig.legend()

#plot_every_gc_error(path[0], read_gc_network.consolidate_gc_spiking())


class DoubleCompass(Compass):
    def __init__(self, compass: GcCompass, true_compass: AnalyticalCompass|None=None):
        self.compass = compass
        self.true_compass = true_compass or AnalyticalCompass()
    def reset_goal(self, new_goal):
        for c, v in zip((self.compass, self.true_compass), new_goal):
            c.reset_goal(v)
    def reset_position(self, new_position):
        for c, v in zip((self.compass, self.true_compass), new_position):
            c.reset_position(v)
    def parse(self, pc):
        return tuple(c.parse(pc) for c in (self.compass, self.true_compass))
    def calculate_goal_vector(self):
        return self.compass.calculate_goal_vector()
    def calculate_estimated_position(self):
        estimated_position = np.array(self.compass.calculate_goal_vector())
        return self.true_compass.current_pos + estimated_position
    @property
    def arrival_threshold(self):
        return self.compass.arrival_threshold
    def error(self):
        estimated_position = self.compass.calculate_goal_vector()
        true_position = self.true_compass.calculate_goal_vector()

        norm_error = np.linalg.norm(true_position - estimated_position)
        angle_error = np.arccos(np.dot(true_position, estimated_position) / np.linalg.norm(true_position) / np.linalg.norm(estimated_position))
        return norm_error, angle_error
    def record_error(self):
        self.errors.append(self.error())
    def error_linearlookahead(self):
        true_position = self.true_compass.calculate_goal_vector()
        compass = LinearLookaheadGcCompass(gc_network=self.compass.gc_network, arena_size=1.5*np.linalg.norm(true_position))
        estimated_position = compass.calculate_goal_vector()

        norm_error = np.linalg.norm(true_position - estimated_position)
        angle_error = np.arccos(np.dot(true_position, estimated_position) / np.linalg.norm(true_position) / np.linalg.norm(estimated_position))
        return norm_error, angle_error

    def _block_on_error(self):
        norm, angle = self.error()
        if angle > np.radians(60):
            breakpoint()
        return norm, angle
    def update_position(self, robot: 'Robot'):
        for c in (self.compass, self.true_compass):
            c.update_position(robot)


class PositionEstimation:
    def __init__(
        self,
        pc_network: PlaceCellNetwork, gc_network: GridCellNetwork, re: ReachabilityEstimator,
        compass: DoubleCompass,
        current_position: Optional[PlaceCell] = None,
        env_model: Optional[str] = None,
    ):
        """
        PositionEstimation constructor

        arguments:
        pc_network: the place cells
        gc_network: a GridCellNetwork that tracks the agent's position
        re: used to determine at which place we are most likely, based on current observations
        compass: used for plotting and error calculation, assumed to be regularly reset
        current_position: the current position of the robot (warning: this is not automatically updated)
        env_model: the env model, used only for plotting
        """
        self.gc_network = gc_network
        self.pc_network = pc_network
        self.re = re
        self.current_position: PlaceCell = current_position # TODO: set this somewhere (i.e. don't rely on client code to set it)
        # and set it regularly when a new place cell is passed
        self.confidence_threshold = 0.8
        self.compass = compass

        #compass_ = LinearLookaheadGcCompass(arena_size=15, gc_network=GridCellNetwork(from_data=True))
        compass_ = PodGcCompass(gc_network=GridCellNetwork(from_data=True))
        positions = np.zeros((len(self.pc_network.place_cells), 2))
        for i, pc in enumerate(tqdm(self.pc_network.place_cells)):
            compass_.reset_goal_pc(pc)
            positions[i] = np.array(pc.pos) - np.array(compass_.calculate_goal_vector())
        self.starting_position = np.sum(positions, axis=0) / len(positions)
        if plotting:
            _fig, axis = plt.subplots()
            if env_model is not None:
                add_environment(axis, env_model)
            plt.scatter(positions[:, 0], positions[:, 1])
            plt.plot(*self.starting_position, 'gx')
            plt.show()

        self.counter = 0

    def print_error(self, end='\n'):
        norm_error, angle_error = self.compass._block_on_error()
        print(f"  Error norm={norm_error}, error angle={np.degrees(angle_error)}°", end=end)
        return norm_error, angle_error

    def draw_correction(self, priors, likelihoods, posteriors, robot):
        for probs, name in zip((priors, likelihoods, posteriors), ('priors', 'likelihoods', 'posteriors')):
            fig, ax = plt.subplots()
            add_environment(ax, robot.env.env_model)
            ax.scatter(x=[pc.pos[0] for pc in pc_network.place_cells], y=[pc.pos[1] for pc in pc_network.place_cells], c=probs, label=name)
            ax.plot(*(robot.position), 'rx', label='Current position')
            estimated_goal = np.array(self.current_position.pos) + np.array(self.compass.calculate_goal_vector())
            ax.plot(*estimated_goal, 'yx', label='Estimated goal position')
            true_goal = self.compass.true_compass.goal_pos
            ax.plot(*true_goal, 'gx', label='True goal position')
            fig.legend()
            ax.set_title(name)
            if 'headless' in PLOTTING:
                plt.savefig('/tmp/grid_cell_drift-' + name + '.png')
                plt.close()
            else:
                plt.show()

    def on_reset_goal(self, new_goal: Vector2D, robot: 'Robot'):
        current_observed_position = place_info((*robot.position_and_angle, self.gc_network.consolidate_gc_spiking().flatten()), robot.env)
        priors = self.pc_network.compute_firing_values(self.gc_network)
        likelihoods = self.re.reachability_factor_batch(current_observed_position, self.pc_network.place_cells)
        posteriors = np.array(priors) * np.array(likelihoods)
        if plotting:
            self.draw_correction(priors, likelihoods, posteriors, robot)

    def __call__(self, goal_vector: Vector2D, robot: 'Robot') -> Vector2D:
        if self.counter > 0:
            self.counter -= 1
            return goal_vector
        #print("Calling PositionEstimation")
        current_observed_position = place_info((*robot.position_and_angle, self.gc_network.consolidate_gc_spiking().flatten()), robot.env)
        confidence = self.re.reachability_factor(self.current_position, current_observed_position)
        self.print_error(end='\r')
        if confidence < self.confidence_threshold:
            priors = self.pc_network.compute_firing_values(self.gc_network)
            likelihoods = self.re.reachability_factor_batch(current_observed_position, self.pc_network.place_cells)
            posteriors = np.array(priors) * np.array(likelihoods)

            max_likelihood_estimate = np.argmax(posteriors)
            if max_likelihood_estimate != np.argmax(priors):
                if np.max(posteriors) >= 0.1:
                    if plotting:
                        self.draw_correction(priors, likelihoods, posteriors, robot)

                    actual_position = self.pc_network.place_cells[max_likelihood_estimate]
                    self.compass.reset_position(self.compass.parse(actual_position))
                    goal_vector = self.compass.calculate_goal_vector()
                    print('\nCorrecting:', end='\t')
                    self.print_error()
                self.counter = 20
        return goal_vector

if __name__ == "__main__":
    from system.controller.topological.topological_navigation import TopologicalNavigation
    from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory
    from system.bio_model.cognitive_map import LifelongCognitiveMap
    from system.controller.local_controller.local_navigation import setup_gc_network, PhaseOffsetDetectorNetwork
    from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig
    from system.controller.local_controller.local_controller import LocalController
    from system.parsers import fullstack_parser
    import numpy as np

    import argparse
    parser = argparse.ArgumentParser(parents=[fullstack_parser])
    parser.add_argument('map_file', default='after_lifelong_learning.gpickle')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    re_weights_file = "re_mse_weights.50"
    model = "combo"

    re = reachability_estimator_factory(args.re_type, backbone_classname='convolutional', weights_file=re_weights_file, env_model=args.env_model)
    cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from=args.map_file, debug=False)
    pc_network = cognitive_map.get_place_cell_network()
    gc_network = GridCellNetwork(from_data=True)
    #pod = PhaseOffsetDetectorNetwork(16, 9, 40)
    compass = GcCompass.factory(model, gc_network=gc_network)#, pod=pod)
    compass = DoubleCompass(compass, AnalyticalCompass())
    tj = TopologicalNavigation(args.env_model, pc_network, cognitive_map, compass)

    [pc.angle for pc in cognitive_map.node_network.nodes]

    true_compass = AnalyticalCompass()
    corrector = PositionEstimation(pc_network, gc_network, re, compass, env_model=args.env_model)
    controller = LocalController.default()
    controller.transform_goal_vector.append(corrector)
    controller.on_reset_goal.append(corrector.on_reset_goal)

    if not args.seed:
        args.seed = np.random.default_rng().integers(low=0, high=0b100000000)
        print("Using seed", args.seed)
    random = np.random.default_rng(seed=args.seed)

    start_index = random.integers(0, len(pc_network.place_cells)-1)
    start = list(pc_network.place_cells)[start_index]
    corrector.current_position = start

    compass.reset_position_pc(start)
    print('Resetting compass start to', start.pos)

    with PybulletEnvironment(args.env_model, start=start.pos, visualize=args.visualize, build_data_set=True) as env:
        env.robot.navigation_hooks.append(true_compass.update_position)
        for i in range(10):
            goal_index = None
            while goal_index == start_index or goal_index is None:
                goal_index = random.integers(0, len(pc_network.place_cells))
            goal = list(cognitive_map.node_network.nodes)[goal_index]

            true_compass.reset_goal(goal.pos)
            compass.reset_goal_pc(goal)
            success = tj.navigate(start, goal, gc_network=gc_network, controller=controller, env=env)
            if not success:
                break
            start_index = goal_index
            start = goal
