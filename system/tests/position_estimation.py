if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.bio_model.place_cell_model import PlaceCell, PlaceCellNetwork
from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.local_controller.compass import AnalyticalCompass, Compass
from system.controller.local_controller.local_navigation import GcCompass
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


class PositionEstimation:
    def __init__(
        self,
        pc_network: PlaceCellNetwork, gc_network: GridCellNetwork, re: ReachabilityEstimator, compass: Compass,
        true_compass: Optional[Compass] = None,
    ):
        self.gc_network = gc_network
        self.pc_network = pc_network
        self.re = re
        self.current_position: PlaceCell = None # TODO: set this somewhere (i.e. don't rely on client code to set it)
        # and set it regularly when a new place cell is passed
        self.confidence_threshold = 0.8
        self.compass = compass
        self.true_compass = true_compass

        positions = np.zeros((len(self.pc_network.place_cells), 2))
        for i, pc in enumerate(self.pc_network.place_cells):
            self.compass.reset_goal(self.compass.parse(pc))
            positions[i] = np.array(pc.pos) - np.array(self.compass.calculate_goal_vector())
        self.starting_position = np.sum(positions, axis=0) / len(positions)
        if plotting:
            _fig, axis = plt.subplots()
            add_environment(axis, env_model)
            plt.scatter(positions[:, 0], positions[:, 1])
            plt.plot(*self.starting_position, 'gx')
            plt.show()

        self.counter = 0
    def print_error(self):
        if self.true_compass is None:
            return
        estimated_position = self.compass.calculate_goal_vector()
        true_position = self.true_compass.calculate_goal_vector()

        norm_error = np.linalg.norm(true_position - estimated_position)
        angle_error = np.arccos(np.dot(true_position, estimated_position) / np.linalg.norm(true_position) / np.linalg.norm(estimated_position))
        print(f"  Error norm={norm_error}, error angle={np.degrees(angle_error)}°")
    def __call__(self, goal_vector: Vector2D, robot: 'Robot') -> Vector2D:
        if self.counter > 0:
            self.counter -= 1
            return goal_vector
        #print("Calling PositionEstimation")
        current_observed_position = place_info((*robot.position_and_angle, self.gc_network.consolidate_gc_spiking().flatten()), robot.env)
        confidence = self.re.reachability_factor(self.current_position, current_observed_position)
        if confidence < self.confidence_threshold:
            print("Correcting place cell")
            self.print_error()
            priors = self.pc_network.compute_firing_values(self.gc_network)
            likelihoods = self.re.reachability_factor_batch(current_observed_position, self.pc_network.place_cells)
            posteriors = np.array(priors) * np.array(likelihoods)
            if plotting:
                for probs, name in zip((priors, likelihoods, posteriors), ('priors', 'likelihoods', 'posteriors')):
                    _fig, ax = plt.subplots()
                    add_environment(ax, env_model)
                    plt.scatter(x=[pc.pos[0] for pc in pc_network.place_cells], y=[pc.pos[1] for pc in pc_network.place_cells], c=probs, label=name)
                    plt.plot(*(robot.position), 'gx', label='Current position')
                    estimated_position = self.starting_position + np.array(self.compass.calculate_goal_vector())
                    plt.plot(*estimated_position, 'yx', label='Estimated position')
                    plt.show()

            max_likelihood_estimate = np.argmax(posteriors)
            if max_likelihood_estimate != np.argmax(priors):
                actual_position = self.pc_network.place_cells[max_likelihood_estimate]
                self.compass.reset_position(self.compass.parse(actual_position))
                goal_vector = self.compass.calculate_goal_vector()
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
    import numpy as np

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('map_file', default='after_lifelong_learning.gpickle')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    re_type = 'view_overlap' #"neural_network"
    re_weights_file = "re_mse_weights.50"
    map_file_after_lifelong_learning = "after_lifelong_learning.gpickle"
    env_model = "Savinov_val3"
    model = "combo"
    input_config = SampleConfig(grid_cell_spikings=True)

    re = reachability_estimator_factory(re_type, backbone_classname='convolutional', weights_file=re_weights_file, config=input_config, env_model=env_model)
    cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from=args.map_file, debug=False)
    pc_network = cognitive_map.get_place_cell_network()
    gc_network = GridCellNetwork(from_data=True)
    #pod = PhaseOffsetDetectorNetwork(16, 9, 40)
    compass = GcCompass.factory(model, gc_network=gc_network)#, pod=pod)
    tj = TopologicalNavigation(env_model, pc_network, cognitive_map, compass)

    true_compass = AnalyticalCompass()
    corrector = PositionEstimation(pc_network, gc_network, re, tj.compass, true_compass)
    controller = LocalController.default()
    controller.transform_goal_vector.append(corrector)

    if not args.seed:
        args.seed = np.random.default_rng().integers(low=0, high=0b100000000)
        print("Using seed", args.seed)
    random = np.random.default_rng(seed=args.seed)

    start_index = random.integers(0, len(pc_network.place_cells)-1)
    start = list(pc_network.place_cells)[start_index]
    corrector.current_position = start

    compass.reset_position(compass.parse(start))
    true_compass.reset_position(start.pos)

    with PybulletEnvironment(start=start.pos, visualize=args.visualize, build_data_set=True) as env:
        env.robot.navigation_hooks.append(true_compass.update_position)
        for i in range(10):
            goal_index = None
            while goal_index == start_index or goal_index is None:
                goal_index = random.integers(0, len(pc_network.place_cells))
            goal = list(cognitive_map.node_network.nodes)[goal_index]

            true_compass.reset_goal(goal.pos)
            success = tj.navigate(start, goal, gc_network=gc_network, controller=controller, env=env)
            if not success:
                break
            start_index = goal_index
            start = goal
