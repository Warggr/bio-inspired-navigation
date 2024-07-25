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
            priors = list(enumerate(priors))
            priors.sort(key=lambda index_and_value: index_and_value[1], reverse=True) # sort by ascending probability
            max_posterior = (0, -1)
            for index, value in priors:
                if value < max_posterior[0]: # the priors are already too small, so the posteriors will also be. We might just as well stop now
                    break
                posterior = value * self.re.reachability_factor(self.pc_network.place_cells[index], current_observed_position)
                if posterior > max_posterior[0]:
                    max_posterior = (posterior, index)
            max_likelihood_estimate = max_posterior[1]
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

    re_type = "neural_network"
    re_weights_file = "re_mse_weights.50"
    map_file = "after_lifelong_learning.gpickle"
    map_file_after_lifelong_learning = "after_lifelong_learning.gpickle"
    env_model = "Savinov_val3"
    model = "combo"
    input_config = SampleConfig(grid_cell_spikings=True)
    visualize = True

    re = reachability_estimator_factory(re_type, backbone_classname='convolutional', weights_file=re_weights_file, config=input_config)
    pc_network = PlaceCellNetwork(from_data=True, reach_estimator=re)
    cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from=map_file, debug=False)
    gc_network = setup_gc_network(1e-2)
    #pod = PhaseOffsetDetectorNetwork(16, 9, 40)
    compass = GcCompass.factory(model, gc_network=gc_network)#, pod=pod)
    tj = TopologicalNavigation(env_model, pc_network, cognitive_map, compass)

    true_compass = AnalyticalCompass()
    corrector = PositionEstimation(pc_network, gc_network, re, tj.compass, true_compass)
    controller = LocalController.default()
    controller.transform_goal_vector.append(corrector)

    seed: int|None = 3
    random = np.random.default_rng(seed=seed)

    start_index = random.integers(0, len(tj.cognitive_map.node_network.nodes)-1)
    start = list(tj.cognitive_map.node_network.nodes)[start_index]
    corrector.current_position = start
    compass.reset_position(compass.parse(start))
    true_compass.reset_position(start.pos)
    with PybulletEnvironment(start=start.pos, visualize=visualize, build_data_set=True, realtime=True) as env:
        env.robot.navigation_hooks.append(true_compass.update_position)
        for i in range(10):
            goal_index = None
            while goal_index == start_index or goal_index is None:
                goal_index = random.integers(0, len(tj.cognitive_map.node_network.nodes))
            goal = list(cognitive_map.node_network.nodes)[goal_index]

            true_compass.reset_goal(goal.pos)
            success = tj.navigate(start, goal, gc_network=gc_network, controller=controller, env=env)
            if not success:
                break
            start_index = goal_index
            start = goal
