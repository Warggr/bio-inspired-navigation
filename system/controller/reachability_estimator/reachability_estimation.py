""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
import numpy
import torch
import numpy as np
import tabulate

import sys
import os
from abc import ABC, abstractmethod
from typing import Type, Dict
import system.controller.reachability_estimator.networks as networks
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.bio_model.place_cell_model import PlaceCell

def reachability_estimator_factory(type: str = 'distance', /, device: str = 'cpu', debug: bool = False, env_model : str = None, **kwargs) -> 'ReachabilityEstimator':
    """ Returns an instance of the reachability estimator interface

    arguments:
    type: str -- type of the reachability estimator, possible values:
                 ['distance' (default), 'neural_network', 'simulation', 'view_overlap']
    kwargs:
        device: str         -- type of the computations, possible values: ['cpu' (default), 'gpu']
        weights_file: str   -- filename of the weights for network-based estimator if exists
        with_grid_cell_spikings: bool -- parameter for network-based estimator, flag to include grid cell spikings into input
        env_model: str      -- model of the environment for simulation-based estimator

    returns:
        ReachabilityEstimator object of the corresponding type
    """
    if type == 'distance':
        return DistanceReachabilityEstimator(device=device, debug=debug)
    elif type == 'neural_network':
        return NetworkReachabilityEstimator.from_file(device=device, debug=debug, **kwargs)
    elif type == 'simulation':
        return SimulationReachabilityEstimator(device=device, debug=debug, env_model=env_model)
    elif type == 'view_overlap':
        return ViewOverlapReachabilityEstimator(device=device, debug=debug)
    print("Reachability estimator type not defined: " + type)
    return None


class ReachabilityEstimator(ABC):
    def __init__(self, threshold_same: float, threshold_reachable: float, device: str = 'cpu', debug: bool = False):
        """ Abstract base class defining the interface for reachability estimator implementations.

        arguments:
        threshold_same: float      -- threshold for determining when nodes are close enough to be considered same node
        threshold_reachable: float -- threshold for determining when nodes are close enough to be considered reachable
        device                     -- device used for calculations (default cpu)
        debug: bool                -- enables logging
        """
        self.device = device
        self.debug = debug
        self.threshold_same = threshold_same
        self.threshold_reachable = threshold_reachable

    def print_debug(self, *params):
        """ Helper function, outputs only when in debug mode """
        if self.debug:
            print(*params)

    @abstractmethod
    def predict_reachability(self, start: PlaceCell, goal: PlaceCell) -> float:
        """ Abstract function, determines reachability factor between two locations """
        pass

    def get_reachability(self, p: PlaceCell, q: PlaceCell) -> (bool, float):
        """ Determines whether two nodes are reachable based on the reachability threshold

        returns:
        bool  -- flag that indicates that locations are reachable
        float -- reachability probability
        """
        reachability_factor = self.predict_reachability(p, q)
        return self.pass_threshold(reachability_factor, self.threshold_reachable), reachability_factor

    @abstractmethod
    def pass_threshold(self, reachability_factor, threshold) -> bool:
        """ Abstract function, decides if the reachability value passes the threshold """
        pass

    def is_same(self, p: PlaceCell, q: PlaceCell) -> bool:
        """ Determine whether two nodes are close to each other sufficiently to consider them the same node """
        return self.pass_threshold(self.predict_reachability(p, q), self.threshold_same)

    def get_connectivity_probability(self, reachability_factor):
        """ Computes connectivity probability based on reachability factor """
        return reachability_factor


class DistanceReachabilityEstimator(ReachabilityEstimator):
    def __init__(self, device='cpu', debug=False):
        """ Creates a reachability estimator that judges reachability between two locations based on the distance
            
        arguments:
        device -- device used for calculations (default cpu)
        debug  -- is in debug mode
        """
        super().__init__(threshold_same=0.4, threshold_reachable=0.75, device=device, debug=debug)

    def predict_reachability(self, start: PlaceCell, goal: PlaceCell) -> float:
        """ Returns distance between start and goal as an estimation of reachability"""
        return np.linalg.norm(start.env_coordinates - goal.env_coordinates)

    def pass_threshold(self, reachability_factor: float, threshold: float) -> bool:
        """ Two nodes are reachable if the distance is less than the threshold """
        return reachability_factor < threshold

WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), "data/models")

class NetworkReachabilityEstimator(ReachabilityEstimator):
    """ Creates a network-based reachability estimator that judges reachability
        between two locations based on observations and grid cell spikings """

    def __init__(self, backbone: networks.Model, device: str = 'cpu', debug: bool = True, batch_size: int = 64, with_grid_cell_spikings : bool = False, with_dist : bool = False):
        """
        arguments:
        backbone: str       -- neural network used as a backbone
        device: str         -- device used for calculations (default cpu)
        debug: bool         -- is in debug mode
        with_grid_cell_spikings: bool -- flag indicates whether to include grid cell firing to input
        batch_size: int     -- size of batches (default 64)
        """
        super().__init__(threshold_same=0.933, threshold_reachable=0.4, device=device, debug=debug)

        self.with_grid_cell_spikings = with_grid_cell_spikings
        self.with_dist = with_dist

        self.backbone = backbone
        self.batch_size = batch_size

    @staticmethod
    def from_file(weights_file: str, weights_folder: str = WEIGHTS_FOLDER, *args, **kwargs):
        """ Loads a NetworkReachabilityEstimator from a snapshot in a file.

        arguments:
        weights_file: str   -- file with weights to load the snapshot from
        weights_folder: sre -- path to the folder with weights files

        other arguments are passed to the NetworkReachabilityEstimator constructor
        """
        backbone_kwargs = { key: value for key, value in kwargs.items() if key.startswith('with_') }
        # TODO more fine-grained control of which args are passed to the backbone

        weights_filepath = os.path.join(weights_folder, weights_file)
        state_dict = torch.load(weights_filepath, map_location='cpu')
        # self.print_debug('loaded %s' % weights_file)

        global_args = state_dict.get('global_args', {})

        # self.print_debug('global args:')
        # self.print_debug(tabulate.tabulate(global_args.items()))

        backbone_classname = global_args.get('backbone')
        batch_size = global_args.get('batch_size')
        model_variant = global_args.get('model_variant')

        backbone = networks.Model.create_from_config(backbone_classname, model_variant, **backbone_kwargs)

        # self.print_debug(backbone.nets)

        for name, net in backbone.nets.items():
            if name == "img_encoder" and 'conv1.weight' in state_dict['nets'][name].keys(): # Backwards compatibility
                for i in range(4):
                    for value_type in ['bias', 'weight']:
                        state_dict['nets'][name][f'layers.{2*i}.{value_type}'] = state_dict['nets'][name][f'conv{i+1}.{value_type}']
                        del state_dict['nets'][name][f'conv{i+1}.{value_type}']
            net.load_state_dict(state_dict['nets'][name])
            net.train(False)

        return NetworkReachabilityEstimator(backbone, *args, **kwargs)

    def predict_reachability(self, start: PlaceCell, goal: PlaceCell) -> float:
        """ Predicts reachability value between two locations """
        args = [ start.observations[0], goal.observations[-1] ]
        if self.with_grid_cell_spikings:
            if isinstance(goal.gc_connections, list):
                goal.gc_connections = np.array(goal.gc_connections)
            args += [
                spikings_reshape(start.gc_connections.flatten()),
                spikings_reshape(goal.gc_connections.flatten())
            ]
        if self.with_dist:
            args += [ start.distances ]
        args = [ [arg] for arg in args ] # predict_reachability_batch expects batches / lists for each param
        return self.predict_reachability_batch(*args)[0]

    def predict_reachability_batch(self, starts: [numpy.ndarray | torch.Tensor], goals: [numpy.ndarray | torch.Tensor],
        src_spikings: [numpy.ndarray | torch.Tensor] = None, goal_spikings: [numpy.ndarray | torch.Tensor] = None,
        src_distances: numpy.ndarray = None, # goal_distances: numpy.ndarray = None,                        
    ) -> [float]:
        """ Predicts reachability for multiple location pairs

        arguments:
        starts: [numpy.ndarray | torch.Tensor]        -- images perceived by the agent on first locations of each pair
        starts: [numpy.ndarray | torch.Tensor]        -- images perceived by the agent on second locations of each pair
        src_spikings: [numpy.ndarray]                 -- grid cell firings corresponding to the first locations
                                                         of each pair, nullable
        goal_spikings: [numpy.ndarray]                -- grid cell firings corresponding to the second locations
                                                         of each pair, nullable
        batch_size: int                               -- length of each input list and of returned list

        returns:
        [float] -- reachability values
        """

        def get_prediction(
            src_batch: [numpy.ndarray | torch.Tensor], dst_batch: [numpy.ndarray | torch.Tensor],
            src_spikings: [numpy.ndarray] = None, goal_spikings: [numpy.ndarray] = None,
            src_distances: [numpy.ndarray] = None, # goal_distances: [numpy.ndarray] = None,
        ) -> (float, float, float):
            """ Helper function, main logic for predicting reachability for multiple location pairs """
            with torch.no_grad():
                if isinstance(src_batch[0], np.ndarray):
                    src_batch = np.array(src_batch)
                    if self.with_grid_cell_spikings:
                        src_spikings = np.array(src_spikings)
                elif isinstance(src_batch[0], torch.Tensor):
                    if not isinstance(src_batch, torch.Tensor):
                        src_batch = torch.stack(src_batch)
                else:
                    raise RuntimeError('Unsupported datatype: %s' % type(src_batch[0]))
                if isinstance(dst_batch[0][0], np.ndarray):
                    dst_batch = np.array(dst_batch)
                    if self.with_grid_cell_spikings:
                        goal_spikings = np.array(goal_spikings)
                elif isinstance(dst_batch[0][0], torch.Tensor):
                    if not isinstance(dst_batch, torch.Tensor):
                        dst_batch = torch.stack(dst_batch)
                else:
                    raise RuntimeError('Unsupported datatype: %s' % type(dst_batch[0]))

                additional_info = {}
                if self.with_grid_cell_spikings:
                    additional_info['batch_src_spikings'] = torch.from_numpy(src_spikings).float()
                    additional_info['batch_dst_spikings'] = torch.from_numpy(goal_spikings).float()
                if self.with_dist:
                    additional_info['batch_src_distances'] = torch.from_numpy(src_distances).float()
                    # additional_info['batch_dst_distances'] = torch.from_numpy(goal_distances).float()

                return self.backbone.get_prediction(
                    torch.from_numpy(src_batch).float(),
                    torch.from_numpy(dst_batch).float(),
                    **additional_info
                )

        assert len(starts) == len(goals)
        n = len(starts)

        results : [(float, float, float)] = []
        n_remaining = n
        batch_size = min(self.batch_size, len(starts))
        while n_remaining > 0:
            batch_indices = slice(n - n_remaining, n - n_remaining + batch_size)
            results.append(
                get_prediction(
                    starts[batch_indices], goals[batch_indices],
                    src_spikings[batch_indices], goal_spikings[batch_indices],
                    src_distances[batch_indices], goal_distances[batch_indices],
                )[0]
            )
            n_remaining -= batch_size
        return torch.cat(results, dim=0).data.cpu().numpy()

    def pass_threshold(self, reachability_factor, threshold) -> bool:
        """ Two nodes are reachable if the confidence value of the network is greater than the threshold """
        return reachability_factor > threshold

    def get_connectivity_probability(self, reachability_factor):
        """ Converts output of the network into connectivity factor """
        return min(1.0, max((self.threshold_reachable - reachability_factor * 0.3) / self.threshold_reachable, 0.1))


class SimulationReachabilityEstimator(ReachabilityEstimator):
    def __init__(self, device='cpu', debug=False, env_model=None):
        """ Creates a reachability estimator that judges reachability
            between two locations based success of navigation simulation

        arguments:
        threshold_same: float      -- threshold for determining when nodes are close enough to be considered same node
        threshold_reachable: float -- threshold for determining when nodes are close enough to be considered reachable
        device                     -- device used for calculations (default cpu)
        debug: bool                -- enables logging
        """
        super().__init__(threshold_same=1.0, threshold_reachable=1.0, device=device, debug=debug)
        self.env_model = env_model
        self.fov = 120 * np.pi / 180

    def predict_reachability(self, start: PlaceCell, goal: PlaceCell) -> float:
        """ Determines reachability factor between two locations """
        from system.controller.local_controller.local_navigation import setup_gc_network, vector_navigation

        """ Return reachability estimate from start to goal using the re_type """
        if not self.env_model:
            raise ValueError("missing env_model; needed for simulating reachability")
        """ Simulate movement between start and goal and return whether goal was reached """
        dt = 1e-2

        # initialize grid cell network and create target spiking
        gc_network = setup_gc_network(dt)
        gc_network.set_as_current_state(start.gc_connections)
        target_spiking = goal.gc_connections
        start_pos = start.env_coordinates
        goal_pos = goal.env_coordinates

        model = "combo"

        from system.controller.simulation.pybullet_environment import PybulletEnvironment
        env = PybulletEnvironment(False, dt, self.env_model, "analytical", start=list(start_pos))

        over, _ = vector_navigation(env, list(goal_pos), gc_network, target_gc_spiking=target_spiking, model=model,
                                    step_limit=750, plot_it=False)

        if over == 1:
            map_layout = MapLayout(self.env_model)

            overlap_ratios = map_layout.view_overlap(env.xy_coordinates[-1], env.orientation_angle[-1],
                                                     goal_pos, env.orientation_angle[-1], self.fov, mode='plane')

            env.end_simulation()
            if overlap_ratios[0] < 0.1 and overlap_ratios[1] < 0.1:
                # Agent is close to the goal, but seperated by a wall.
                return 0.0
            elif np.linalg.norm(goal_pos - env.xy_coordinates[-1]) > 0.7:
                # Agent actually didn't reach the goal and is too far away.
                return 0.0
            else:
                # Agent did actually reach the goal
                return 1.0
        else:
            env.end_simulation()
            return 0.0

    def pass_threshold(self, reachability_factor, threshold) -> bool:
        return reachability_factor >= threshold


class ViewOverlapReachabilityEstimator(ReachabilityEstimator):
    def __init__(self, device='cpu', debug=False):
        """ Creates a reachability estimator that judges reachability
            between two locations based the overlap of their fields of view

        arguments:
        threshold_same: float      -- threshold for determining when nodes are close enough to be considered same node
        threshold_reachable: float -- threshold for determining when nodes are close enough to be considered reachable
        device                     -- device used for calculations (default cpu)
        debug: bool                -- enables logging
        """
        super().__init__(threshold_same=0.4, threshold_reachable=0.3, device=device, debug=debug)
        self.env_model = "Savinov_val3"
        self.fov = 120 * np.pi / 180
        self.distance_threshold = 0.7
        self.map_layout = MapLayout(self.env_model)

    def predict_reachability(self, start: PlaceCell, goal: PlaceCell) -> float:
        """ Reachability Score based on the view overlap of start and goal in the environment """
        # untested and unfinished
        start_pos = start.env_coordinates
        goal_pos = goal.env_coordinates

        heading1 = np.degrees(np.arctan2(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]))

        overlap_ratios = self.map_layout.view_overlap(start_pos, heading1, goal_pos, heading1, self.fov, mode='plane')

        return (overlap_ratios[0] + overlap_ratios[1]) / 2

    def pass_threshold(self, reachability_factor, threshold) -> bool:
        return reachability_factor > threshold


def spikings_reshape(img_array):
    """ Helper function, image stored in array form to image in correct shape for nn """
    img = np.reshape(img_array, (6, 40, 40))
    return img
