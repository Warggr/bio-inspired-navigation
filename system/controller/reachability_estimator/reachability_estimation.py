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
from typing import Type, Dict, Self
import system.controller.reachability_estimator.networks as networks
from system.controller.simulation.environment.map_occupancy import MapLayout
from system.bio_model.place_cell_model import PlaceCell
from system.controller.reachability_estimator.types import ReachabilityController, PlaceInfo, types

try:
    from typing import override
except ImportError:
    from system.polyfill import override

def reachability_estimator_factory(type: str = 'distance', /, debug: bool = False, **kwargs) -> 'ReachabilityEstimator':
    """ Returns an instance of the reachability estimator interface

    arguments:
    type: str -- type of the reachability estimator, possible values:
                 ['distance' (default), 'neural_network', 'simulation', 'view_overlap']
    kwargs:
        device: str         -- type of the computations, possible values: ['cpu' (default), 'gpu']
        weights_file: str   -- filename of the weights for network-based estimator if exists
        with_grid_cell_spikings: bool -- parameter for network-based estimator, flag to include grid cell spikings into input

    returns:
        ReachabilityEstimator object of the corresponding type
    """
    if type == 'distance':
        return DistanceReachabilityEstimator(debug=debug)
    elif type == 'neural_network':
        return NetworkReachabilityEstimator.from_file(debug=debug, **kwargs)
    elif type == 'simulation':
        return SimulationReachabilityEstimator(debug=debug)
    elif type == 'view_overlap':
        return ViewOverlapReachabilityEstimator(debug=debug)
    else:
        raise ValueError("Reachability estimator type not defined: " + type)


class ReachabilityEstimator(ReachabilityController):
    def __init__(self, threshold_same: float, threshold_reachable: float, debug: bool = False):
        """ Abstract base class defining the interface for reachability estimator implementations.

        arguments:
        threshold_same: float      -- threshold for determining when nodes are close enough to be considered same node
        threshold_reachable: float -- threshold for determining when nodes are close enough to be considered reachable
        debug: bool                -- enables logging
        """
        self.debug = debug
        self.threshold_same = threshold_same
        self.threshold_reachable = threshold_reachable

    def print_debug(self, *params):
        """ Helper function, outputs only when in debug mode """
        if self.debug:
            print(*params)

    @abstractmethod
    def reachability_factor(self, start : PlaceInfo, goal : PlaceInfo) -> float:
        """
        Abstract function, determines reachability factor between two locations
        The meaning of the reachability factor depends on the implementation, but bigger is always better.
        To make the factor a probability for all implementations, use self.get_connectivity_probability
        """
        pass

    @override
    def reachable(self, env, start : PlaceInfo, goal : PlaceInfo, path_l = None) -> bool:
        reachability_factor = self.reachability_factor(start, goal)
        return reachability_factor >= self.threshold_reachable

    def get_reachability(self, start : PlaceInfo, goal : PlaceInfo) -> (bool, float):
        """ Determines whether two nodes are reachable based on the reachability threshold

        returns:
        bool  -- flag that indicates that locations are reachable
        float -- reachability probability
        """
        reachability_factor = self.reachability_factor(start, goal)
        return (reachability_factor >= self.threshold_reachable), self.get_connectivity_probability(reachability_factor)

    def is_same(self, p: PlaceInfo, q: PlaceInfo) -> bool:
        """ Determine whether two nodes are close to each other sufficiently to consider them the same node """
        return self.reachability_factor(p, q) >= self.threshold_same

    def get_connectivity_probability(self, reachability_factor):
        """
        Computes connectivity probability based on reachability factor.
        By default the factor is the probability itself.
        """
        return reachability_factor


class DistanceReachabilityEstimator(ReachabilityEstimator):
    THRESHOLD_SAME = 0.4

    def __init__(self, debug=False):
        """ Creates a reachability estimator that judges reachability between two locations based on the distance
            
        arguments:
        debug  -- is in debug mode
        """
        super().__init__(threshold_same=-self.THRESHOLD_SAME, threshold_reachable=-0.75, debug=debug)

    @override
    def reachability_factor(self, env_model, src : PlaceInfo, dst : PlaceInfo, path_l=None) -> bool:
        """ Returns distance between start and goal as an estimation of reachability"""
        return -np.linalg.norm(src.pos - dst.pos) # Since a bigger factor means better reachability, we use negative distance


WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), "data/models")

NNResult = (float, float, float)

from .ReachabilityDataset import SampleConfig

class NetworkReachabilityEstimator(ReachabilityEstimator):
    """ Creates a network-based reachability estimator that judges reachability
        between two locations based on observations and grid cell spikings """

    def __init__(self, backbone: networks.Model, device: str = 'cpu', debug: bool = True, batch_size: int = 64, config : SampleConfig = SampleConfig()):
        """
        arguments:
        backbone: str       -- neural network used as a backbone
        device: str         -- device used for calculations (default cpu)
        debug: bool         -- is in debug mode
        batch_size: int     -- size of batches (default 64)
        config              -- the SampleConfig to use for inputs
        """
        super().__init__(threshold_same=0.933, threshold_reachable=0.4, debug=debug)

        self.config = config

        self.backbone = backbone
        self.batch_size = batch_size

    @staticmethod
    def from_file(weights_file: str, weights_folder: str = WEIGHTS_FOLDER, **kwargs):
        """ Loads a NetworkReachabilityEstimator from a snapshot in a file.

        arguments:
        weights_file: str   -- file with weights to load the snapshot from
        weights_folder: sre -- path to the folder with weights files

        other arguments are passed to the NetworkReachabilityEstimator constructor
        """
        re = NetworkReachabilityEstimator(backbone=None, **kwargs)

        weights_filepath = os.path.join(weights_folder, weights_file)
        state_dict = torch.load(weights_filepath, map_location='cpu')
        # self.print_debug('loaded %s' % weights_file)

        global_args = state_dict.get('global_args', {})

        # self.print_debug('global args:')
        # self.print_debug(tabulate.tabulate(global_args.items()))

        backbone_classname = global_args.pop('backbone')
        global_args = { key: value for key, value in global_args.items() if key in ['with_conv_layer'] }
        # TODO: other possible global_args

        backbone = networks.Model.create_from_config(backbone_classname=backbone_classname, config=re.config, **global_args)

        # self.print_debug(backbone.nets)

        for name, net in backbone.nets.items():
            if name == "img_encoder" and 'conv1.weight' in state_dict['nets'][name].keys(): # Backwards compatibility
                for i in range(4):
                    for value_type in ['bias', 'weight']:
                        state_dict['nets'][name][f'layers.{2*i}.{value_type}'] = state_dict['nets'][name][f'conv{i+1}.{value_type}']
                        del state_dict['nets'][name][f'conv{i+1}.{value_type}']
            net.load_state_dict(state_dict['nets'][name])
            net.train(False)

        re.backbone = backbone
        return re

    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo) -> float:
        """ Predicts reachability value between two locations """
        # types.Image format, returned by env.camera, has size (64x64x4) (channels last)
        # while the NN expects (4x64x64) (channels first)

        args = [
            start.img, goal.img,
            start.spikings, goal.spikings,
            start.lidar, goal.lidar,
        ]
        args = [ [arg] for arg in args ] # reachability_factor_batch expects batches / lists for each param
        return self.reachability_factor_batch(*args)[0]

    def reachability_factor_batch(self, starts: [numpy.ndarray | torch.Tensor], goals: [numpy.ndarray | torch.Tensor],
        src_spikings: [numpy.ndarray | torch.Tensor] = None, goal_spikings: [numpy.ndarray | torch.Tensor] = None,
        src_distances: numpy.ndarray = None, goal_distances: numpy.ndarray = None,
    ) -> [float]:
        """ Predicts reachability for multiple location pairs

        arguments:
        starts: [numpy.ndarray | torch.Tensor]        -- images perceived by the agent on first locations of each pair
        starts: [numpy.ndarray | torch.Tensor]        -- images perceived by the agent on second locations of each pair
        src_spikings: [numpy.ndarray]                 -- grid cell firings corresponding to the first locations
                                                         of each pair, nullable
        goal_spikings: [numpy.ndarray]                -- grid cell firings corresponding to the second locations
                                                         of each pair, nullable

        returns:
        [float] -- reachability values
        """

        def get_prediction(
            src_batch: [numpy.ndarray | torch.Tensor], dst_batch: [numpy.ndarray | torch.Tensor],
            src_spikings: [numpy.ndarray] = None, goal_spikings: [numpy.ndarray] = None,
            src_distances: [numpy.ndarray] = None, # goal_distances: [numpy.ndarray] = None,
        ) -> networks.Batch[networks.Model.Prediction]:
            """ Helper function, main logic for predicting reachability for multiple location pairs """
            with torch.no_grad():
                if isinstance(src_batch[0], np.ndarray):
                    src_batch = np.array(src_batch)
                    if self.config.with_grid_cell_spikings:
                        src_spikings = np.array(src_spikings)
                elif isinstance(src_batch[0], torch.Tensor):
                    if not isinstance(src_batch, torch.Tensor):
                        src_batch = torch.stack(src_batch)
                else:
                    raise RuntimeError('Unsupported datatype: %s' % type(src_batch[0]))
                if isinstance(dst_batch[0][0], np.ndarray):
                    dst_batch = np.array(dst_batch)
                    if self.config.with_grid_cell_spikings:
                        goal_spikings = np.array(goal_spikings)
                elif isinstance(dst_batch[0][0], torch.Tensor):
                    if not isinstance(dst_batch, torch.Tensor):
                        dst_batch = torch.stack(dst_batch)
                else:
                    raise RuntimeError('Unsupported datatype: %s' % type(dst_batch[0]))

                additional_info = {}
                if self.config.with_grid_cell_spikings:
                    additional_info['batch_src_spikings'] = torch.from_numpy(src_spikings).float()
                    additional_info['batch_dst_spikings'] = torch.from_numpy(goal_spikings).float()
                if self.config.with_dist:
                    additional_info['batch_src_distances'] = torch.from_numpy(src_distances).float()
                    # additional_info['batch_dst_distances'] = torch.from_numpy(goal_distances).float()

                return self.backbone.get_prediction(
                    torch.from_numpy(src_batch).float(),
                    torch.from_numpy(dst_batch).float(),
                    **additional_info
                )

        assert len(starts) == len(goals)
        n = len(starts)

        results : [networks.Model.Prediction] = []
        n_remaining = n
        batch_size = min(self.batch_size, len(starts))
        while n_remaining > 0:
            batch_indices = slice(n - n_remaining, n - n_remaining + batch_size)
            batch_args = [
                batch[batch_indices] if batch is not None else None
                for batch in [starts, goals, src_spikings, goal_spikings, src_distances]
            ] # TODO Pierre: this is ugly
            results.append(
                get_prediction(*batch_args)[0]
            )
            n_remaining -= batch_size
        return torch.cat(results, dim=0).data.cpu().numpy()

    def get_connectivity_probability(self, reachability_factor):
        """ Converts output of the network into connectivity factor """
        # TODO Pierre: I don't understand this, I thought the value was already a probability?
        return min(1.0, max((self.threshold_reachable - reachability_factor * 0.3) / self.threshold_reachable, 0.1))


class SimulationReachabilityEstimator(ReachabilityEstimator):
    def __init__(self, debug=False):
        """ Creates a reachability estimator that judges reachability
            between two locations based success of navigation simulation
        """
        super().__init__(threshold_same=None, threshold_reachable=1.0, debug=debug) # we can't decide whether two points are the same, only whether they are reachable
        self.fov = 120 * np.pi / 180
        self.dt = 1e-2

    @override
    def reachable(self, env : 'PybulletEnvironment', start: PlaceInfo, goal: PlaceInfo, path_l = None) -> bool:
        """ Determines reachability factor between two locations """
        from system.controller.simulation.pybullet_environment import Robot
        from system.controller.local_controller.local_navigation import setup_gc_network, vector_navigation, GcCompass

        """ Return reachability estimate from start to goal using the re_type """

        """ Simulate movement between start and goal and return whether goal was reached """

        # initialize grid cell network and create target spiking
        gc_network = setup_gc_network(self.dt)
        gc_network.set_as_current_state(start.spikings)
        target_spiking = goal.spikings

        compass = GcCompass.factory(mode="combo", gc_network=gc_network, goal_pos=goal.env_coordinates)
        with Robot(env=env, base_position=start.env_coordinates, base_orientation=start.angle) as robot:
            goal_reached, _ = vector_navigation(env, compass, gc_network, target_gc_spiking=target_spiking,
                                        step_limit=750, plot_it=False)
            final_position, final_angle = robot.position_and_angle

        if goal_reached:
            map_layout = MapLayout(env.env_model)

            overlap_ratios = map_layout.view_overlap(final_position, final_angle, goal.env_coordinates, goal.angle, self.fov, mode='plane')

            if overlap_ratios[0] < 0.1 and overlap_ratios[1] < 0.1:
                # Agent is close to the goal, but seperated by a wall.
                return False
            elif np.linalg.norm(goal.env_coordinates - final_position) > 0.7:
                # Agent actually didn't reach the goal and is too far away.
                return False
            else:
                # Agent did actually reach the goal
                return True
        else:
            return False

    @override
    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo) -> float:
        if self.reachable(start, goal): # TODO provide the env somewhere
            return 1.0
        else:
            return 0.0


class ViewOverlapReachabilityEstimator(ReachabilityEstimator):
    def __init__(self, debug=False):
        """ Creates a reachability estimator that judges reachability
            between two locations based the overlap of their fields of view

        arguments:
        threshold_same: float      -- threshold for determining when nodes are close enough to be considered same node
        threshold_reachable: float -- threshold for determining when nodes are close enough to be considered reachable
        debug: bool                -- enables logging
        """
        super().__init__(threshold_same=0.4, threshold_reachable=0.3, debug=debug)
        self.env_model = "Savinov_val3"
        self.fov = 120 * np.pi / 180
        self.distance_threshold = 0.7
        self.map_layout = MapLayout(self.env_model)

    def reachability_factor(self, start: PlaceCell, goal: PlaceCell) -> float:
        """ Reachability Score based on the view overlap of start and goal in the environment """
        # untested and unfinished
        start_pos = start.env_coordinates
        goal_pos = goal.env_coordinates

        heading1 = np.degrees(np.arctan2(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1]))

        overlap_ratios = self.map_layout.view_overlap(start_pos, heading1, goal_pos, heading1, self.fov, mode='plane')

        return (overlap_ratios[0] + overlap_ratios[1]) / 2


def spikings_reshape(img_array):
    """ Helper function, image stored in array form to image in correct shape for nn """
    img = np.reshape(img_array, (6, 40, 40))
    return img
