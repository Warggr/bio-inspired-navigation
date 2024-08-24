""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
import torch
import numpy as np
# import tabulate

import os
from abc import ABC, abstractmethod
from typing import Iterable, Literal
from system.bio_model.place_cell_model import PlaceCell
from system.bio_model.bc_network.bc_encoding import BoundaryCellNetwork, HDCActivity
from system.bio_model.bc_network.parameters import HeadingCellActivity, BoundaryCellActivity
from system.bio_model.bc_network.bc_activity import bcActivityForLidar
import system.controller.reachability_estimator.networks as networks
from system.controller.simulation.environment.map_occupancy import MapLayout
import system.types as types
from system.controller.reachability_estimator.types import Prediction, ReachabilityController, PlaceInfo

try:
    from typing import override
except ImportError:
    from system.polyfill import override

Batch = networks.Batch

def reachability_estimator_factory(
    type: Literal['distance', 'neural_network', 'simulation', 'view_overlap', 'spikings', 'bvc'],
    *, debug: bool = False, **kwargs
) -> 'ReachabilityEstimator':
    """
    arguments:
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
        return SimulationReachabilityEstimator(debug=debug, env=kwargs['env'])
    elif type == 'view_overlap':
        return ViewOverlapReachabilityEstimator(debug=debug, env_model=kwargs['env_model'])
    elif type == 'spikings':
        return SpikingsReachabilityEstimator(debug=debug, axis=kwargs.get('axis', None))
    elif type == 'bvc':
        return BVCReachabilityEstimator()
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
    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo) -> float:
        """
        Abstract function, determines reachability factor between two locations
        The meaning of the reachability factor depends on the implementation, but bigger is always better.
        To make the factor a probability for all implementations, use self.get_connectivity_probability
        """
        pass

    def reachability_factor_batch(self, p: PlaceInfo, qs: Iterable[PlaceInfo]) -> list[float]:
        """ Batch version of reachability_factor. Implementations that can do this efficiently (e.g. NetworkRE) override this """
        return [self.reachability_factor(p, q) for q in qs]

    def reachability_factor_batch_2(self, ps: Iterable[PlaceInfo], q: PlaceInfo) -> list[float]:
        """ Similar to reachability_factor_batch, but the list comes first """
        return [self.reachability_factor(p, q) for p in ps]

    def reachability_factor_batch_mm(self, ps: Iterable[PlaceInfo], qs: Iterable[PlaceInfo]) -> list[float]:
        """ Similar to reachability_factor_batch, but both inputs are lists (which should have the same size) """
        return [self.reachability_factor(p, q) for p, q in zip(ps, qs)]

    @override
    def reachable(self, src: PlaceInfo, dst: PlaceInfo, path_l=None) -> bool:
        reachability_factor = self.reachability_factor(src, dst)
        return reachability_factor >= self.threshold_reachable

    def get_reachability(self, start: PlaceInfo, goal: PlaceInfo) -> tuple[bool, float]:
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

    def is_same_batch(self, p: PlaceInfo, qs: Iterable[PlaceInfo]) -> Batch[bool]:
        """ Batch version of is_same. Implementations that can do this efficiently (e.g. NetworkRE) override this. """
        return [self.is_same(p, q) for q in qs]

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
    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo, path_l=None) -> float:
        """ Returns distance between start and goal as an estimation of reachability"""
        # Since a bigger factor means better reachability, we use negative distance
        return -np.linalg.norm(np.array(start.pos) - np.array(goal.pos))

    def __str__(self):
        return f"DistanceRE({self.threshold_same},{self.threshold_reachable})"


WEIGHTS_FOLDER = os.path.join(os.path.dirname(__file__), "data/models")

NNResult = (float, float, float)

from .ReachabilityDataset import SampleConfig

def allo_bc_spikings(boundaryCellEncoder: BoundaryCellNetwork, p: PlaceInfo) -> 'HDCActivity':
    heading = HDCActivity.headingCellsActivityTraining(p.angle)
    ego = bcActivityForLidar(p.lidar)
    _, allo = boundaryCellEncoder.calculateActivities(ego, heading)
    assert not np.all(allo == 0)
    return allo


class NetworkReachabilityEstimator(ReachabilityEstimator):
    """ Creates a network-based reachability estimator that judges reachability
        between two locations based on observations and grid cell spikings """

    def __init__(
        self,
        backbone: networks.Model,
        device: str = 'cpu',
        debug: bool = True,
        batch_size: int = 64,
        config: SampleConfig = SampleConfig(),
        **_kwargs, # kwargs ignored for compatibility with reachability_estimator_factory
    ):
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
        if self.config.lidar == 'allo_bc':
            self.boundaryCellEncoder = BoundaryCellNetwork.load()

        self.backbone = backbone
        self.batch_size = batch_size

        LIDAR_LENGTHS = {
            'raw_lidar': types.LidarReading.DEFAULT_NUMBER_OF_ANGLES,
            'ego_bc': BoundaryCellActivity.size,
            'allo_bc': BoundaryCellActivity.size,
            None: 0,
        }
        lidar_length = LIDAR_LENGTHS[self.config.lidar]

        self.dtype = np.dtype([
            ('src_image', (np.int32, (64, 64, 4))),
            ('dst_image', (np.int32, (64, 64, 4))),
            ('src_spikings', (np.float32, (6, 1600))),
            ('dst_spikings', (np.float32, (6, 1600))),
            ('src_lidar', (np.float32, lidar_length)),
            ('dst_lidar', (np.float32, lidar_length)),
            ('transformation', (np.float32, 3))
        ])

    @staticmethod
    def from_file(weights_file: str, weights_folder: str = WEIGHTS_FOLDER, backbone_classname=None, **kwargs):
        """ Loads a NetworkReachabilityEstimator from a snapshot in a file.

        arguments:
        weights_file: str   -- file with weights to load the snapshot from
        weights_folder: sre -- path to the folder with weights files

        other arguments are passed to the NetworkReachabilityEstimator constructor
        """
        if weights_file == 're_mse_weights.50':
            config, new_kwargs = SampleConfig(grid_cell_spikings=True), {'with_conv': True}
        else:
            config, new_kwargs = SampleConfig.from_filename(os.path.basename(weights_file))

        re = NetworkReachabilityEstimator(backbone=None, config=config, **kwargs, **new_kwargs)

        weights_filepath = os.path.join(weights_folder, weights_file)
        state_dict = torch.load(weights_filepath, map_location='cpu')
        # self.print_debug('loaded %s' % weights_file)

        global_args = state_dict.get('global_args', {})
        if type(global_args) is not dict: # fix for some networks being saved with Hyperparameters
            global_args = {'backbone': 'convolutional'}

        # self.print_debug('global args:')
        # self.print_debug(tabulate.tabulate(global_args.items()))

        if backbone_classname is None:
            backbone_classname = global_args.pop('backbone')
        global_args = { key: value for key, value in global_args.items() if key in ['image_encoder', 'hidden_fc_layers'] }
        # TODO: other possible global_args

        backbone = networks.Model.create_from_config(backbone_classname=backbone_classname, config=re.config, **global_args)

        # self.print_debug(backbone.nets)

        nets_state = state_dict['nets']
        for name, net in backbone.nets.items():
            if name == "img_encoder" and 'conv1.weight' in state_dict['nets'][name].keys(): # Backwards compatibility
                for i in range(4):
                    for value_type in ['bias', 'weight']:
                        nets_state[name][f'layers.{2*i}.{value_type}'] = nets_state[name][f'conv{i+1}.{value_type}']
                        del nets_state[name][f'conv{i+1}.{value_type}']
            elif name == "fully_connected" and 'fc1.weight' in nets_state[name].keys():
                for i in range(net.nb_fc):
                    for value_type in ['bias', 'weight']:
                        nets_state[name][f'fc.{2*i}.{value_type}'] = nets_state[name][f'fc{i+1}.{value_type}']
                        del nets_state[name][f'fc{i+1}.{value_type}']

            net.load_state_dict(nets_state[name])
            net.train(False)

        re.backbone = backbone
        return re

    def to_batch(self, p: PlaceInfo, q: PlaceInfo):
        def get_lidar(p: PlaceInfo):
            if hasattr(p, 'lidar') and p.lidar is not None:
                if self.config.lidar == 'raw_lidar':
                    return p.lidar.distances
                elif self.config.lidar == 'ego_bc':
                    return bcActivityForLidar(p.lidar)
                elif self.config.lidar == 'allo_bc':
                    return allo_bc_spikings(self.boundaryCellEncoder, p)
            else:
                return -1 * np.ones(shape=(), dtype=self.dtype['dst_lidar'])

        p_angle, q_angle = getattr(p, 'angle', 0), getattr(q, 'angle', 0)
        args = (
            p.img, q.img,
            p.spikings, q.spikings,
            get_lidar(p), get_lidar(q),
            np.concatenate([np.array(q.pos) - np.array(p.pos), [(q_angle - p_angle) % 2*np.pi]])
        )
        return np.array([args], dtype=self.dtype)

    def is_same_batch(self, p: PlaceInfo, qs: Iterable[PlaceInfo]) -> Batch[bool]:
        return self.reachability_factor_batch(p, qs) >= self.threshold_same

    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo) -> float:
        """ Predicts reachability value between two locations """
        args = self.to_batch(start, goal)
        return self.reachability_factor_of_array(args)[0]

    def reachability_factor_batch(self, p: PlaceInfo, qs: Iterable[PlaceInfo]) -> list[float]:
        batches: list[Batch['NetworkReachabilityEstimator.dtype']] = [self.to_batch(p, q) for q in qs]
        if batches == []: return np.array([])
        args_batch = np.concatenate(batches, axis=0)
        return self.reachability_factor_of_array(args_batch)

    def reachability_factor_batch_2(self, ps: Iterable[PlaceInfo], q: PlaceInfo) -> list[float]:
        batches: list[Batch['NetworkReachabilityEstimator.dtype']] = [self.to_batch(p, q) for p in ps]
        if batches == []: return np.array([])
        args_batch = np.concatenate(batches, axis=0)
        return self.reachability_factor_of_array(args_batch)

    def reachability_factor_batch_mm(self, ps: Iterable[PlaceInfo], qs: Iterable[PlaceInfo]) -> list[float]:
        batches: list[Batch['NetworkReachabilityEstimator.dtype']] = [self.to_batch(p, q) for p, q in zip(ps, qs)]
        if batches == []: return np.array([])
        args_batch = np.concatenate(batches, axis=0)
        return self.reachability_factor_of_array(args_batch)

    def reachability_factor_of_array(self,
        data: np.ndarray['self.dtype']
    ) -> networks.Batch[float]:
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
            data: Batch['self.dtype']
        ) -> networks.Batch[Prediction]:
            batch_size = len(data)
            src_batch, goal_batch = data['src_image'], data['dst_image']
            src_spikings, goal_spikings = data['src_spikings'], data['dst_spikings']
            src_lidar, goal_lidar = data['src_lidar'], data['dst_lidar']
            transform = data['transformation']

            """ Helper function, main logic for predicting reachability for multiple location pairs """
            with torch.no_grad():
                additional_info = {}
                if self.config.images:
                    additional_info['batch_src_images'] = torch.from_numpy(src_batch).float()
                    additional_info['batch_dst_images'] = torch.from_numpy(goal_batch).float()
                if self.config.with_grid_cell_spikings:
                    additional_info['batch_src_spikings'] = torch.from_numpy(src_spikings).float()
                    additional_info['batch_dst_spikings'] = torch.from_numpy(goal_spikings).float()
                if self.config.lidar:
                    additional_info['batch_src_lidar'] = torch.from_numpy(src_lidar).float()
                    additional_info['batch_dst_lidar'] = torch.from_numpy(goal_lidar).float()
                if self.config.with_dist:
                    additional_info['batch_transformation'] = torch.from_numpy(transform).float()

                return self.backbone.get_prediction(
                    **additional_info,
                    batch_size=batch_size,
                )

        n = len(data)

        results: np.ndarray[float] = np.array([])
        n_remaining = n
        batch_size = min(self.batch_size, n)
        while n_remaining > 0:
            predictions = get_prediction(data[n-n_remaining:n-n_remaining+batch_size])[0]
            results = np.concatenate([results, predictions], axis=0)
            n_remaining -= batch_size
        return results

    def get_connectivity_probability(self, reachability_factor):
        """ Converts output of the network into connectivity factor """
        # TODO Pierre: I don't understand this, I thought the value was already a probability?
        return min(1.0, max((self.threshold_reachable - reachability_factor * 0.3) / self.threshold_reachable, 0.1))

    def __str__(self) -> str:
        return f"NetworkRE({self.config.suffix()},{self.threshold_same},{self.threshold_reachable})"


class SimulationReachabilityEstimator(ReachabilityEstimator):
    def __init__(self, env: 'PybulletEnvironment', debug=False):
        """ Creates a reachability estimator that judges reachability
            between two locations based success of navigation simulation
        """
        super().__init__(threshold_same=None, threshold_reachable=1.0, debug=debug) # we can't decide whether two points are the same, only whether they are reachable
        self.fov = 120 * np.pi / 180
        self.dt = 1e-2
        self.env = env

    @override
    def reachable(self, start: PlaceInfo, goal: PlaceInfo, path_l = None) -> bool:
        """ Determines reachability factor between two locations """
        from system.controller.simulation.pybullet_environment import Robot
        from system.controller.local_controller.local_navigation import vector_navigation, GcCompass
        from system.bio_model.grid_cell_model import GridCellNetwork

        """ Return reachability estimate from start to goal using the re_type """

        """ Simulate movement between start and goal and return whether goal was reached """

        # initialize grid cell network and create target spiking
        gc_network = GridCellNetwork(self.dt, from_data=True)

        compass = GcCompass.factory(mode="combo", gc_network=gc_network)
        compass.reset_position(compass.parse(start))
        compass.reset_goal(compass.parse(goal))
        angle = start.angle
        angle = 0.0 if angle is NotImplemented else angle
        with Robot(env=self.env, base_position=start.pos, base_orientation=angle) as robot:
            goal_reached, _ = vector_navigation(self.env, compass, gc_network, step_limit=750, plot_it=False)
            final_position, final_angle = robot.position_and_angle

        if goal_reached:
            map_layout = MapLayout(self.env.env_model)

            overlap_ratios = map_layout.view_overlap(final_position, final_angle, goal.pos, (0.0 if goal.angle is NotImplemented else goal.angle), self.fov, mode='plane')

            if overlap_ratios[0] < 0.1 and overlap_ratios[1] < 0.1:
                # Agent is close to the goal, but seperated by a wall.
                return False
            elif np.linalg.norm(goal.pos - final_position) > 0.7:
                # Agent actually didn't reach the goal and is too far away.
                return False
            else:
                # Agent did actually reach the goal
                return True
        else:
            return False

    @override
    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo) -> float:
        if self.reachable(start, goal):
            return 1.0
        else:
            return 0.0

    def __str__(self) -> str:
        return f'SimulationRE'


class ViewOverlapReachabilityEstimator(ReachabilityEstimator):
    def __init__(self, env_model: types.AllowedMapName, env_variant=None, debug=False):
        """ Creates a reachability estimator that judges reachability
            between two locations based the overlap of their fields of view

        arguments:
        threshold_same: float      -- threshold for determining when nodes are close enough to be considered same node
        threshold_reachable: float -- threshold for determining when nodes are close enough to be considered reachable
        debug: bool                -- enables logging
        """
        super().__init__(threshold_same=0.4, threshold_reachable=0.3, debug=debug)
        self.env_model = env_model
        assert env_model != 'plane', 'View overlap does not really make sense in an empty space'
        self.fov = 120 * np.pi / 180
        self.distance_threshold = 0.7
        self.map_layout = MapLayout(self.env_model)

    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo) -> float:
        """ Reachability Score based on the view overlap of start and goal in the environment """
        # untested and unfinished
        start_pos = start.pos
        goal_pos = goal.pos

        heading1 = np.arctan2(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])

        overlap_ratios = self.map_layout.view_overlap(start_pos, heading1, goal_pos, heading1, self.fov, mode='plane')

        return (overlap_ratios[0] + overlap_ratios[1]) / 2

    def __str__(self) -> str:
        return f'ViewOverlapRE({self.threshold_same},{self.threshold_reachable})'


class SpikingsReachabilityEstimator(ReachabilityEstimator):
    def __init__(self, axis=None, debug=False):
        super().__init__(threshold_same=0.9, threshold_reachable=0.4, debug=debug)
        self.axis = axis

    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo) -> float:
        if not isinstance(goal, PlaceCell):
            start, goal = goal, start
        assert isinstance(goal, PlaceCell)
        if self.axis is None:
            return goal.compute_firing(start.spikings)
        else:
            return goal.compute_firing_2x(start.spikings, axis=self.axis)

    def __str__(self) -> str:
        return f'SpikingsRE({self.threshold_same},{self.threshold_reachable})'


def spikings_reshape(img_array):
    """ Helper function, image stored in array form to image in correct shape for nn """
    img = np.reshape(img_array, (6, 40, 40))
    return img


class BVCReachabilityEstimator(ReachabilityEstimator):
    def __init__(self):
        # can't decide whether nodes are similar - we just want to detect same nodes
        super().__init__(threshold_reachable=0.4, threshold_same=0.9)
        from system.bio_model.bc_network.bc_encoding import BoundaryCellNetwork
        self.boundaryCellEncoder = BoundaryCellNetwork.load()

    def _allo_bc_spikings(self, pc: PlaceInfo):
        return allo_bc_spikings(self.boundaryCellEncoder, pc)

    def reachability_factor(self, start: PlaceInfo, goal: PlaceInfo) -> float:
        spikings_start, spikings_goal = self._allo_bc_spikings(start), self._allo_bc_spikings(goal)
        return -np.mean(np.abs(spikings_start - spikings_goal))

    def __str__(self) -> str:
        return f'BVCRE({self.threshold_same},{self.threshold_reachable})'
