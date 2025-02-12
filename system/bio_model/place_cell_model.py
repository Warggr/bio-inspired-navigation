""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""

import random

import networkx as nx
import numpy as np

import os

from matplotlib import pyplot as plt

from system.controller.reachability_estimator._types import PlaceInfo, untranspose_image
from system.plotting.plotHelper import add_environment, TUM_colors
from system.types import Vector2D
from system.debug import DEBUG
from typing import Optional

def get_path_top():
    """ returns path to topological data folder """
    dirname = os.path.join(os.path.dirname(__file__))
    return dirname


class PlaceCell(PlaceInfo):
    """
    Class to keep track of an individual Place Cell
    TODO Pierre: there should be no distinction between this and the pure data class PlaceInfo,
    both methods of this class should be moved to the reachability estimator
    """

    def __init__(self, gc_connections, observations, coordinates: Vector2D, angle: float = NotImplemented, lidar: Optional['LidarReading'] = None):
        # explicitly not call super().__init__ because we'll provide data members ourselves (as aliases)
        self.gc_connections = gc_connections  # Connection matrix to grid cells of all modules; has form (n^2 x M)
        self.env_coordinates = coordinates  # Save x and y coordinate at moment of creation

        self.plotted_found = [False, False]  # Was used for debug plotting, of linear lookahead

        self.observations = observations
        assert observations[0] is not None
        self.lidar = lidar
        self.angle = angle
        if 'strict_place_cells' in DEBUG:
            assert self.angle is not NotImplemented

    @staticmethod
    def from_data(data: PlaceInfo):
        return PlaceCell(gc_connections=data.spikings, observations=[data.img], coordinates=data.pos, lidar=data.lidar, angle=data.angle)

    def compute_firing(self, s_vectors):
        """Computes firing value based on current grid cell spiking"""
        gc_connections = np.where(self.spikings > 0.1, 1, 0)  # determine where connection exist to grid cells
        filtered = np.multiply(gc_connections, s_vectors)  # filter current grid cell spiking, by connections
        modules_firing = np.sum(filtered, axis=1) / np.sum(s_vectors, axis=1)  # for each module determine pc firing
        firing = np.average(modules_firing)  # compute overall pc firing by summing averaging over modules
        return firing

    def compute_firing_2x(self, s_vectors, axis, plot=False):
        """Computes firing projected on one axis, based on current grid cell spiking"""
        new_dim = int(np.sqrt(len(s_vectors[0])))  # n

        s_vectors = np.where(s_vectors > 0.1, 1, 0)  # mute weak grid cell spiking, transform to binary vector
        gc_connections = np.where(self.spikings > 0.1, 1, 0)  # mute weak connections, transform to binary vector

        proj_s_vectors = np.empty((len(s_vectors[:, 0]), new_dim))
        for i, s in enumerate(s_vectors):
            s = np.reshape(s, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
            proj_s_vectors[i] = np.sum(s, axis=axis)  # sum over column/row

        proj_gc_connections = np.empty_like(proj_s_vectors)
        for i, gc_vector in enumerate(gc_connections):
            gc_vector = np.reshape(gc_vector, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
            proj_gc_connections[i] = np.sum(gc_vector, axis=axis)  # sum over column/row

        filtered = np.multiply(proj_gc_connections, proj_s_vectors)  # filter projected firing, by projected connections

        norm = np.sum(np.multiply(proj_s_vectors, proj_s_vectors), axis=1)  # compute unnormed firing at optimal case

        firing = 0
        modules_firing = 0
        for idx, filtered_vector in enumerate(filtered):
            # We have to distinguish between modules tuned for x direction and modules tuned for y direction
            if np.amin(filtered_vector) == 0:
                # If tuned for right direction there will be clearly distinguishable spikes
                firing = firing + np.sum(filtered_vector) / norm[idx]  # normalize firing and add to firing
                modules_firing = modules_firing + 1

        firing = firing / modules_firing  # divide by modules that we considered to get overall firing

        # # Plotting options, used for linear lookahead debugging
        # if plot:
        #     for idx, s_vector in enumerate(s_vectors):
        #         plot_vectors(s_vectors[idx], gc_connections[idx], axis=axis, i=idx)
        #     plot_linear_lookahead_function(proj_gc_connections, proj_s_vectors, filtered, axis=axis)

        # if firing > 0.97 and not self.plotted_found[axis]:
        #     for idx, s_vector in enumerate(s_vectors):
        #         plot_vectors(s_vectors[idx], gc_connections[idx], axis=axis, i=idx, found=True)
        #     plot_linear_lookahead_function(proj_gc_connections, proj_s_vectors, filtered, axis=axis, found=True)
        #     self.plotted_found[axis] = True

        return firing

    def __eq__(self, obj):
        return isinstance(obj, PlaceCell) and np.isclose(obj.env_coordinates, self.env_coordinates, rtol=1e-08,
                                                         atol=1e-10, equal_nan=False).all()

    def __hash__(self):
        return hash(self.env_coordinates[0])

    # Introducing aliases for some properties so this can be used as a PlaceInfo
    @property
    def pos(self): return self.env_coordinates
    @property
    def img(self):
        if self.observations[-1].shape == (4, 64, 64): # backwards compatibiliy: Anna saved images in the transposed format
            return untranspose_image(np.array([self.observations[-1]]))[0]
        return self.observations[-1]
    @property
    def spikings(self): return self.gc_connections

class PlaceCellNetwork:
    """A PlaceCellNetwork holds information about all Place Cells"""


    def __init__(self, reach_estimator: Optional['ReachabilityEstimator'] = None, from_data=False, map_name=None):
        """ Place Cell Network  of the environment.

        arguments:
        from_data   -- if True: load existing place cells (default False)
        re_type     -- type of reachability estimator determining whether a new node gets created
                    see ReachabilityEstimator class for explanation of different types (default distance)
                    plus additional type "firing" that uses place cell spikings
        map_name    -- the map name; only used when from_data is True so the PC network for the correct map is loaded
        """
        if reach_estimator is None:
            from system.controller.reachability_estimator.reachability_estimation import SpikingsReachabilityEstimator
            self.reach_estimator = SpikingsReachabilityEstimator()
        else:
            self.reach_estimator = reach_estimator

        self.place_cells: list[PlaceCell] = []

        if from_data:
            # Load place cells if wanted
            directory = os.path.join(get_path_top(), "data", "pc_model")

            gc_connections = np.load(directory + f"/gc_connections-{map_name}.npy")
            env_coordinates = np.load(directory + f"/env_coordinates-{map_name}.npy")
            observations = np.load(directory + f"/observations-{map_name}.npy", allow_pickle=True)

            for idx, gc_connection in enumerate(gc_connections):
                pc = PlaceCell(gc_connection, observations[idx], env_coordinates[idx])
                self.place_cells.append(pc)

    def create_new_pc(self, data: PlaceInfo):
        # Consolidate grid cell spiking vectors to matrix of size n^2 x M
        pc = PlaceCell.from_data(data)
        self.place_cells.append(pc)

    def in_range(self, reach: list[float]) -> bool:
        """ Determine whether one value meets the threshold """
        return any(reach_value > self.reach_estimator.threshold_same for reach_value in reach)

    def track_movement(self, current_position: PlaceInfo, creation_allowed):
        """Keeps track of current grid cell firing"""
        firing_values = list(self.reach_estimator.reachability_factor_batch_2( self.place_cells, current_position))

        if not creation_allowed:
            return [firing_values, False]

        created_new_pc = False
        if len(firing_values) == 0 or not self.in_range(firing_values):
            self.create_new_pc(current_position)
            firing_values.append(1)
            created_new_pc = True

        return [firing_values, created_new_pc]

    def compute_firing_values(self, gc_network: 'GridCellNetwork|Spikings', axis=None, plot=False):

        if type(gc_network) == np.ndarray:
            s_vectors = gc_network
        else:
            s_vectors = gc_network.consolidate_gc_spiking()

        firing_values = []
        for i, pc in enumerate(self.place_cells):
            if axis is not None:
                plot = plot if i == 0 else False  # linear lookahead debugging plotting
                firing = pc.compute_firing_2x(s_vectors, axis, plot=plot)  # firing along axis
            else:
                firing = pc.compute_firing(s_vectors)  # overall firing
            firing_values.append(firing)
        return firing_values

    def add_angles_and_lidar(self, env: 'PybulletEnvironment'):
        """
        For PC networks saved without angles or lidar data, add a reasonable guess of these
        """
        prev_pos = None
        for pc in self.place_cells:
            try:
                pc.angle
            except AttributeError:
                if prev_pos is None:
                    angle = 0
                else:
                    dist = pc - prev_pos
                    angle = np.angle(dist[0] + 1.0j*dist[1])
                pc.angle = angle
                # overwriting previous image and (possibly) lidar, because we don't know from what angle they were
                pc.lidar = env.lidar((pc.pos, angle))[0]
                pc.observations = [env.camera((pc.pos, angle))]

    def save_pc_network(self, filename=""):
        """ Save current place cell network """
        gc_connections = []
        env_coordinates = []
        observations = []
        for pc in self.place_cells:
            gc_connections.append(pc.gc_connections)
            env_coordinates.append(pc.env_coordinates)
            observations.append(pc.observations)

        directory = os.path.join(get_path_top(), "data/pc_model")
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, "gc_connections" + filename + ".npy"), gc_connections)
        np.save(os.path.join(directory, "env_coordinates" + filename + ".npy"), env_coordinates)
        np.save(os.path.join(directory, "observations" + filename + ".npy"), observations)


if __name__ == '__main__':
    from system.controller.local_controller.local_navigation import GridCellNetwork, vector_navigation, ComboGcCompass
    from system.controller.local_controller.local_controller import LocalController
    from system.bio_model.cognitive_map import LifelongCognitiveMap
    from system.controller.local_controller.decoder.phase_offset_detector import PhaseOffsetDetectorNetwork
    from system.controller.simulation.pybullet_environment import PybulletEnvironment
    from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory

    # setup place cell network, cognitive map and grid cell network (from data)
    weights_file = "re_mse_weights.50"
    env_model = "Savinov_val3"

    re = reachability_estimator_factory("neural_network", weights_file=weights_file, env_model=env_model,
                                        with_spikings=True)
    pc_network = PlaceCellNetwork(from_data=True, reach_estimator=re, map_name=env_model)
    cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from="after_exploration.gpickle")
    gc_network = GridCellNetwork()
    pod = PhaseOffsetDetectorNetwork(16, 9, 40)

    fr = random.choice(list(cognitive_map.node_network.nodes))
    to = random.choice(list(cognitive_map.node_network.nodes))
    env = PybulletEnvironment(env_model, visualize=False, build_data_set=True,
                              start=list(fr.env_coordinates))
    compass = ComboGcCompass(gc_network, pod)
    compass.reset_goal_pc(to)
    gc_network.set_as_current_state(fr.gc_connections)
    controller = LocalController.default()
    stop, pc = vector_navigation(env, compass, gc_network=gc_network, controller=controller,
                                 exploration_phase=False, pc_network=pc_network,
                                 cognitive_map=cognitive_map, plot_it=True, step_limit=1000)

    fig, ax = plt.subplots()

    if cognitive_map:
        G = cognitive_map.node_network
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_color='#0065BD', node_size=10)
        G = G.to_undirected()
        nx.draw_networkx_nodes(G, pos, node_color='#0065BD60', node_size=40)
        nx.draw_networkx_edges(G, pos, edge_color='#99999980')
    if pc:
        circle2 = plt.Circle(pc.pos, 0.2, color=TUM_colors['TUMAccentGreen'],
                             alpha=1)
        ax.add_artist(circle2)
    circle1 = plt.Circle(env.robot.data_collector.xy_coordinates[-1], 0.2,
                         color=TUM_colors['TUMAccentOrange'], alpha=1)
    ax.add_artist(circle1)
    add_environment(ax, env.env_model)
    plt.show()
