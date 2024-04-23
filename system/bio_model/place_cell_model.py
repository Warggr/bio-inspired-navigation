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

import sys
import os

from matplotlib import pyplot as plt

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.plotting.plotHelper import add_environment, TUM_colors
from system.types import Vector2D
from typing import Optional

def get_path_top():
    """ returns path to topological data folder """
    dirname = os.path.join(os.path.dirname(__file__))
    return dirname


class PlaceCell:
    """Class to keep track of an individual Place Cell"""

    def __init__(self, gc_connections, observations, coordinates : Vector2D):
        self.gc_connections = gc_connections  # Connection matrix to grid cells of all modules; has form (n^2 x M)
        self.env_coordinates = coordinates  # Save x and y coordinate at moment of creation

        self.plotted_found = [False, False]  # Was used for debug plotting, of linear lookahead

        self.observations = observations
        self.distances : Optional[np.ndarray] = None

    def compute_firing(self, s_vectors):
        """Computes firing value based on current grid cell spiking"""
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # determine where connection exist to grid cells
        filtered = np.multiply(gc_connections, s_vectors)  # filter current grid cell spiking, by connections
        modules_firing = np.sum(filtered, axis=1) / np.sum(s_vectors, axis=1)  # for each module determine pc firing
        firing = np.average(modules_firing)  # compute overall pc firing by summing averaging over modules
        return firing

    def compute_firing_2x(self, s_vectors, axis, plot=False):
        """Computes firing projected on one axis, based on current grid cell spiking"""
        new_dim = int(np.sqrt(len(s_vectors[0])))  # n

        s_vectors = np.where(s_vectors > 0.1, 1, 0)  # mute weak grid cell spiking, transform to binary vector
        gc_connections = np.where(self.gc_connections > 0.1, 1, 0)  # mute weak connections, transform to binary vector

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
        return hash(tuple(self.env_coordinates))


class PlaceCellNetwork:
    """A PlaceCellNetwork holds information about all Place Cells"""

    def __init__(self, reach_estimator: 'ReachabilityEstimator', from_data=False):
        """ Place Cell Network  of the environment. 
        
        arguments:
        from_data   -- if True: load existing place cells (default False)
        re_type     -- type of reachability estimator determining whether a new node gets created
                    see ReachabilityEstimator class for explanation of different types (default distance)
                    plus additional type "firing" that uses place cell spikings
        """
        self.reach_estimator = reach_estimator
        self.place_cells = []  # array of place cells

        if from_data:
            # Load place cells if wanted
            directory = os.path.join(get_path_top(), "data/pc_model")

            gc_connections = np.load(directory + "/gc_connections.npy")
            env_coordinates = np.load(directory + "/env_coordinates.npy")
            observations = np.load(directory + "/observations.npy", allow_pickle=True)

            for idx, gc_connection in enumerate(gc_connections):
                pc = PlaceCell(gc_connection, observations[idx], env_coordinates[idx])
                self.place_cells.append(pc)

    def create_new_pc(self, gc_connections, obs, coordinates):
        # Consolidate grid cell spiking vectors to matrix of size n^2 x M
        pc = PlaceCell(gc_connections, obs, coordinates)
        self.place_cells.append(pc)

    def in_range(self, reach: [float]) -> bool:
        """ Determine whether one value meets the threshold """
        return any(
            [self.reach_estimator.pass_threshold(reach_value, self.reach_estimator.threshold_same) for reach_value in
             reach])

    def track_movement(self, gc_network, observations, coordinates, creation_allowed):
        """Keeps track of current grid cell firing"""
        firing_values = self.compute_firing_values(gc_network.gc_modules)

        if not creation_allowed:
            return [firing_values, False]

        created_new_pc = False
        if len(firing_values) == 0 or not self.in_range(firing_values):
            self.create_new_pc(gc_network.consolidate_gc_spiking(), observations, coordinates)
            firing_values.append(1)
            created_new_pc = True

        return [firing_values, created_new_pc]

    def compute_firing_values(self, gc_modules, virtual=False, axis=None, plot=False):

        s_vectors = np.empty((len(gc_modules), len(gc_modules[0].s)))
        # Consolidate grid cell spiking vectors that we want to consider
        for m, gc in enumerate(gc_modules):
            if virtual:
                s_vectors[m] = gc.s_virtual
            else:
                s_vectors[m] = gc.s

        firing_values = []
        for i, pc in enumerate(self.place_cells):
            if axis is not None:
                plot = plot if i == 0 else False  # linear lookahead debugging plotting
                firing = pc.compute_firing_2x(s_vectors, axis, plot=plot)  # firing along axis
            else:
                firing = pc.compute_firing(s_vectors)  # overall firing
            firing_values.append(firing)
        return firing_values

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
    from system.controller.local_controller.local_navigation import setup_gc_network, vector_navigation
    from system.bio_model.cognitive_map import LifelongCognitiveMap
    from system.controller.local_controller.decoder.phase_offset_detector import PhaseOffsetDetectorNetwork
    from system.controller.simulation.pybullet_environment import PybulletEnvironment
    from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory

    # setup place cell network, cognitive map and grid cell network (from data)
    weights_file = "re_mse_weights.50"
    env_model = "Savinov_val3"

    re = reachability_estimator_factory("neural_network", weights_file=weights_file, env_model=env_model,
                                        with_spikings=True)
    pc_network = PlaceCellNetwork(from_data=True, reach_estimator=re)
    cognitive_map = LifelongCognitiveMap(reachability_estimator=re, load_data_from="after_exploration.gpickle")
    gc_network = setup_gc_network(1e-2)
    pod = PhaseOffsetDetectorNetwork(16, 9, 40)

    fr = random.choice(list(cognitive_map.node_network.nodes))
    to = random.choice(list(cognitive_map.node_network.nodes))
    env = PybulletEnvironment(env_model, visualize=False, mode="combo", build_data_set=True,
                              start=list(fr.env_coordinates))
    gc_network.set_as_current_state(fr.gc_connections)
    stop, pc = vector_navigation(env, list(to.env_coordinates), gc_network, to.gc_connections, model="combo",
                                 obstacles=True, exploration_phase=False, pc_network=pc_network,
                                 pod=pod, cognitive_map=cognitive_map, plot_it=True, step_limit=1000)

    fig, ax = plt.subplots()

    if cognitive_map:
        G = cognitive_map.node_network
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_color='#0065BD', node_size=10)
        G = G.to_undirected()
        nx.draw_networkx_nodes(G, pos, node_color='#0065BD60', node_size=40)
        nx.draw_networkx_edges(G, pos, edge_color='#99999980')
    if pc:
        circle2 = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.2, color=TUM_colors['TUMAccentGreen'],
                             alpha=1)
        ax.add_artist(circle2)
    circle1 = plt.Circle((env.xy_coordinates[-1][0], env.xy_coordinates[-1][1]), 0.2,
                         color=TUM_colors['TUMAccentOrange'], alpha=1)
    ax.add_artist(circle1)
    add_environment(ax, env)
    plt.show()
