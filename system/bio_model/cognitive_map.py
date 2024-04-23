""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
from abc import ABC, abstractmethod

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from system.plotting.helper import plot_cognitive_map_path
from system.plotting.plotThesis import plot_grid_cell

from system.bio_model.place_cell_model import PlaceCell, PlaceCellNetwork
from system.controller.reachability_estimator.reachability_estimation import reachability_estimator_factory, \
    ReachabilityEstimator


def get_path_top() -> str:
    """ returns path to the folder of the current file """
    dirname = os.path.join(os.path.dirname(__file__))
    return dirname


def sample_normal(m, s):
    return np.random.normal(m, s)


class CognitiveMapInterface:
    def __init__(self, reachability_estimator: ReachabilityEstimator, load_data_from: str = None, debug: bool = True):
        """ Abstract base class defining the interface for cognitive map implementations.

        arguments:
        reachability_estimator: ReachabilityEstimator -- reachability estimator that should be used for defining
                                                         the proximity of nodes
        load_data_from: str                           -- filename of the snapshot of the cognitive map,
                                                         None if a new cognitive map is being created
        debug: bool                                   -- enables logging
        """

        self.reach_estimator = reachability_estimator
        self.node_network = nx.DiGraph()
        self.debug = debug
        if load_data_from is not None:
            self.load(filename=load_data_from)
        # threshold used for determining nodes that represents current location of the agent
        self.active_threshold = 0.9
        # last active node
        self.prior_idx_pc_firing = None

    def track_vector_movement(self, pc_firing: [float], created_new_pc: bool, pc: PlaceCell, **kwargs):
        """ Abstract function used to incorporate changes to the map after each vector navigation

        arguments:
        pc_firing: [float]   -- current firings of all place cells
        created_new_pc: bool -- indicates if a new place cell was created after vector navigation
        pc: PlaceCell        -- current location of the agent
        """
        pass

    def find_path(self, start: PlaceCell, goal: PlaceCell) -> [PlaceCell]:
        """ Returns a path in the graph from start to goal nodes"""
        try:
            path = nx.shortest_path(self.node_network, source=start, target=goal)
        except nx.NetworkXNoPath:
            return None

        return path

    def add_node_to_map(self, p: PlaceCell):
        """ Adds a new node to the cognitive map """
        self.node_network.add_node(p, pos=tuple(p.env_coordinates))

    def add_edge_to_map(self, p: PlaceCell, q: PlaceCell, w: float = 1, **kwargs):
        """ Adds a new directed weighted edge to the cognitive map with given weight and parameters

        arguments:
        p: PlaceCell -- source node of the edge
        q: PlaceCell -- target node of the edge
        w: float     -- weight of the edge
        **kwargs     -- parameters of the edge
        """
        self.node_network.add_edge(p, q, weight=w, **kwargs)

    def add_bidirectional_edge_to_map_no_weight(self, p: PlaceCell, q: PlaceCell, **kwargs):
        """ Adds a new bidirectional edge to the cognitive map with given parameters

        arguments:
        p: PlaceCell -- first node of the edge
        q: PlaceCell -- second node of the edge
        **kwargs     -- parameters of the edge
        """
        self.node_network.add_edge(p, q, **kwargs)
        self.node_network.add_edge(q, p, **kwargs)

    def add_bidirectional_edge_to_map(self, p, q, w=1, **kwargs):
        """ Adds a new bidirectional weighted edge to the cognitive map with given parameters

        arguments:
        p: PlaceCell -- first node of the edge
        q: PlaceCell -- second node of the edge
        w: float     -- weight of the edge
        **kwargs     -- parameters of the edge
        """
        self.node_network.add_edge(p, q, weight=w, **kwargs)
        self.node_network.add_edge(q, p, weight=w, **kwargs)

    def save(self, filename: str, relative_folder: str = "data/cognitive_map"):
        """ Stores the current state of the node_network to the file

        arguments:
        filename: str        -- filename of the snapshot
        relative_folder: str -- relative folder (counting from the folder of the current file) of the snapshot file
        """
        directory = os.path.join(get_path_top(), "data/cognitive_map")
        if not os.path.exists(directory):
            os.makedirs(directory)
        nx.write_gpickle(self.node_network, os.path.join(directory, filename))

    def load(self, filename: str, relative_folder: str = "data/cognitive_map"):
        """ Loads the state of the node_network from the file

        arguments:
        filename: str        -- filename of the snapshot
        relative_folder: str -- relative folder (counting from the folder of the current file) of the snapshot file
        """

        directory = os.path.join(get_path_top(), relative_folder)
        if not os.path.exists(directory):
            raise ValueError("cognitive map not found")
        self.node_network = nx.read_gpickle(os.path.join(directory, filename))
        if self.debug:
            self.draw()

    def draw(self, with_labels: bool = True):
        """ Plot the cognitive map

        arguments:
        with_labels: bool -- flag to include node indices as labels
        """
        pos = nx.get_node_attributes(self.node_network, 'pos')
        if with_labels:
            node_list = list(self.node_network.nodes)
            nx.draw(self.node_network, pos, labels={i: str(node_list.index(i)) for i in node_list})
        else:
            nx.draw(self.node_network, pos, node_color='#0065BD', node_size=120, edge_color='#4A4A4A80', width=2)
        plt.show()

    def print_debug(self, *params):
        """ Logs information if debug mode is on """
        if self.debug:
            print(*params)

    def postprocess_topological_navigation(self):
        """ Performs map processing after one full topological navigation cycle """
        pass

    def postprocess_vector_navigation(self, node_p: PlaceCell, node_q: PlaceCell, observation_q: PlaceCell,
                                      observation_p: PlaceCell, success: bool):
        """ Performs map processing after one full vector navigation

        arguments:
        node_p: PlaceCell        -- source node in the graph on the start of the vector navigation
        node_q: PlaceCell        -- estimated target node in the graph
        observation_q: PlaceCell -- actual location of the agent on the start of the vector navigation
        observation_p: PlaceCell -- actual location of the agent after vector navigation
        success: bool            -- indicates if the agent reached the target graph node
        """
        pass


class CognitiveMap(CognitiveMapInterface):
    def __init__(self, reachability_estimator=None, mode="exploration", connection=("all", "delayed"),
                 load_data_from=None, debug=False):
        """ Baseline cognitive map representation of the environment.
        
        arguments:
        reachability_estimator: ReachabilityEstimator -- reachability estimator that should be used for
                                                         defining the proximity of nodes
        mode                                          -- distinguishes between navigation and exploration mode for
                                                         differences in node connection process (default exploration)
        connection                                    -- (which nodes, when) Decides when the connection between which
                                                         nodes is calculated.
            all / radius:      all possible connections are calculated
                               or just the connections between two nodes within each others radius are calculated
            delayed / instant: the connection calculation is delayed until after the agent has explored the maze
                               or every node is connected to other nodes as soon as it is created
        load_data_from: str                           -- filename of the snapshot of the cognitive map, None if a new
                                                         cognitive map is being created
        debug: bool                                   -- enables logging

        """
        super().__init__(reachability_estimator, load_data_from=load_data_from, debug=debug)

        self.connection = connection

        self.mode = mode

        self.radius = 5  # radius in which node connection is calculated

    def update_reachabilities(self):
        """ Update reachability between the nodes. """
        nr_nodes = len(list(self.node_network.nodes))
        for i, p in enumerate(list(self.node_network.nodes)):
            self.print_debug("currently updating node " + str(i))
            progress_str = "Progress: " + str(int((i + 1) * 100 / nr_nodes)) + "%"
            print(progress_str)

            for q in list(self.node_network.nodes):

                if q == p:
                    continue

                if self.connection[0] == "radius" and np.linalg.norm(
                        q.env_coordinates - p.env_coordinates) > self.radius:
                    # No connection above radius
                    continue

                reachable, reachability_factor = self.reach_estimator.get_reachability(p, q)
                if reachable:
                    self.node_network.add_weighted_edges_from([(p, q, reachability_factor)])
                else:
                    self.node_network.remove_edges_from([(p, q, reachability_factor)])

    def _connect_single_node(self, p):
        """ Calculate reachability of node p with other nodes """
        for q in list(self.node_network.nodes):

            if q == p:
                continue

            if self.connection[0] == "radius" and np.linalg.norm(q.env_coordinates - p.env_coordinates) > self.radius:
                # No connection above radius
                continue

            reachable_pq, reachability_factor_pq = self.reach_estimator.get_reachability(p, q)
            reachable_qp, reachability_factor_qp = self.reach_estimator.get_reachability(q, p)

            if reachable_pq:
                self.node_network.add_weighted_edges_from([(p, q, reachability_factor_pq)])
            if reachable_qp:
                self.node_network.add_weighted_edges_from([(q, p, reachability_factor_qp)])

    def add_node_to_map(self, p: PlaceCell):
        super().add_node_to_map(p)

        if self.connection[1] == "instant":
            # Connect the new node to all other nodes in the graph
            self.print_debug("connecting new node")
            self._connect_single_node(p)
            self.print_debug("connecting finished")

    def track_vector_movement(self, pc_firing: [float], created_new_pc: bool, pc: PlaceCell, **kwargs):
        """Keeps track of curren/t place cell firing and creation of new place cells"""

        # get the currently active place cell
        idx_pc_active = np.argmax(pc_firing)
        pc_active_firing = np.max(pc_firing)

        # Check if we have entered a new place cell
        if created_new_pc:
            entered_different_pc = True
            self.add_node_to_map(pc)

        elif pc_active_firing > self.active_threshold and self.prior_idx_pc_firing != idx_pc_active:
            entered_different_pc = True
        else:
            entered_different_pc = False

        if entered_different_pc:
            if self.mode == "navigation" and self.prior_idx_pc_firing:
                # If we have entered place cell p after being in place cell q during
                # navigation, q is definitely reachable and the edge gets updated accordingly.
                q = list(self.node_network.nodes)[self.prior_idx_pc_firing]
                pc = list(self.node_network.nodes)[idx_pc_active]
                self.node_network.add_weighted_edges_from([(q, pc, 1)])

            self.prior_idx_pc_firing = idx_pc_active

    def save(self, relative_folder="data/cognitive_map", filename=None):
        CognitiveMapInterface.save(self, filename, relative_folder=relative_folder)
        directory = os.path.join(get_path_top(), relative_folder)

        if self.connection[1] == "delayed":
            self.update_reachabilities()
            nx.write_gpickle(self.node_network, os.path.join(directory, filename))

    def test_place_cell_network(self, env, gc_network, from_data=False):
        """ Test the drift error of place cells stored in the cognitive map """
        from system.controller.local_controller.local_navigation import find_new_goal_vector

        delta_avg = 0
        pred_gvs = []  # goal vectors decoded using linear lookahead
        true_gvs = []  # analytically calculated goal vectors
        error = []

        if from_data:
            dirname = os.path.join(os.path.dirname(__file__), "../../experiments/drift_error")

            pred_gvs = np.load(os.path.join(dirname, "pred_gvs.npy"))
            true_gvs = np.load(os.path.join(dirname, "true_gvs.npy"))
            error = true_gvs - pred_gvs
            delta = [np.linalg.norm(i) for i in error]
            delta_avg = np.mean(delta)

        else:
            env.mode = "linear_lookahead"
            # decode goal vectors from current position to every place cell on the cognitive map 
            node_list = list(self.node_network.nodes)
            nodes_length = len(node_list)
            for i, p in enumerate(node_list):
                print("Decoding goal vector to place Cell", i, "out of", nodes_length)
                target_spiking = p.gc_connections
                gc_network.set_as_target_state(target_spiking)
                env.goal_pos = p.env_coordinates

                find_new_goal_vector(gc_network, env, env.mode)

                pred_gv = env.goal_vector
                true_gv = env.calculate_goal_vector_analytically()

                error_gv = true_gv - pred_gv
                delta = np.linalg.norm(error_gv)

                delta_avg += delta
                pred_gvs.append(pred_gv)
                true_gvs.append(true_gv)
                error.append(error_gv)

            delta_avg /= nodes_length

        print("Average error:", delta_avg)

        # Plot the drift error on the cognitive map
        import system.plotting.plotHelper as pH

        plt.figure()
        ax = plt.gca()
        pH.add_environment(ax, env)
        pH.add_robot(ax, env)
        pos = nx.get_node_attributes(self.node_network, 'pos')
        nx.draw_networkx_nodes(self.node_network, pos, node_color='#0065BD80', node_size=3000)

        directory = "experiments/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = "experiments/drift_error"
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save("experiments/drift_error/pred_gvs", pred_gvs)
        np.save("experiments/drift_error/true_gvs", true_gvs)

        for i, gv in enumerate(pred_gvs):
            # control the amount of goal vectors displayed in the plot
            if i % 2 == 0 or i % 3 == 0 or i % 5 == 0:
                continue
            plt.quiver(env.xy_coordinates[-1][0], env.xy_coordinates[-1][1], gv[0], gv[1], color='grey', angles='xy',
                       scale_units='xy', scale=1, width=0.005)
            plt.quiver(env.xy_coordinates[-1][0] + gv[0], env.xy_coordinates[-1][1] + gv[1], error[i][0], error[i][1],
                       color='red', angles='xy', scale_units='xy', scale=1, width=0.005)

        plt.show()

    def postprocess_topological_navigation(self):
        self.update_reachabilities()


class LifelongCognitiveMap(CognitiveMapInterface):
    def __init__(
            self,
            reachability_estimator: ReachabilityEstimator = None,
            load_data_from: str = None,
            debug: bool = False,
            add_edges: bool = True,
            remove_edges: bool = True,
            remove_nodes: bool = True,
            add_nodes: bool = True
    ):
        """ Implements a cognitive map with lifelong learning algorithm.

        arguments:
        reachability_estimator: ReachabilityEstimator -- reachability estimator that should be used for defining the
                                                         proximity of nodes
        load_data_from: str                           -- filename of the snapshot of the cognitive map, None if a new
                                                         cognitive map is being created
        debug: bool                                   -- enables logging
        add_edges: bool                               -- defines if edge addition is enabled
        remove_edges: bool                            -- defines if edge cleanup is enabled
        remove_nodes: bool                            -- defines if node cleanup is enabled
        add_nodes: bool                               -- defines if node addition is enabled
        """

        super().__init__(reachability_estimator, load_data_from=load_data_from, debug=debug)
        # values used for probabilistic calculations
        self.sigma = 0.015
        self.sigma_squared = self.sigma ** 2
        self.threshold_edge_removal = 0.5
        self.p_s_given_r = 0.55
        self.p_s_given_not_r = 0.15

        self.add_edges = add_edges
        self.remove_edges = remove_edges
        self.add_nodes = add_nodes
        self.remove_nodes = remove_nodes

        self.min_node_degree_for_deletion = 4
        self.max_number_unique_neighbors_for_deletion = 2

    def track_vector_movement(self, pc_firing: [float], created_new_pc: bool, pc: PlaceCell, **kwargs) -> PlaceCell:
        """ Incorporate changes to the map after each vector navigation tryout. Adds nodes during exploration phase and
            edges during navigation.

        arguments:
        pc_firing: [float]                -- current firings of all place cells
        created_new_pc: bool              -- indicates if a new place cell was created after vector navigation
        pc: PlaceCell                     -- current location of the agent
        kwargs:
             exploration_phase: bool      -- indicates exploration or navigation phase
             pc_network: PlaceCellNetwork -- place cell network

        returns:
        pc: PlaceCell                 -- current active node if it exists
        """
        exploration_phase = kwargs.get('exploration_phase', True)
        pc_network = kwargs.get('pc_network', None)
        if exploration_phase and created_new_pc:
            is_mergeable, mergeable_values = self.is_mergeable(pc)
            if is_mergeable:
                return pc
            self.add_and_connect_node(pc)
        elif not exploration_phase and not created_new_pc:
            if self.add_edges:
                self.process_add_edge(pc_firing, pc_network)
        if np.max(pc_firing) > self.active_threshold:
            self.prior_idx_pc_firing = np.argmax(pc_firing)
        if self.prior_idx_pc_firing is not None:
            return pc_network.place_cells[self.prior_idx_pc_firing]
        return None

    def process_add_edge(self, pc_firing: [float], pc_network: PlaceCellNetwork):
        """ Helper function. Decides if a new edge should be added between the last active node and the
            current active node

        arguments:
        pc_firing: [float]           -- current firings of all place cells
        pc_network: PlaceCellNetwork -- place cell network
        """
        idx_pc_active = np.argmax(pc_firing)
        pc_active_firing = np.max(pc_firing)

        if pc_active_firing > self.active_threshold and self.prior_idx_pc_firing != idx_pc_active:
            if self.prior_idx_pc_firing:
                # If we have entered place cell p after being in place cell q during
                # navigation, q is definitely reachable and the edge gets updated accordingly.
                q = pc_network.place_cells[self.prior_idx_pc_firing]
                pc_new = pc_network.place_cells[idx_pc_active]
                if (q in self.node_network and pc_new in self.node_network and
                        q not in self.node_network[pc_new] and q != pc_new):
                    self.print_debug(f"adding edge [{self.prior_idx_pc_firing}-{idx_pc_active}]")
                    self.add_bidirectional_edge_to_map(q, pc_new,
                                                       sample_normal(0.5, self.sigma),
                                                       connectivity_probability=0.8,
                                                       mu=0.5,
                                                       sigma=self.sigma)

    def is_connectable(self, p: PlaceCell, q: PlaceCell) -> (bool, float):
        """ Helper function. Checks if two waypoints p and q are connectable."""
        return self.reach_estimator.get_reachability(p, q)

    def is_mergeable(self, p: PlaceCell) -> (bool, [bool]):
        """ Helper function. Checks if the waypoint p is mergeable with the existing graph"""
        mergeable_values = [self.reach_estimator.is_same(p, q) for q in self.node_network.nodes]
        return any(self.reach_estimator.is_same(p, q) for q in self.node_network.nodes), mergeable_values

    def postprocess_vector_navigation(self, node_p: PlaceCell, node_q: PlaceCell, observation_p: PlaceCell,
                                      observation_q: PlaceCell, success: bool):
        """ Performs map processing after one full vector navigation. Updates edge connectivity probabilities.
            May add new nodes and edges, and remove edges.

        arguments:
        node_p: PlaceCell        -- source node in the graph on the start of the vector navigation
        node_q: PlaceCell        -- estimated target node in the graph
        observation_q: PlaceCell -- actual location of the agent on the start of the vector navigation
        observation_p: PlaceCell -- actual location of the agent after vector navigation
        success: bool            -- indicates if the agent reached the target graph node
        """

        if node_q == node_p:
            return
        if not success and observation_q not in self.node_network and self.add_nodes:
            self.add_and_connect_node(observation_q)
        if self.add_edges:
            if observation_p != observation_p and observation_p in self.node_network and observation_p not in \
                    self.node_network[observation_q]:
                self.add_bidirectional_edge_to_map(observation_p, observation_q,
                                                   sample_normal(0.5, self.sigma),
                                                   connectivity_probability=0.8,
                                                   mu=0.5,
                                                   sigma=self.sigma)
        if node_p not in self.node_network or node_q not in self.node_network[node_p]:
            return

        self.update_edge_parameters(node_p, node_q, observation_p, success)
        if not success and self.remove_edges:
            if self.node_network[node_q][node_p]['connectivity_probability'] < self.threshold_edge_removal:
                self.remove_bidirectional_edge(node_p, node_q)

        self.print_debug(
            f"edge [{list(self.node_network.nodes).index(node_p)}-{list(self.node_network.nodes).index(node_q)}]: " +
            f"success {success} conn {self.node_network[node_q][node_p]['connectivity_probability']}")

    def update_edge_parameters(self, node_p: PlaceCell, node_q: PlaceCell, observation_p: PlaceCell, success: bool):
        """ Helper function. Performs map processing after one full vector navigation.
            Updates edge connectivity probabilities. May add new nodes and edges.

        arguments:
        node_p: PlaceCell        -- source node in the graph on the start of the vector navigation
        node_q: PlaceCell        -- estimated target node in the graph
        observation_q: PlaceCell -- actual location of the agent on the start of the vector navigation
        observation_p: PlaceCell -- actual location of the agent after vector navigation
        success: bool            -- indicates if the agent reached the target graph node
        """
        edges = [self.node_network[node_p][node_q], self.node_network[node_q][node_p]]

        t = self.conditional_probability(success, True) * edges[0]['connectivity_probability']
        connectivity_probability = t / (
                t + self.conditional_probability(success, False) * (1 - edges[0]['connectivity_probability']))
        connectivity_probability = min(connectivity_probability, 0.95)
        for edge in edges:
            edge['connectivity_probability'] = connectivity_probability

        if success:
            weight = self.reach_estimator.get_reachability(observation_p, node_q)[1]
            sigma_ij_t_squared = edges[0]['sigma'] ** 2
            mu_ij_t = edges[0]['mu']
            mu = (self.sigma_squared * mu_ij_t + sigma_ij_t_squared * weight) / (
                    sigma_ij_t_squared + self.sigma_squared)
            sigma = np.sqrt(1 / (1 / sigma_ij_t_squared + 1 / self.sigma_squared))
            weight = sample_normal(mu, sigma)

            for edge in edges:
                edge['mu'] = mu
                edge['sigma'] = sigma
                edge['weight'] = weight

    def conditional_probability(self, s: bool = True, r: bool = True):
        """ Helper function, computes conditional probability values for edge connectivity computations """
        if s:
            if r:
                return self.p_s_given_r
            return self.p_s_given_not_r
        if r:
            return 1 - self.p_s_given_r
        return 1 - self.p_s_given_not_r

    def remove_bidirectional_edge(self, node_p: PlaceCell, node_q: PlaceCell):
        """ Helper function, removes bidirectional edge between two nodes """
        self.node_network.remove_edge(node_p, node_q)
        self.node_network.remove_edge(node_q, node_p)
        nodelist = list(self.node_network.nodes)
        self.print_debug(
            f"deleting edge [{nodelist.index(node_p)}-{nodelist.index(node_q)}]: " +
            f"conn {self.node_network[node_q][node_p]['connectivity_probability']}")

    def deduplicate_nodes(self):
        """ Helper function, performs node cleanup. If nodes have too many common neighbors,
            they are considered duplicates.
        """
        nodes = list(self.node_network.nodes)
        deleted = []

        def skip_pair(node_p: PlaceCell, node_q: PlaceCell):
            return (node_p in deleted or node_q in deleted or node_q == node_p or
                    node_q not in self.node_network or node_p not in self.node_network or
                    node_p not in self.node_network[node_q])

        for node_p in nodes:
            for node_q in nodes:
                if skip_pair(node_p, node_q):
                    continue
                if self.are_duplicates(node_p, node_q):
                    self.print_debug(f"Nodes {nodes.index(node_p)} and {nodes.index(node_q)} are duplicates, " +
                                     f"deleting {nodes.index(node_p)}")
                    for neighbor in self.node_network[node_p]:
                        if neighbor not in self.node_network[node_q] and neighbor != node_q:
                            edge_attributes_dict = self.node_network.edges[node_p, neighbor]
                            self.add_bidirectional_edge_to_map_no_weight(node_q, neighbor, **edge_attributes_dict)
                    deleted.append(node_p)
        for node in deleted:
            self.node_network.remove_node(node)

    def are_duplicates(self, node_p: PlaceCell, node_q: PlaceCell):
        """ Helper function, checks if two nodes are duplicates of each other """
        set_p = set(self.node_network[node_p])
        set_q = set(self.node_network[node_q])
        common = len(set_p.intersection(set_q))

        return common >= len(set_p) - self.max_number_unique_neighbors_for_deletion and common >= len(
            set_q) - self.max_number_unique_neighbors_for_deletion and len(
            set_p) >= self.min_node_degree_for_deletion and len(set_q) >= self.min_node_degree_for_deletion

    def postprocess_topological_navigation(self):
        """ Performs map processing after one full topological navigation cycle. Calls node deduplication if enabled """

        self.prior_idx_pc_firing = None
        if not self.remove_nodes:
            self.deduplicate_nodes()

    def add_and_connect_node(self, pc: PlaceCell):
        """ Helper function. Adds new node to the map and edges to adjacent nodes with standard parameters  """
        self.add_node_to_map(pc)
        for node in self.node_network.nodes:
            if node != pc:
                reachable, weight = self.reach_estimator.get_reachability(node, pc)
                if reachable:
                    connectivity_probability = self.reach_estimator.get_connectivity_probability(weight)
                    self.add_bidirectional_edge_to_map(pc, node,
                                                       sample_normal(1 - weight, self.sigma),
                                                       connectivity_probability=connectivity_probability,
                                                       mu=1 - weight,
                                                       sigma=self.sigma)


if __name__ == "__main__":
    """ Load and visualize cognitive map + observations with grid cell spikings on both ends of distinct edges  """
    from system.controller.simulation.pybullet_environment import PybulletEnvironment

    # Adjust what sort of RE you want to use for connecting nodes
    connection_re_type = "neural_network"  # "neural_network" #"simulation" #"view_overlap"
    weights_filename = "re_mse_weights.50"
    map_filename = "after_exploration.gpickle"
    env_model = "Savinov_val3"
    debug = True

    re = reachability_estimator_factory(connection_re_type, weights_file=weights_filename, env_model=env_model,
                                        debug=debug, with_spikings=True)
    # Select the version of the cognitive map to use
    cm = LifelongCognitiveMap(reachability_estimator=re, load_data_from=map_filename)
    cm.draw()

    dt = 1e-2
    env = PybulletEnvironment(env_model, dt, visualize=False, mode="analytical", build_data_set=True)
    import random

    for i in range(10):
        # Select an edge to visualize or use a random one
        start, finish = random.sample(list(cm.node_network.edges()), 1)[0]

        plot_cognitive_map_path(cm.node_network, [start, finish], env)

        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        ax.imshow(start.observations[-1].transpose(1, 2, 0))
        ax = fig.add_subplot(1, 2, 2)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        ax.imshow(finish.observations[-1].transpose(1, 2, 0))

        plt.show()
        plt.close()

        plot_grid_cell(start.gc_connections, finish.gc_connections)
