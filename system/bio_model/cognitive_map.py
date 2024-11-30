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

import os
from abc import ABC, abstractmethod

from system.plotting.helper import plot_cognitive_map_path
from system.plotting.plotThesis import plot_grid_cell

from system.bio_model.place_cell_model import PlaceCell, PlaceCellNetwork
ReachabilityEstimator = 'ReachabilityEstimator'

from system.controller.reachability_estimator._types import PlaceInfo
from system.types import AllowedMapName, PositionAndOrientation
from typing import Optional, Literal, Callable, Self, Iterable
from system.debug import DEBUG
from deprecation import deprecated


def get_path_top() -> str:
    """ returns path to the folder of the current file """
    dirname = os.path.join(os.path.dirname(__file__))
    return dirname


def sample_normal(m, s):
    return np.random.normal(m, s)

from functools import wraps
def report(fun):
    @wraps(fun)
    def _wrapper(self: 'CognitiveMapInterface', *args, **kwargs):
        self.print_debug('Calling', fun.__name__)
        return fun(self, *args, **kwargs)
    return _wrapper


class CognitiveMapInterface(ABC):
    def __init__(
        self,
        reachability_estimator: ReachabilityEstimator,
        load_data_from: Optional[str] = None,
        debug = False,
        metadata: dict = {},
        max_capacity: int|None = None,
        absolute_path=False,
    ):
        """ Abstract base class defining the interface for cognitive map implementations.

        arguments:
        reachability_estimator: ReachabilityEstimator -- reachability estimator used to connect nodes
        load_data_from: str                           -- filename of the snapshot of the cognitive map,
                                                         None if a new cognitive map is being created
        debug: bool                                   -- enables logging
        """

        self.reach_estimator = reachability_estimator
        self.node_network = nx.DiGraph()
        self.node_network.graph.update(metadata)
        self.debug = debug
        if load_data_from is not None:
            self.load(filename=load_data_from, absolute_path=absolute_path)
        # threshold used for determining nodes that represents current location of the agent
        self.active_threshold = 0.9
        # last active node
        self.prior_idx_pc_firing = None
        self.max_capacity = max_capacity

        if self.debug:
            import tempfile
            self.tmpdir = tempfile.TemporaryDirectory()

    class TooManyPlaceCells(Exception):
        pass

    @abstractmethod
    def track_vector_movement(self, pc_firing: list[float], created_new_pc: bool, pc: PlaceCell, **kwargs) -> Optional[PlaceCell]:
        """ Abstract function used to incorporate changes to the map after each vector navigation

        arguments:
        pc_firing: [float]   -- current firings of all place cells
        created_new_pc: bool -- indicates if a new place cell was created after vector navigation
        pc: PlaceCell        -- current location of the agent
        """
        ...

    def find_path(self, start: PlaceCell, goal: PlaceCell) -> Optional[list[PlaceCell]]:
        """ Returns a path in the graph from start to goal nodes"""
        try:
            path = nx.shortest_path(self.node_network, source=start, target=goal, weight='weight')
            # TODO Pierre: we could use A* with distances or something
        except nx.NetworkXNoPath:
            return None

        return path

    def _place_cell_number(self, p: PlaceCell) -> int|Literal['not in map']:
        try:
            return list(self.node_network.nodes).index(p)
        except ValueError:
            return 'not in map'

    def add_node_to_map(self, p: PlaceCell):
        """ Adds a new node to the cognitive map """
        if self.debug:
            pc_id = len(self.node_network.nodes)
            filename = os.path.join(self.tmpdir.name, str(pc_id) + '.npz')
            try:
                with open(filename, 'xb') as file:
                    p.dump(file)
            except Exception:
                filename = '(could not dump)'
            self.print_debug(f'[cognitive_map] Adding node: #={pc_id}, position={p.env_coordinates}, angle={p.angle}, dump_file={filename}')
        if self.max_capacity is not None and len(self.node_network.nodes) > self.max_capacity:
            raise self.TooManyPlaceCells()
        self.node_network.add_node(p, pos=tuple(p.pos))

    @report
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

    def add_bidirectional_edge_to_map(self, p, q, w: float=1, **kwargs):
        """ Adds a new bidirectional weighted edge to the cognitive map with given parameters

        arguments:
        p: PlaceCell -- first node of the edge
        q: PlaceCell -- second node of the edge
        w: float     -- weight of the edge
        **kwargs     -- parameters of the edge
        """
        self.node_network.add_edge(p, q, weight=w, **kwargs)
        self.node_network.add_edge(q, p, weight=w, **kwargs)
        self.print_debug(f'[cognitive_map] Adding bidirectional edge: p={self._place_cell_number(p)}, q={self._place_cell_number(q)}, weight={w}')

    def save(self, filename: str, absolute_path=False):
        """ Stores the current state of the node_network to the file

        arguments:
        filename: str        -- filename of the snapshot
        relative_folder: str -- relative folder (counting from the folder of the current file) of the snapshot file
        """
        relative_folder: str = "data/cognitive_map"
        if not absolute_path:
            directory = os.path.join(get_path_top(), relative_folder)
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, filename)
        nx.write_gpickle(self.node_network, filename)

    def load(self, filename: str, absolute_path=False):
        """ Loads the state of the node_network from the file

        arguments:
        filename: str        -- filename of the snapshot
        relative_folder: str -- relative folder (counting from the folder of the current file) of the snapshot file
        """

        relative_folder: str = "data/cognitive_map"
        if not absolute_path:
            directory = os.path.join(get_path_top(), relative_folder)
            #if not os.path.exists(directory):
            #    raise ValueError("cognitive map not found")
            filename = os.path.join(directory, filename)
        self.node_network = nx.read_gpickle(filename)

    def draw(self, with_labels: bool = True, colors: Optional[list[float]] = None, **draw_kwargs):
        """ Plot the cognitive map

        arguments:
        with_labels: bool -- flag to include node indices as labels
        """
        pos = nx.get_node_attributes(self.node_network, 'pos')
        kwargs = {}
        if with_labels:
            node_list = list(self.node_network.nodes)
            kwargs['labels'] = {i: str(node_list.index(i)) for i in node_list}
        else:
            kwargs.update(dict(node_color='#0065BD', node_size=120, edge_color='#4A4A4A80', width=2))
        if colors:
            kwargs['node_color'] = colors
        nx.draw(self.node_network, pos, **kwargs, **draw_kwargs)
        plt.show()

    def print_debug(self, *params):
        """ Logs information if debug mode is on """
        if self.debug:
            print(*params)

    def postprocess_topological_navigation(self):
        """ Performs map processing after one full topological navigation cycle """
        pass

    def postprocess_vector_navigation(self, node_p: PlaceCell, node_q: PlaceCell, observation_p: PlaceCell,
                                      observation_q: PlaceCell, success: bool):
        """ Performs map processing after one full vector navigation

        arguments:
        node_p: PlaceCell        -- source node in the graph on the start of the vector navigation
        node_q: PlaceCell        -- estimated target node in the graph
        observation_q: PlaceCell -- actual location of the agent on the start of the vector navigation
        observation_p: PlaceCell -- actual location of the agent after vector navigation
        success: bool            -- indicates if the agent reached the target graph node
        """
        pass

    def get_place_cell_network(self, reach_estimator: ReachabilityEstimator = None) -> PlaceCellNetwork:
        """
        Exports a PlaceCellNetwork from the current nodes.
        I (Pierre) don't understand why these are two different classes in the first place.

        arguments:
        reach_estimator -- reachability estimator for the PC network. If not set, `self.reach_estimator` will be used.
        """
        reach_estimator = reach_estimator or self.reach_estimator

        pc_network = PlaceCellNetwork(reach_estimator)
        pc_network.place_cells = list(self.node_network.nodes.keys())
        return pc_network

    def test_place_cell_network(
        self,
        start_pos_angle: PositionAndOrientation,
        arena_size: int,
        env_model: AllowedMapName,
        gc_network,
        from_data: str|bool=False,
        display_freq: int=8,
    ):
        """ Test the drift error of place cells stored in the cognitive map """
        from system.controller.local_controller.local_navigation import LinearLookaheadGcCompass, PodGcCompass
        from system.controller.local_controller.compass import AnalyticalCompass

        delta_avg = 0
        pred_gvs = []  # goal vectors decoded using linear lookahead
        true_gvs = []  # analytically calculated goal vectors
        error = []

        if from_data:
            dirname = os.path.join(os.path.dirname(__file__), "../../experiments/drift_error")

            suffix = f'-{from_data}' if type(from_data) == str else ''

            pred_gvs = np.load(os.path.join(dirname, f"pred_gvs{suffix}.npy"))
            true_gvs = np.load(os.path.join(dirname, f"true_gvs{suffix}.npy"))
            error = true_gvs - pred_gvs
            delta = [np.linalg.norm(i) for i in error]
            delta_avg = np.mean(delta)

        else:
            dirname = os.path.join(os.path.dirname(__file__), "../../experiments/drift_error")
            suffix = '-plane-ll'
            pred_gvs = np.load(os.path.join(dirname, f"pred_gvs{suffix}.npy"))

            from tqdm import tqdm

            compass = LinearLookaheadGcCompass(arena_size=arena_size, gc_network=gc_network)
            # decode goal vectors from current position to every place cell on the cognitive map
            node_list: list[PlaceCell] = list(self.node_network.nodes)
            nodes_length = len(node_list)
            for i, p in enumerate(tqdm(node_list)):
                compass.reset_goal_pc(p)

                #pred_gv = compass.calculate_goal_vector()
                pred_gv = pred_gvs[i]
                true_gv = AnalyticalCompass(start_pos=start_pos_angle[0], goal_pos=p.pos).calculate_goal_vector()

                error_gv = true_gv - pred_gv
                delta = np.linalg.norm(error_gv)

                delta_avg += delta
                #pred_gvs.append(pred_gv)
                true_gvs.append(true_gv)
                error.append(error_gv)

            delta_avg /= nodes_length

        print("Average error:", delta_avg)

        # Plot the drift error on the cognitive map
        import system.plotting.plotHelper as pH

        plt.figure()
        ax = plt.gca()
        pH.add_environment(ax, env_model)
        pH.add_robot(ax, *start_pos_angle)
        pos = nx.get_node_attributes(self.node_network, 'pos')
        nx.draw_networkx_nodes(self.node_network, pos, node_color='#0065BD80')

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
            if i % display_freq != 0:
                continue
            plt.quiver(start_pos_angle[0][0], start_pos_angle[0][1], gv[0], gv[1], color='grey', angles='xy',
                       scale_units='xy', scale=1, width=0.005)
            plt.quiver(start_pos_angle[0][0] + gv[0], start_pos_angle[0][1] + gv[1], error[i][0], error[i][1],
                       color='red', angles='xy', scale_units='xy', scale=1, width=0.005)

        plt.show()


class CognitiveMap(CognitiveMapInterface):
    def __init__(
        self,
        reachability_estimator=None,
        mode: Literal['exploration', 'navigation']="exploration",
        connection: tuple[Literal['all', 'radius'], Literal['delayed', 'instant']]=("all", "delayed"),
        load_data_from=None,
        debug=False
    ):
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
        """
        Update reachability between the nodes.
        Asks the reachability estimator for an estimated reachability and overwrites the existing value.
        """
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
        if self.debug:
            i = self._place_cell_number(p)
        for j, q in enumerate(list(self.node_network.nodes)):

            if q == p:
                continue

            if self.connection[0] == "radius" and np.linalg.norm(q.env_coordinates - p.env_coordinates) > self.radius:
                # No connection above radius
                continue

            reachable_pq, reachability_factor_pq = self.reach_estimator.get_reachability(p, q)
            reachable_qp, reachability_factor_qp = self.reach_estimator.get_reachability(q, p)

            if reachable_pq:
                self.print_debug(f"[cognitive_map] Connecting new node: p={i}, q={j}, factor={reachability_factor_pq}")
                self.node_network.add_weighted_edges_from([(p, q, reachability_factor_pq)])
            if reachable_qp:
                self.print_debug(f"[cognitive_map] Connecting new node: q={j}, p={i}, factor={reachability_factor_qp}")
                self.node_network.add_weighted_edges_from([(q, p, reachability_factor_qp)])

    def add_node_to_map(self, p: PlaceCell):
        super().add_node_to_map(p)

        if self.connection[1] == "instant":
            # Connect the new node to all other nodes in the graph
            self._connect_single_node(p)

    def track_vector_movement(self, pc_firing: list[float], created_new_pc: bool, pc: PlaceCell, lidar: list[float]|None = None, **kwargs) -> Optional[PlaceCell]:
        """Keeps track of current place cell firing and creation of new place cells"""

        # get the currently active place cell
        idx_pc_active = np.argmax(pc_firing)
        pc_active_firing = np.max(pc_firing)

        # Check if we have entered a new place cell
        if created_new_pc:
            entered_different_pc = True
            pc.lidar = lidar
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

            self.print_debug(f"[proprioception] Updating {self.prior_idx_pc_firing=}")
            self.prior_idx_pc_firing = idx_pc_active

    def save(self, *args, **kwargs):
        if self.connection[1] == "delayed":
            self.update_reachabilities()
        CognitiveMapInterface.save(self, *args, **kwargs)

    def postprocess_topological_navigation(self):
        self.update_reachabilities()


class LifelongCognitiveMap(CognitiveMapInterface):
    def __init__(
            self,
            *args,
            add_edges: bool = True,
            remove_edges: bool = True,
            remove_nodes: bool = True,
            add_nodes: bool = True,
            **kwargs,
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

        super().__init__(*args, **kwargs)
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

    def track_vector_movement(self, pc_firing: list[float], created_new_pc: bool, pc: PlaceCell, *, exploration_phase=True, **kwargs) -> Optional[PlaceCell]:
        """ Incorporate changes to the map after each vector navigation tryout. Adds nodes during exploration phase and
            edges during navigation.

        arguments:
        pc_firing: [float]                -- current firings of all place cells
        created_new_pc: bool              -- indicates if a new place cell was created after vector navigation
        pc: PlaceCell                     -- current location of the agent
        kwargs:
             exploration_phase: bool      -- indicates exploration or navigation phase
                                             exploration: do not add edges
                                             navigation: ignore place cell creation
             pc_network: PlaceCellNetwork -- place cell network

        returns:
        pc: PlaceCell                 -- current active node if it exists
        """
        pc_network: PlaceCellNetwork = kwargs.get('pc_network', None)
        if exploration_phase and created_new_pc:
            is_mergeable, mergeable_values = self.is_mergeable(pc)
            if not is_mergeable:
                self.add_and_connect_node(pc)
            else:
                pc_network.place_cells.pop()
                pc_firing = pc_firing[:-1]
        elif not exploration_phase and not created_new_pc:
            if self.add_edges:
                self.process_add_edge(pc_firing, pc_network)
        if np.max(pc_firing) > self.active_threshold:
            if self.prior_idx_pc_firing is None or pc_firing[self.prior_idx_pc_firing] / np.max(pc_firing) < 0.8:
                self.prior_idx_pc_firing = np.argmax(pc_firing)
                self.print_debug(f'[proprioception] Updating {self.prior_idx_pc_firing=}')
            return pc_network.place_cells[self.prior_idx_pc_firing]
#        else:
#            print(f'max. firing[{np.argmax(pc_firing)}] = {np.max(pc_firing)} does not meet threshold {self.active_threshold}')
        return None

    def process_add_edge(self, pc_firing: list[float], pc_network: PlaceCellNetwork):
        """ Helper function. Decides if a new edge should be added between the last active node and the
            current active node

        arguments:
        pc_firing: [float]           -- current firings of all place cells
        pc_network: PlaceCellNetwork -- place cell network
        """
        idx_pc_active = np.argmax(pc_firing)
        pc_active_firing = np.max(pc_firing)

        if pc_active_firing > self.active_threshold and self.prior_idx_pc_firing != idx_pc_active:
            if self.prior_idx_pc_firing and pc_firing[self.prior_idx_pc_firing] / pc_firing[idx_pc_active] < 0.8:
                # If we have entered place cell p after being in place cell q during
                # navigation, q is definitely reachable and the edge gets updated accordingly.
                q = pc_network.place_cells[self.prior_idx_pc_firing]
                pc_new = pc_network.place_cells[idx_pc_active]
                if (q in self.node_network and pc_new in self.node_network and
                        q not in self.node_network[pc_new] and q != pc_new):
                    self.print_debug(f"[cognitive_map] adding edge [{self.prior_idx_pc_firing}-{idx_pc_active}]")
                    self.add_bidirectional_edge_to_map(q, pc_new,
                                                       sample_normal(0.5, self.sigma),
                                                       connectivity_probability=0.8,
                                                       mu=0.5,
                                                       sigma=self.sigma)

    def is_connectable(self, p: PlaceCell, q: PlaceCell) -> tuple[bool, float]:
        """ Helper function. Checks if two waypoints p and q are connectable."""
        return self.reach_estimator.get_reachability(p, q)

    def is_mergeable(self, p: PlaceCell) -> tuple[bool, list[bool]]:
        """ Helper function. Checks if the waypoint p is mergeable with the existing graph"""
        mergeable_values = [self.reach_estimator.is_same(p, q) for q in self.node_network.nodes]
        try:
            return np.any(self.reach_estimator.is_same_batch(p, self.node_network.nodes)), mergeable_values
        except AttributeError:
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

        self.print_debug(
            f"[navigation] edge [{list(self.node_network.nodes).index(node_p)}-{list(self.node_network.nodes).index(node_q)}]: " +
            f"success {success} conn {self.node_network[node_q][node_p]['connectivity_probability']}")

        if not success and self.remove_edges:
            if self.node_network[node_q][node_p]['connectivity_probability'] < self.threshold_edge_removal:
                self.remove_bidirectional_edge(node_p, node_q)

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
        nodelist = list(self.node_network.nodes)
        self.print_debug(
            f"[cognitive_map] deleting edge [{nodelist.index(node_p)}-{nodelist.index(node_q)}]: " +
            f"conn {self.node_network[node_q][node_p].get('connectivity_probability', '(no value)')}")
        self.node_network.remove_edge(node_p, node_q)
        self.node_network.remove_edge(node_q, node_p)

    def add_bidirectional_edge_to_map(self, node_p: PlaceCell, node_q: PlaceCell, w: float, *, connectivity_probability: float, mu: float, sigma: float):
        """ Just to assert that all edges in a LifelongCognitiveMap have these properties """
        super().add_bidirectional_edge_to_map(node_p, node_q, w, connectivity_probability=connectivity_probability, mu=mu, sigma=sigma)

    def add_bidirectional_edge_to_map_by_probability(self, node_p: PlaceCell, node_q: PlaceCell, p: float):
        self.add_bidirectional_edge_to_map(node_p, node_q,
            sample_normal(1 - p, self.sigma),
            connectivity_probability=p,
            mu = 1 - p,
            sigma=self.sigma
        )

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
                    self.print_debug(f"[cognitive_map] Nodes {nodes.index(node_p)} and {nodes.index(node_q)} are duplicates, " +
                                     f"deleting {nodes.index(node_p)}")
                    for neighbor in self.node_network[node_p]:
                        if neighbor not in self.node_network[node_q] and neighbor != node_q:
                            edge_attributes_dict = self.node_network.edges[node_p, neighbor]
                            self.add_bidirectional_edge_to_map_no_weight(node_q, neighbor, **edge_attributes_dict)
                    deleted.append(node_p)
        for node in deleted:
            self.print_debug(f'[cognitive_map] Remove node: reason=dedup, #={self._place_cell_number(node)}')
            self.node_network.remove_node(node)

    def are_duplicates(self, node_p: PlaceCell, node_q: PlaceCell):
        """ Helper function, checks if two nodes are duplicates of each other """
        set_p = set(self.node_network[node_p])
        set_q = set(self.node_network[node_q])
        common = len(set_p.intersection(set_q))

        return (
            common >= len(set_p) - self.max_number_unique_neighbors_for_deletion and
            common >= len(set_q) - self.max_number_unique_neighbors_for_deletion and
            len(set_p) >= self.min_node_degree_for_deletion and
            len(set_q) >= self.min_node_degree_for_deletion
        )

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
                    self.add_bidirectional_edge_to_map_by_probability(pc, node, weight)

    def retrace_logs(self, lines: Iterable[str],
        callbacks: list[Callable[[int, Self], None]] = [],
        robot_positions: list|None = None,
        accept_incomplete_last_line=False,
        recreate_pcs=True,
    ) -> int:
        """
        Read a log file and replay each step described there. So a canceled navigation can be continued later.
        parameters:
        lines -- lines of the log file, without ending newline
        callback -- a function that will be called after every line
        robot_positions -- a list that will be filled with the reported positions of the robot
        accept_incomplete_last_line -- do not raise an error if the last line raises an error. Buffering can cause the last line to be incomplete
        recreate_pcs -- recreate the PCs that are reported as created in the log. Not necessary if you don't need the full final cognitive map.

        returns:
        nb_steps: int    -- the number of navigation steps
        """
        import sys

        gc_network = None

        nb_steps = 0
        pcs = list(self.node_network.nodes)
        lines = iter(lines)
        for i, line in enumerate(lines):
            full_line = line
            try:
                if line.startswith('Correcting:\t'):
                    line = line.removeprefix('Correcting:\t')
                tag = None
                if ' ' in line:
                    tag, line_without_tag = line.split(' ', maxsplit=1)
                    if tag.startswith('[') and tag.endswith(']'):
                        tag = tag[1:-1]
                        line = line_without_tag
                if i == 0 and line.startswith('#') or ('python' in line.split(' ')):
                    print(line, file=sys.stderr)
                elif tag == 'navigation' or tag == 'drift' or tag == 'environment':
                    pass
                elif full_line == '': # Sometimes happens with position estimation
                    pass
                elif any(line.startswith(it) for it in ('Using seed', 'Path', 'Last PC', 'adding edge', 'edge [', 'Vector navigation', 'Recomputed path:', 'LIMIT WAS REACHED STOPPING HERE')):
                    pass
                elif line.startswith('Fail :(') or line.startswith('Success!'):
                    pass
                elif line.startswith('Navigation '):
                    nb_steps += 1
                elif line.startswith('Adding bidirectional edge: '):
                    line = line.removeprefix('Adding bidirectional edge: ')
                    kvpairs = [ kv.split('=') for kv in line.split(', ')]
                    kvpairs = { key: value for key, value in kvpairs }
                    self.add_bidirectional_edge_to_map_by_probability(pcs[int(kvpairs['p'])], pcs[int(kvpairs['q'])], p=float(kvpairs['weight']))
                elif line.startswith('deleting edge ['):
                    line = line.removeprefix('deleting edge [')
                    line = line.split(']')[0]
                    p, q = line.split('-')
                    p, q = int(p), int(q)
                    self.remove_bidirectional_edge(pcs[p], pcs[q])
                elif line.startswith('Adding node: '):
                    line = line.removeprefix('Adding node: ')
                    kvpairs = [kv.split('=') for kv in line.split(', ')]
                    kvpairs = {key: value for key, value in kvpairs}
                    assert int(kvpairs['#']) == len(self.node_network.nodes), (int(kvpairs['#']), len(self.node_network.nodes))
                    pos = kvpairs['position']
                    assert pos[0] == '[' and pos[-1] == ']'; pos = pos[1:-1]
                    pos = tuple(map(float, filter(lambda i:i, pos.split(' '))))
                    try:
                        pc = PlaceCell.from_data(PlaceCell.load(kvpairs['dump_file']))
                    except (KeyError, FileNotFoundError):
                        if recreate_pcs:
                            from system.controller.local_controller.local_navigation import create_gc_spiking
                            from system.bio_model.grid_cell_model import GridCellNetwork
                            if gc_network is None:
                                gc_network = GridCellNetwork(from_data=True)
                            gc_network.set_as_current_state(pcs[-1].spikings)
                            if np.linalg.norm(np.array(pos) - np.array(pcs[-1].pos)) < 0.1:
                                spikings = pcs[-1].spikings
                            else:
                                spikings = create_gc_spiking(start=pcs[-1].pos, goal=pos, gc_network_at_start=gc_network)
                        else:
                            spikings = NotImplemented
                        pc = PlaceCell.from_data(PlaceInfo(pos=pos, angle=NotImplemented, img=[0], lidar=NotImplemented, spikings=spikings))
                    pcs.append(pc)
                    self.add_node_to_map(pc)
                elif line.startswith('Updating self.prior_idx_pc_firing'):
                    pass
                elif line.startswith('Robot position'):
                    if robot_positions is not None:
                        line = line.removeprefix('Robot position: ')
                        assert line[0] == '[' and line[-1] == ']'; line = line[1:-1]
                        line = line.strip()
                        x, *_optional_space, y = line.split(' ')
                        x, y = float(x), float(y)
                        robot_positions.append((x, y))
                elif line.startswith('> /') or line.startswith('Uncaught exception'):
                    next(lines) # consuming next value from iterator
                elif line.startswith('(Pdb)') or line.startswith('Post mortem debugger') or line.startswith('->'):
                    pass
                elif line.split(' : ')[0] in ('scalar coverage', 'unobstructed_lines', 'mean_distance_between_nodes'):
                    pass
                else:
                    int(line.split(' ')[0]) # PDB l
                for callback in callbacks:
                    callback(i, self, line=full_line)
            except Exception as err:
                if accept_incomplete_last_line and next(lines, None) is None: # check if the line is really the last
                    # (this will consume the next line if it exists, but we don't care because if it exists we exit the function)
                    pass
                else:
                    raise ValueError('Couldn\'t parse line:', full_line) from err
        return nb_steps

if __name__ == "__main__":
    """ Load and visualize cognitive map + observations with grid cell spikings on both ends of distinct edges  """
    from system.controller.reachability_estimator.ReachabilityDataset import SampleConfig

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('map_filename', nargs='?', default='after_exploration.gpickle')
    parser.add_argument('-n', help='Number of samples to show', type=int, default=10)
    args = parser.parse_args()

    # Adjust what sort of RE you want to use for connecting nodes
    connection_re_type = "neural_network"  # "neural_network" #"simulation" #"view_overlap"
    weights_filename = "re_mse_weights.50"
    map_filename = args.map_filename
    env_model = "Savinov_val3"
    debug = True

    re_config = SampleConfig(grid_cell_spikings=True)

    re = reachability_estimator_factory(connection_re_type, weights_file=weights_filename, #env_model=env_model,
                                        debug=debug, config=re_config)
    # Select the version of the cognitive map to use
    cm = LifelongCognitiveMap(reachability_estimator=re)
    try:
        cm.load(filename=map_filename, absolute_path=True)
    except FileNotFoundError:
        cm.load(filename=map_filename)
    cm.draw()

    import random

    for i in range(args.n):
        # Select an edge to visualize or use a random one
        start, finish = random.sample(list(cm.node_network.edges()), 1)[0]

        plot_cognitive_map_path(cm.node_network, [start, finish], env_model)

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

        plot_grid_cell(start.gc_connections, finish.gc_connections, rows_per_module=2)
