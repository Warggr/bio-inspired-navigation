import numpy as np

from system.bio_model.grid_cell_model import GridCellNetwork
from system.controller.local_controller.compass import Compass, AnalyticalCompass
from system.controller.local_controller.local_navigation import PodGcCompass
from system.controller.local_controller.position_estimation import DoubleCompass
from system.utils import normalize
from typing import BinaryIO, Literal


class MockEnv:
    def __init__(self, print_freq=10, start=(0.0,0.0), start_spikings=None):
        self.print_freq = print_freq

        self.gc_network = GridCellNetwork(from_data=True)
        if start_spikings is not None:
            self.gc_network.set_as_current_state(start_spikings)
        else:
            start_spikings = self.gc_network.consolidate_gc_spiking()
        self.true_compass = AnalyticalCompass(start_pos=start, goal_pos=start)

        compass = Compass.factory('linear_lookahead', gc_network=GridCellNetwork(from_data=True), arena_size=10, cache=False)
        self.error_calculator = DoubleCompass(compass, AnalyticalCompass())
        self.error_calculator.reset_position((start_spikings, start))
        self.robot_position = np.array(start)
        self.t = 0

    def go_to(self, goal, robot_speed=0.5, dt=1e-2, report: Literal['errors', 'positions']='errors'):
        norm_errors = []
        angle_errors = []

        self.true_compass.reset_goal(goal)

        loops = 0
        while not self.true_compass.reached_goal():
            loops += 1
            goal_vector = self.true_compass.calculate_goal_vector()
            if np.linalg.norm(goal_vector) == 0:
                break

            xy_speed = robot_speed * normalize(goal_vector)
            self.t += dt
            self.robot_position += xy_speed * dt
            self.true_compass.reset_position(self.robot_position)
            self.gc_network.track_movement(xy_speed)

            if (loops - 1) % self.print_freq == 0:
                self.error_calculator.reset_goal((self.gc_network.consolidate_gc_spiking(), self.robot_position))
                true_pos, est_pos = self.error_calculator.true_compass.calculate_goal_vector(), self.error_calculator.compass.calculate_goal_vector()
                if report == 'errors':
                    # assert np.all(robot_position == true_pos)
                    # assert not np.all(start_spikings == gc_network.consolidate_gc_spiking())
                    norm_error, angle_error = self.error_calculator.error()
                    print(
                        f"{loops=}: {true_pos=}, {est_pos=}. Error norm={norm_error}, error angle={np.degrees(angle_error)}°",
                        end='\r')
                    for li, i in zip((norm_errors, angle_errors), (norm_error, angle_error)):
                        li.append(i)
                else:
                    for li, i in zip((norm_errors, angle_errors), (true_pos, est_pos)):
                        li.append(i)
        return norm_errors, angle_errors

    def dump(self, file: BinaryIO|str):
        if type(file) == str:
            with open(file, 'wb') as file:
                self.dump(file)
        else:
            np.savez(file, pos=self.robot_position, spikings=self.gc_network.consolidate_gc_spiking(), t=self.t)

    def load(self, file: BinaryIO|str) -> "MockEnv":
        if type(file) == str:
            with open(file, 'rb') as file:
                return self.load(file)
        else:
            data = np.load(file, allow_pickle=True)
            self.robot_position = data['pos']
            self.gc_network.set_as_current_state(data['spikings'])
            self.t = data['t']
            return self


if __name__ == "__main__":
    from system.bio_model.cognitive_map import CognitiveMap

    env_model = 'Savinov_val3'

    cogmap = CognitiveMap(reachability_estimator=None, mode='navigation', load_data_from='after_exploration.gpickle', debug=False)
    #cogmap.draw()

    from system.controller.topological.topological_navigation import TopologicalNavigation
    from system.controller.local_controller.compass import AnalyticalCompass

    pc_network = cogmap.get_place_cell_network()
    pc_network.max_capacity = len(pc_network.place_cells)
    start, end = 73, 82
    start, end = pc_network.place_cells[start], pc_network.place_cells[end]

    gc_network = GridCellNetwork(from_data=True)
    compass = Compass.factory('combo', gc_network=gc_network)
    tj = TopologicalNavigation(env_model, pc_network, cogmap, compass)

    from system.controller.simulation.pybullet_environment import PybulletEnvironment
    from system.bio_model.grid_cell_model import GridCellNetwork
    from system.controller.local_controller.local_controller import LocalController

    import matplotlib.pyplot as plt
    from system.plotting.plotHelper import add_environment

    place_cell_positions = np.array([pc.pos for pc in pc_network.place_cells])

    def plot_every_gc_error(double_compass: DoubleCompass):
        fig, ax = plt.subplots()
        add_environment(ax, env_model)

        pc_firing = pc_network.compute_firing_values(gc_network=double_compass.gc_network)

        analytical_compass = double_compass.true_compass

        ax.scatter(x=[pc.pos[0] for pc in pc_network.place_cells], y=[pc.pos[1] for pc in pc_network.place_cells], c=pc_firing, label='Place cells with firing strength')
        ax.plot(*(analytical_compass.current_pos), 'bx', label='Current position')

        ax.plot(*(analytical_compass.goal_pos), 'gx', label='Next goal node')
        pod_estimated_position = double_compass.compass.calculate_goal_vector()
        ax.plot(*(analytical_compass.current_pos + pod_estimated_position), 'yx', label='Estimated goal node')

        real_place_cell = pc_network.place_cells[np.argmin(np.linalg.norm(place_cell_positions - analytical_compass.current_pos, axis=1))]
        ax.plot(*(real_place_cell.pos), 'cx', label='Current place cell')

        tmp_compass = DoubleCompass(PodGcCompass(), AnalyticalCompass())

        pod_compass = tmp_compass.compass
        pod_compass.reset_position(real_place_cell.spikings)
        pod_compass.reset_goal(double_compass.compass.gc_network.target_spiking)
        ax.plot(*(np.array(real_place_cell.pos) + pod_compass.calculate_goal_vector()), 'rx', label='Estimated goal node with corrected spikings')

        tmp_compass.reset_position(tmp_compass.parse(pc_network.place_cells[0]))
        # decode goal vectors from current position to every place cell on the cognitive map
        quivers = np.zeros((len(pc_network.place_cells), 4))
        for i, p in enumerate(pc_network.place_cells):
            tmp_compass.reset_goal(tmp_compass.parse(p))

            error_gv = tmp_compass.error()[0][0]
            quivers[i, 0:2] = p.pos
            quivers[i, 2:4] = error_gv
        ax.quiver(quivers[:, 0], quivers[:, 1], quivers[:, 2], quivers[:, 3], label='Grid cell drift as measured by start node', width=0.005)

        analytical_compass = double_compass.true_compass
        pod_compass = double_compass.compass

        estimations = np.zeros((len(pc_network.place_cells), 2))
        tmp_compass.reset_goal((analytical_compass.current_pos, pod_compass.gc_network.consolidate_gc_spiking()))
        for i, p in enumerate(pc_network.place_cells):
            tmp_compass.reset_goal(tmp_compass.parse(p))

            estimation = np.array(p.pos) + tmp_compass.compass.calculate_goal_vector()
            estimations[i, 0:2] = estimation
        plt.scatter(estimations[:, 0], estimations[:, 1], label='Estimations of the current position')
        mean_est = np.mean(estimations, axis=0)
        plt.plot(*mean_est, 'go', label='Mean estimation')

        fig.legend()
        plt.show()

    with PybulletEnvironment(env_model, visualize=False, start=start.pos, build_data_set=True) as env:
        robot = env.robot
        pc_network.add_angles_and_lidar(env)

        compass.reset_position(compass.parse(start))

        double_compass = DoubleCompass(PodGcCompass(), AnalyticalCompass())
        double_compass.reset_position(double_compass.parse(start))

        gc_network = GridCellNetwork(from_data=True)
        gc_network.set_as_current_state(start.spikings)

        errors = []
        def on_reset_goal(new_goal, robot):
            double_compass.reset_position((robot.position, gc_network.consolidate_gc_spiking()))
            double_compass.reset_goal((robot.position + new_goal, gc_network.target_spiking))
            errors.append(double_compass.error())
            plot_every_gc_error(double_compass)
            print(errors[-1])

        controller = LocalController.default()
        controller.on_reset_goal.append(on_reset_goal)

        tj.navigate(start, end, gc_network, controller, env=env, robot=robot, add_nodes=False)
