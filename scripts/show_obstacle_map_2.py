from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.controller.local_controller.local_navigation import vector_navigation
from system.controller.local_controller.local_controller import LocalController, controller_rules
import numpy as np
from system.types import Angle
import matplotlib.pyplot as plt
from system.plotting.plotHelper import add_environment

from system.tests.local_controller_angle_test import DirectionalCompass, WALL_CENTER

fig, ax = plt.subplots()

add_environment(ax, 'obstacle_map_2')

a = [], [], [], [], [], []

def angle_test(
    angle_: Angle,
    controller: LocalController,
    env: PybulletEnvironment,
) -> bool:
    angle = np.radians(angle_)
    goal_direction = np.array([-np.sin(angle), -np.cos(angle)])
    start = WALL_CENTER - 2*goal_direction
    compass = DirectionalCompass(angle, goal_offset=3, zero=WALL_CENTER)
    compass.reset_position(compass.zvalue(start))
    with Robot(env, base_position=start, base_orientation=-angle-np.radians(90)) as robot:
        success, _ = vector_navigation(env, compass, controller=controller)
        x, y = zip(*robot.data_collector.xy_coordinates)
        ax.plot(x, y, marker='.', label=f'angle = {angle_}°')
        d = 100
        a[0].append(2*x[0] - x[d]); a[1].append(2*y[0] - y[d]); a[2].append(x[d] - x[0]); a[3].append(y[d] - y[0]); a[4].append(x[0]); a[5].append(y[0])
    return success


controller = LocalController.default()
controller.on_reset_goal = [hook for hook in controller.on_reset_goal if type(hook) != controller_rules.TurnToGoal]

with PybulletEnvironment(env_model="obstacle_map_2", visualize=False, contains_robot=False) as env:
    precision = 5
    for i in range(0, precision+1):
        success = angle_test(90 * (i / precision), controller, env)

breakpoint()
#ax.quiver(a[0], a[1], a[2], a[3], width=0.01, zorder = 10)
#ax.scatter([], [], marker=r'$\longrightarrow$', c="black", s=120, label="Start of navigation")
ax.scatter(a[4], a[5], c='r', marker='x', label="Start of navigation", zorder=10)
ax.legend(loc='upper left')
plt.show()
