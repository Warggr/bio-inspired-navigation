""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
import math

import matplotlib.colors as mcolors

import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib import rc

from system.plotting.plotHelper import TUM_colors

mpl.rcParams['animation.ffmpeg_path'] = "ffmpeg/ffmpeg"
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from system.plotting.helper import compute_theta, compute_axis_limits
from system.plotting.plotHelper import *

# from plotting.plotThesis import add_cognitive_map

from typing import Optional

colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
colors = [(0, 0, 1, c) for c in np.linspace(0, 1, 100)]
cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)
csfont = {'fontname': 'Comic Sans MS'}
hfont = {'fontname': 'Avenir'}

cmap = plt.cm.get_cmap("tab20")  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.9, .9, .9, 0.8)
# create the new map
cmap20 = mcolors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

cmap_binary = mcolors.ListedColormap([TUM_colors['TUMWhite'], TUM_colors['TUMGray']])

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(256 / 256, 0 / 256, N)
vals[:, 1] = np.linspace(256 / 256, 101 / 256, N)
vals[:, 2] = np.linspace(256 / 256, 189 / 256, N)
tum_blue_map = mcolors.ListedColormap(vals)

vals2 = np.ones((N, 4))
vals2[:, 0] = np.linspace(256 / 256, 128 / 256, N)
vals2[:, 1] = np.linspace(256 / 256, 128 / 256, N)
vals2[:, 2] = np.linspace(256 / 256, 128 / 256, N)
tum_grey_map = mcolors.ListedColormap(vals2)

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plotTrajectory(xy_coordinates):
    x, y = zip(*xy_coordinates)
    plt.figure(1)
    plt.scatter(x, y, s=0.2)

    nr_labels = 10
    step_label = int(len(xy_coordinates) / nr_labels)
    for i in range(len(x)):
        if i % step_label == 0:
            xi = x[i]
            yi = y[i]
            label = str(int(i / step_label))

            plt.annotate(label,  # this is the text
                         (xi, yi),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 0.1),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

    plt.axis('equal')
    plt.legend(['Trajectory'])
    plt.show()


def plotTrajectoryInEnvironment(env: Optional['PybulletEnvironment'] = None, title="", xy_coordinates=None, env_model=None, cognitive_map=None, path=None,
                                goal=None, trajectory=True, start=None, end=None):
    if xy_coordinates is None and env is not None and env.robot is not None:
        xy_coordinates = env.robot.data_collector.xy_coordinates
    if not env_model:
        env_model = env.env_model

    fig, ax = plt.subplots()
    add_environment(ax, env_model)

    # plot goal vector
    if goal:
        X, Y = xy_coordinates[-1]
        U, V = goal

        q = ax.quiver(X, Y, U, V, units='xy', scale=1, color=TUM_colors["TUMAccentGreen"])

    if cognitive_map:
        G = cognitive_map.node_network
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_nodes(G, pos, node_color='#0065BD80', node_size=60)
        nx.draw_networkx_edges(G, pos, edge_color='#CCCCC6')

        if path:
            # draw_path
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='#E3722280', node_size=60)
            G = G.to_undirected()
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='#E3722280', width=3)
    if start is not None:
        circle = plt.Circle((start[0], start[1]), 0.2, color=TUM_colors['TUMAccentOrange'], alpha=0.8)
        ax.add_artist(circle)
    if end is not None:
        circle = plt.Circle((end[0], end[1]), 0.2, color=TUM_colors['TUMAccentOrange'], alpha=0.8)
        ax.add_artist(circle)

    if trajectory:
        x, y = zip(*xy_coordinates)
        ax.scatter(x, y, color='#992225', s=10, linewidths=0.5)

    add_environment(ax, env_model)

    # add_robot(ax, env)
    # if env.goal_pos:
        # add_goal(ax, env)

    # add title
    plt.title(title)
    plt.show()


def plotStartGoalDataset(env_model, starts_goals):
    # get the dimensions without having to adjust them here
    fig, ax = plt.subplots()
    add_environment(ax, env_model)
    for i, e in enumerate(starts_goals):
        start, goal = e
        circle = plt.Circle((start[0], start[1]), 0.2, color=TUM_colors['TUMAccentBlue'], alpha=1)
        ax.add_artist(circle)
        circle = plt.Circle((goal[0], goal[1]), 0.2, color=TUM_colors['TUMAccentOrange'], alpha=1)
        ax.add_artist(circle)
    plt.show()


def plotStartGoalPair(env_model, start_position, start_heading, target_position, target_heading):
    fig, ax = plt.subplots()
    add_environment(ax, env_model)
    circle = plt.Circle(start_position, 0.2, color=TUM_colors['TUMAccentBlue'], alpha=1)
    ax.add_artist(circle)
    arrow = plt.Arrow(start_position[0], start_position[1], math.cos(start_heading), math.sin(start_heading),
                      color=TUM_colors['TUMAccentBlue'], alpha=1)
    ax.add_artist(arrow)
    circle = plt.Circle(target_position, 0.2, color=TUM_colors['TUMAccentOrange'], alpha=1)
    ax.add_artist(circle)
    arrow = plt.Arrow(target_position[0], target_position[1], math.cos(target_heading), math.sin(target_heading),
                      color=TUM_colors['TUMAccentOrange'], alpha=1)
    ax.add_artist(arrow)
    plt.show()



def plotSpeeds(xy_speed):
    plt.figure()
    plt.plot(xy_speed)
    # plt.legend(['x-speed', 'y-speed'])
    plt.show()


def plotGridCellSheet(gc_modules):
    fig = plt.figure()

    for i in range(len(gc_modules)):
        gc = gc_modules[i]
        s = np.reshape(gc.s, (gc.n, gc.n))
        fig.add_subplot(1, len(gc_modules), i + 1)
        plt.imshow(s, origin="lower")
    plt.show()


def plotMotorOutputNeuron(mo_network):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    N = mo_network.n_theta
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
    radii = []
    for mon in mo_network.motor_output_neurons:
        radii.append(mon.u)
    width = 1.9 * np.pi / N
    bars = ax.bar(theta, radii, width=width, bottom=0.0)
    for r, bar in zip(radii, bars):
        bar.set_alpha(0.5)

    plt.show()


def convertInVector(p_array, n_theta, n_xy, n):
    V = []
    origin_x = []
    origin_y = []
    nr = 0
    for a in range(n_theta):
        theta = (2 * np.pi / n_theta) * a
        for i in range(n_xy):
            x = int((n / n_xy) * i)
            for j in range(n_xy):
                y = int((n / n_xy) * j)
                origin_x.append(x)
                origin_y.append(y)
                length = p_array[nr]
                if length > 1:
                    length = 1
                else:
                    length = 0
                vec = [length * np.cos(theta), length * np.sin(theta)]
                V.append(vec)
                nr = nr + 1
    return [V, [origin_x, origin_y]]


def plotPhaseOffsetDetector(gc_modules, p_array_modules, n_theta, n_xy):
    fig = plt.figure()

    for m, gc in enumerate(gc_modules):
        p_array = p_array_modules[m]
        [V1, origin1] = convertInVector(p_array, n_theta, n_xy, gc.n)
        V = np.array(V1)
        origin = np.array(origin1)

        s = np.reshape(gc.s, (gc.n, gc.n))
        t = np.reshape(gc.t, (gc.n, gc.n))
        fig.add_subplot(1, len(gc_modules), m + 1)
        plt.imshow(s, origin="lower")
        plt.imshow(t, alpha=0.8, cmap=cmapred, origin="lower")
        plt.quiver(*origin, V[:, 0], V[:, 1], color='r', scale=7)

    plt.show()


def plotSheet(sheet):
    plt.imshow(sheet, origin="lower")
    plt.colorbar()
    plt.show()


def plotSinglePhaseOffsetDetector(gc, pod):
    plotVector(gc.s)

    # i = int(y * 40 + x)
    # plotVector(gc.w[i])

    sum_s = np.sum(gc.s)
    print("Sum s", sum_s)

    plotVector(pod.w_in)
    multiply_s = np.multiply(gc.s, pod.w_in)
    plotVector(multiply_s)
    dot_s = np.dot(gc.s, pod.w_in)
    print("Dot product s", dot_s)

    sum_t = np.sum(gc.t)
    print("Sum t", sum_t)

    plotVector(gc.t, target=True)
    plotVector(pod.w_ex)
    multiply_t = np.multiply(gc.t, pod.w_ex)
    plotVector(multiply_t)
    dot_t = np.dot(gc.t, pod.w_ex)
    print("Dot product t", dot_t)

    p = pod.calculate_p(gc.s, gc.t)
    print("p", p)


def plotTrajectoryWithVector(xy_coordinates, vec_array, step_size, i):
    x, y = zip(*xy_coordinates)
    x = x[:i]
    y = y[:i]
    plt.figure(1)
    # plt.subplot(211)
    plt.scatter(x, y, s=0.2)

    if i != 0:
        for j in range(int(i / step_size) + 1):
            vec = vec_array[j]
            length = np.linalg.norm(vec)
            if length != 0:
                n_vec = vec / length
                # n_vec = vec * 10**3
                if j * step_size < len(x):
                    xi = x[j * step_size]
                    yi = y[j * step_size]
                else:
                    xi = x[-1]
                    yi = y[-1]
                label = str(j * step_size)

                plt.annotate(label,  # this is the text
                             (xi, yi),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center

                label2 = '{:.2e}'.format(length)
                plt.annotate(label2,  # this is the text
                             (xi, yi),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, - 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center

                plt.quiver(*[xi, yi], n_vec[0], n_vec[1], scale=10)

    plt.axis('equal')
    plt.legend(['Trajectory'])
    # plt.subplot(212)
    # plt.plot(orientation_angle)
    # plt.legend(['angle'])
    plt.show()


def plot3DSheet(s_vectors):
    fig = plt.figure()

    for idx, s in enumerate(s_vectors):
        n = int(np.sqrt(len(s)))
        sheet = np.reshape(s, (n, n))
        xmin, xmax, nx = 0, sheet.shape[0] - 1, sheet.shape[0]
        ymin, ymax, ny = 0, sheet.shape[1] - 1, sheet.shape[1]
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)

        ax = fig.add_subplot(2, int(len(s_vectors) / 2), idx + 1, projection='3d')
        ax.plot_surface(X, Y, sheet, cmap='plasma')
        ax.set_zlim(0, np.max(sheet) + 2)
    plt.show()


def plotSheetsWithMaxima(s, t, s_max, t_max):
    s_max_x, s_max_y = zip(*s_max)
    t_max_x, t_max_y = zip(*t_max)

    n = int(np.sqrt(len(s)))

    s = np.reshape(s, (n, n))
    t = np.reshape(t, (n, n))
    plt.imshow(s, origin="lower")
    plt.imshow(t, alpha=0.8, cmap=cmapred, origin="lower")
    plt.scatter(s_max_x, s_max_y, color="blue")
    plt.scatter(t_max_x, t_max_y, color="red")

    plt.show()


def plotCurrentAndTarget(gc_modules, virtual=False):
    fig = plt.figure()

    for m, gc in enumerate(gc_modules):
        if virtual:
            s = np.reshape(gc.s_virtual, (gc.n, gc.n))
        else:
            s = np.reshape(gc.s, (gc.n, gc.n))
        t = np.reshape(gc.t, (gc.n, gc.n))
        fig.add_subplot(1, len(gc_modules), m + 1)
        plt.imshow(s, origin="lower")
        plt.imshow(t, alpha=0.8, cmap=cmapred, origin="lower")

    plt.show()


def plotCurrentAndTargetMatched(gc_modules, matches_array, vectors_array):
    fig = plt.figure()

    for m, gc in enumerate(gc_modules):
        s = np.reshape(gc.s, (gc.n, gc.n))
        t = np.reshape(gc.t, (gc.n, gc.n))

        fig.add_subplot(1, len(gc_modules), m + 1)
        plt.imshow(s, origin="lower")
        plt.imshow(t, alpha=0.8, cmap=cmapred, origin="lower")

        matches = matches_array[m]
        vectors = vectors_array[m]

        if len(matches) != 0 and len(vectors) != 0:
            s_max = list(matches.keys())
            s_max_x, s_max_y = zip(*s_max)
            t_max = list(matches.values())
            t_max_x, t_max_y = zip(*t_max)

            origin_x, origin_y = zip(*list(vectors.keys()))
            vectors_x, vectors_y = zip(*list(vectors.values()))

            plt.scatter(s_max_x, s_max_y, color="blue", s=1)
            plt.scatter(t_max_x, t_max_y, color="red", s=1)

            plt.quiver(origin_x, origin_y, vectors_x, vectors_y, color='w', width=0.01, scale=1, scale_units='xy')

    plt.show()


def plot_angles(real_trajectory, target, vec_array1=None, vec_array2=None, vec_array3=None):
    fig = plt.figure()
    legend = []

    start = 400
    if len(real_trajectory) < start:
        start = 0
    stop = len(real_trajectory)
    num = int((stop - start))
    x = np.linspace(start, stop, num=num)

    if vec_array1 is not None and len(vec_array1) > 0:
        angle_array = []
        for i, vec in enumerate(vec_array1):
            if i >= start:
                angle = compute_theta(vec)
                angle_array.append(angle)
        plt.plot(x, angle_array, '--')
        legend.append('Path Integration')

    if vec_array2 is not None and len(vec_array2) > 0:
        angle_array = []
        for i, vec in enumerate(vec_array2):
            if i >= start:
                angle = compute_theta(vec)
                angle_array.append(angle)
        plt.plot(x, angle_array, '--')
        legend.append('Spike detection')

    if vec_array3 is not None and len(vec_array3) > 0:
        angle_array = []
        for i, vec in enumerate(vec_array3):
            if i >= start:
                angle = compute_theta(vec)
                angle_array.append(angle)
        plt.plot(x, angle_array, '--')
        legend.append('Phase Offset Detector')

    angle_array_real = []
    for i, xy in enumerate(real_trajectory):
        if i >= start:
            vec = np.array(target) - np.array(xy)
            angle = compute_theta(vec)
            angle_array_real.append(angle)
    plt.plot(x, angle_array_real)
    legend.append('Real Angle')

    plt.legend(legend)
    plt.show()


def plot_current_state(env : 'PybulletEnvironment', gc_modules, f_gc, f_t, f_mon,
                       matches_array=None, vectors_array=None, pc_active_array=None,
                       pc_network=None, cognitive_map=None, exploration_phase=False, goal_vector=None):
    xy_coordinates = env.xy_coordinates

    # Trajectory plot
    f_t.clear()
    limits_t = compute_axis_limits(env.arena_size, environment=env.env_model)

    if env.env_model == "linear_sunburst":
        f_t.axis('square')
        f_t.set_xlim(-0.5, 11.5)
        f_t.set_ylim(-0.5, 11.5)
    else:
        f_t.set_xlim(limits_t[0], limits_t[1])
        f_t.set_ylim(limits_t[2], limits_t[3])

    ''' pc-network and cognitive map not used'''
    # if pc_network is not None and cognitive_map is not None:
    #     ax = f_t
    #     add_cognitive_map(ax, pc_network, cognitive_map)

    x, y = zip(*xy_coordinates)
    f_t.scatter(x[0], y[0], color=TUM_colors['TUMGray'], s=1)

    # Plot obstacles
    add_environment(f_t, env.env_model)
    add_robot(f_t, env.robot.data_collector.data[-1].xy_coordinates, env.robot.data_collector.data[-1].angle) # TODO

    # Grid Cell Modules plot
    for m, gc in enumerate(gc_modules):
        if m < 4:
            f_gc[m].clear()
            s = np.reshape(gc.s, (gc.n, gc.n))
            t = np.reshape(gc.t, (gc.n, gc.n))
            f_gc[m].imshow(s, origin="lower", cmap=tum_blue_map)
            f_gc[m].imshow(t, alpha=0.5, cmap=tum_grey_map, origin="lower")

            if matches_array is not None and len(matches_array[m]) != 0:
                matches = matches_array[m]

                s_max = list(matches.keys())
                s_max_x, s_max_y = zip(*s_max)
                t_max = list(matches.values())
                t_max_x, t_max_y = zip(*t_max)

                f_gc[m].scatter(s_max_x, s_max_y, color=TUM_colors['TUMBlue'], s=1)
                f_gc[m].scatter(t_max_x, t_max_y, color=TUM_colors['TUMGray'], s=1)

            if vectors_array is not None and len(vectors_array[m]) != 0:
                vectors = vectors_array[m]
                origin_x, origin_y = zip(*list(vectors.keys()))
                vectors_x, vectors_y = zip(*list(vectors.values()))

                f_gc[m].quiver(origin_x, origin_y, vectors_x, vectors_y, color=TUM_colors['TUMDarkGray'], width=0.01,
                               scale=1, scale_units='xy')

    # Description Plot
    f_mon.clear()
    f_mon.axis("off")
    if exploration_phase:
        description_string = r"Currently in exploration phase"
    else:
        description_string = r"Currently in navigation phase"
    f_mon.annotate(description_string, xy=(0, 0.8), fontweight='bold')

    if goal_vector is not None and not exploration_phase:
        goal_vector_string = r"Computed vector: [" + "{:.2f}".format(goal_vector[0]) + ", " + "{:.2f}".format(
            goal_vector[1]) + "]"
        f_mon.annotate(goal_vector_string, xy=(0, 0.6))

        actual_vector = env.goal_location - xy_coordinates[-1]
        goal_vector_string = r"Actual vector:        [" + "{:.2f}".format(actual_vector[0]) + ", " + "{:.2f}".format(
            actual_vector[1]) + "]"
        f_mon.annotate(goal_vector_string, xy=(0, 0.5))

        error_vector = actual_vector - goal_vector
        error_string = r"Error: " + "{:.2f}".format(np.linalg.norm(error_vector))
        f_mon.annotate(error_string, xy=(0, 0.4))


def layout_video():
    fig = plt.figure(constrained_layout=False)
    fig.suptitle(r'Biologically inspired navigation', fontsize=12, x=0.08, y=0.91, ha='left', fontweight='semibold')
    logo = plt.imread('plotting/tum_logo.png')
    fig.figimage(logo, 530, 395, zorder=1)

    gs0 = fig.add_gridspec(2, 1)

    gs01 = gs0[0].subgridspec(nrows=1, ncols=4, wspace=0.3)

    f_gc1 = fig.add_subplot(gs01[0:1, 0:1])
    f_gc2 = fig.add_subplot(gs01[0:1, 1:2])
    f_gc3 = fig.add_subplot(gs01[0:1, 2:3])
    f_gc4 = fig.add_subplot(gs01[0:1, 3:4])
    f_gc = [f_gc1, f_gc2, f_gc3, f_gc4]

    gs02 = gs0[1].subgridspec(nrows=2, ncols=4)

    f_t = fig.add_subplot(gs02[0:2, 0:2])
    f_mon = fig.add_subplot(gs02[0:2, 2:4])
    f_mon.axis('off')

    return [fig, f_gc, f_t, f_mon]


def place_cell_plot(xy_coordinates, pc_active_array):
    x, y = zip(*xy_coordinates)
    plt.figure()
    idx_pc_active = np.array(pc_active_array)[:, 0] + 5
    # Threshold clearly differentiates place cells from each each other
    spiking_value = np.where(np.array(pc_active_array)[:, 1] > 0.75, 1, 0)
    idx_pc_active = np.multiply(idx_pc_active, spiking_value)

    plt.scatter(x, y, s=3, c=idx_pc_active, cmap=cmap20)

    plt.axis('equal')
    plt.legend(['Trajectory'])
    plt.show()


def error_plot(error_array):
    plt.hist(error_array, 50, density=True)
    plt.show()


def cognitive_map_plot(pc_network, cognitive_map, xy_coordinates=None, pc_active_array=None, environment=None):
    plt.figure()

    if xy_coordinates is not None:
        x, y = zip(*xy_coordinates)
        if pc_active_array is not None:
            idx_pc_active = np.array(pc_active_array)[:, 0] + 5
            # Threshold clearly differentiates place cells from each each other
            spiking_value = np.where(np.array(pc_active_array)[:, 1] > 0.75, 1, 0)
            idx_pc_active = np.multiply(idx_pc_active, spiking_value)
            plt.scatter(x, y, s=3, c=idx_pc_active, cmap=cmap20)
        else:
            plt.scatter(x, y, s=3, c=cmaplist[0])

    ax = plt.gca()
    for i, pc in enumerate(pc_network.place_cells):
        circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.3,
                            fc='r', alpha=cognitive_map.reward_cells[i] ** 2 * 0.6, ec='k')
        ax.add_artist(circle)
        circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.3,
                                   alpha=0.2, ec='k', fill=False)
        ax.add_artist(circle_border)

        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j:
                x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                plt.plot(x_values, y_values, color='k', alpha=0.2)

    # Plot obstacles
    add_environment(ax, environment)

    limits_t = compute_axis_limits(11, environment=environment)
    plt.xlim(limits_t[0], limits_t[1])
    plt.ylim(limits_t[2], limits_t[3])
    plt.show()


def plot_linear_lookahead(f_gc, f_t, f_mon, frame, gc_network, xy_coordinates=None, reward_array=None, goal_found=None):
    for m, gc in enumerate(gc_network.gc_modules):
        if m > 1:
            m = m - 2
            f_gc[m].clear()
            s = np.reshape(gc.s_video_array[frame], (gc.n, gc.n))
            t = np.reshape(gc.t, (gc.n, gc.n))
            f_gc[m].imshow(s, origin="lower")
            f_gc[m].imshow(t, alpha=0.8, cmap=cmapred, origin="lower")

    f_mon.clear()
    f_mon.axis("off")
    if reward_array is not None:
        reward_string = "Reward is: " + str(reward_array[frame])
        f_mon.annotate(reward_string, xy=(0, 0.6))
    if goal_found is not None:
        goal_string = "Found at:  " + str(goal_found) + " | Currently at: " + str(frame)
        f_mon.annotate(goal_string, xy=(0, 0.8))

    if xy_coordinates is not None:
        # Trajectory plot
        environment = "linear_sunburst"
        f_t.clear()
        limits_t = compute_axis_limits(11, environment=environment)
        f_t.set_xlim(limits_t[0], limits_t[1])
        f_t.set_ylim(limits_t[2], limits_t[3])
        if len(xy_coordinates) > 1:
            x, y = zip(*xy_coordinates)
            f_t.scatter(x, y, color="grey", s=0.3)
            f_t.scatter(x[0], y[0], color="red", s=1)

            size = 0.05
            heading = np.array([x[-1] - x[-10], y[-1] - y[-10]])
            heading = 10 ** -5 * heading / np.linalg.norm(heading)
            f_t.quiver(x[-1], y[-1], heading[0], heading[1], scale_units="dots", width=size, color="k",
                       headwidth=8, headlength=10, headaxislength=10)

        ax = plt.gca()
        add_environment(ax, environment)


def export_linear_lookahead_video(gc_network, filename, xy_coordinates=None, reward_array=None, goal_found=None):
    [fig, f_gc, f_t, f_mon] = layout_video()
    fps = 5
    length = len(gc_network.gc_modules[0].s_video_array)
    step = int((1 / fps) / gc_network.dt)
    frames = np.arange(0, length, step)

    def animation_frame(frame):
        plot_linear_lookahead(f_gc, f_t, f_mon, frame, gc_network, xy_coordinates=xy_coordinates,
                              reward_array=reward_array, goal_found=goal_found)

    anim = animation.FuncAnimation(fig, func=animation_frame, frames=frames, interval=1 / fps, blit=False)

    # Finished simulation
    f = filename
    video_writer = animation.FFMpegWriter(fps=fps)
    anim.save(f, writer=video_writer)
    plt.close()


def plot_sub_goal_localization(env, cognitive_map, pc_network, goal_spiking,
                               goal_vector, chosen_idx):
    xy_coordinates = env.xy_coordinates
    fig = plt.figure()

    # Trajectory plot
    maze = True if env.env_model == "linear_sunburst" else False
    limits_t = compute_axis_limits(11, environment=env.env_model)
    plt.xlim(limits_t[0], limits_t[1])
    plt.ylim(limits_t[2], limits_t[3])

    # Plot obstacles
    ax = plt.gca()
    add_environment(ax, env.env_model)

    for i, pc in enumerate(pc_network.place_cells):
        circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.4,
                            fc='r', alpha=cognitive_map.reward_cells[i] ** 2 * 0.6, ec='k')
        ax.add_artist(circle)
        circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.4,
                                   alpha=0.2, ec='k', fill=False)
        ax.add_artist(circle_border)

        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j:
                x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                plt.plot(x_values, y_values, color='k', alpha=0.2)

    x, y = zip(*xy_coordinates)
    plt.scatter(x[0], y[0], color="red", s=1)

    # Plot robot
    add_robot(ax, env.robot.data_collector.xy_position, env.robot.data_collector.angle) # TODO

    plt.quiver(x[-1], y[-1], goal_vector[0], goal_vector[1], color='grey', angles='xy', scale_units='xy', scale=1)

    for idx, angle in enumerate(goal_spiking):

        color = "b"

        if idx == chosen_idx:
            color = "r"
        if goal_spiking[angle]["reward"] == -1:
            color = "grey"
        if goal_spiking[angle]["blocked"]:
            color = "gainsboro"

        vector = np.array([np.cos(angle), np.sin(angle)])
        plt.quiver(x[-1], y[-1], vector[0], vector[1], color=color, angles='xy', scale_units='xy', scale=1)

    plt.show()


def plot_sub_goal_localization_pod(env, cognitive_map, pc_network, sub_goal_dict,
                                   goal_vector, chosen_idx):
    xy_coordinates = env.xy_coordinates
    fig = plt.figure()

    # Trajectory plot
    maze = True if env.env_model == "linear_sunburst" else False
    limits_t = compute_axis_limits(11, environment=env.env_model)
    plt.xlim(limits_t[0], limits_t[1])
    plt.ylim(limits_t[2], limits_t[3])

    # Plot obstacles
    ax = plt.gca()
    add_environment(ax, env.env_model)

    for i, pc in enumerate(pc_network.place_cells):
        circle = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.4,
                            fc='r', alpha=cognitive_map.reward_cells[i] ** 2 * 0.6, ec='k')
        ax.add_artist(circle)
        circle_border = plt.Circle((pc.env_coordinates[0], pc.env_coordinates[1]), 0.4,
                                   alpha=0.2, ec='k', fill=False)
        ax.add_artist(circle_border)

        for j, connection in enumerate(cognitive_map.topology_cells[i]):
            if connection == 1 and i != j:
                x_values = [pc.env_coordinates[0], pc_network.place_cells[j].env_coordinates[0]]
                y_values = [pc.env_coordinates[1], pc_network.place_cells[j].env_coordinates[1]]
                plt.plot(x_values, y_values, color='k', alpha=0.2)

    x, y = zip(*xy_coordinates)
    plt.scatter(x[0], y[0], color="red", s=1)

    # Plot robot
    add_robot(ax, env)

    plt.quiver(x[-1], y[-1], goal_vector[0], goal_vector[1], angles='xy', scale_units='xy', scale=1, )

    for idx, pc_idx in enumerate(sub_goal_dict):

        color = "b"

        if idx == chosen_idx:
            color = "r"
        if sub_goal_dict[idx]["blocked"]:
            color = "gainsboro"

        vector = sub_goal_dict[idx]["goal_vector"]
        plt.quiver(x[-1], y[-1], vector[0], vector[1], color=color, angles='xy', scale_units='xy', scale=1)

    plt.show()


def plot_angle_detection(ray_dist, obstacle_angles, valid_vectors):
    fig = plt.figure()
    for idx in ray_dist:
        if ray_dist[idx]["dist"] < 1.2:
            vector = np.array([np.cos(ray_dist[idx]["angle"]), np.sin(ray_dist[idx]["angle"])]) * ray_dist[idx]["dist"]
            plt.quiver(0, 0, vector[0], vector[1], color='k', angles='xy', scale_units='xy', scale=1)

    for vector in obstacle_angles:
        plt.quiver(0, 0, vector[0], vector[1], color='b', angles='xy', scale_units='xy', scale=1)

    for vector in valid_vectors:
        plt.quiver(0, 0, vector[0], vector[1], color='r', angles='xy', scale_units='xy', scale=1)
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    plt.show()
    plt.close()

# import os
# dirname = os.path.dirname(__file__)
# dirname = dirname[:-24]
# filename = os.path.join(dirname, "controller/simulation/environment/SMTP_map/maze.png")
# print(filename)
# img = plt.imread(filename)
# fig, ax = plt.subplots()
# ax.imshow(img,extent=[-9, 6, -5, 4],origin="lower")
