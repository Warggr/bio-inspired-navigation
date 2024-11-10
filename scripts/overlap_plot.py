import matplotlib.pyplot as plt
import numpy as np
from system.controller.reachability_estimator.reachability_estimation import ViewOverlapReachabilityEstimator
from system.controller.reachability_estimator._types import PlaceInfo
from math import isnan

env_model = 'obstacle_map_0'
env_variant = '1.0'
re = ViewOverlapReachabilityEstimator(env_model)

distance_values = [i / 8 for i in range(8*3 + 1)]

@np.vectorize
def get_overlap(start_distance, goal_distance):
    if isnan(start_distance) or isnan(goal_distance):
        return float('nan')
    corner = (-0.75, 1.25)
    start = (corner[0], corner[1] - start_distance)
    goal = (corner[0] + goal_distance, corner[1])
    start_heading = np.pi / 2 # up
    goal_heading = 0 # left
    start = PlaceInfo(start, start_heading, None, None, None)
    goal = PlaceInfo(goal, goal_heading, None, None, None)
    return re.reachability_factor(start, goal)

#background_function = get_overlap
#background_function = lambda x, y: np.sqrt(x**2 + y**2)
background_function = None

def plot_figure(background_function):
    plot_2d = plt.figure()
    ax_2d = plot_2d.subplots()
    plot_1d = plt.figure()
    ax_1d = plot_1d.subplots()

    if background_function is not None:
        x, y = np.meshgrid(distance_values, distance_values)
        ratios = background_function(x, y)
        distances = ax_2d.pcolor(x, y, ratios, cmap='gnuplot2_r', label='View overlap factor')
        plot_2d.colorbar(distances)

    all_data_points = np.array((0,))

    for bydist, color, xaxis in (
            ('adist', 'lightgreen', 'start'),
            ('zdist', 'darkgreen', 'goal'),
    ):
        yaxis = 'start' if xaxis == 'goal' else 'goal'
        label = f'$d_{{{yaxis},max}} = f(d_{{{xaxis}}})$'
        line_start = f'Maximum handle-able {yaxis}_distance is '
        starts, stops = (np.zeros((len(distance_values), 2)) for _ in range(2))
        lowers, uppers = (np.zeros((len(distance_values),)) for _ in range(2))

        for i, x in enumerate(distance_values):
            filename = f"results/local_controller_width/{bydist}={x}.log"
            with open(filename, 'r') as f:
                line = f.read().strip()
            assert line.startswith(line_start), line; line = line.removeprefix(line_start)
            start, stop = line.split(' ~ ')
            start, stop = [float(s) for s in (start, stop)]
            if stop == 2.0 and not (bydist == 'zdist' and x == 0.875):
                start, stop = float('nan'), float('nan')
            if bydist == 'adist':
                start, stop = (x, start), (x, stop)
            elif bydist == 'zdist':
                start, stop = (start, x), (stop, x)
            if background_function is not None:
                uppers[i], lowers[i] = background_function(*start), background_function(*stop)
            starts[i] = start
            stops[i] = stop

        plot_kwargs = {}
        if bydist == 'adist':
            x, y = distance_values, (stops[:, 1] + starts[:, 1]) / 2
            plot_kwargs['yerr'] = (stops[:, 1] - starts[:, 1]) / 2
        elif bydist == 'zdist':
            x, y = (stops[:, 0] + starts[:, 0]) / 2, distance_values
            plot_kwargs['xerr'] = (stops[:, 0] - starts[:, 0]) / 2
        else:
            raise ValueError()

        all_data_points = np.concatenate([all_data_points, lowers, uppers])

        if background_function is not None:
           plot_kwargs['color'] = color

        ax_2d.errorbar(x, y, label=label, **plot_kwargs)
        if background_function is not None:
            ax_1d.errorbar(distance_values,(lowers + uppers)/2, yerr=abs(uppers - lowers)/2, label=label)

    print('Mean:', np.nanmean(all_data_points))
    print('Standard deviation:', np.nanstd(all_data_points))

    ax_2d.set_xlabel('Start distance (m)')
    ax_2d.set_ylabel('Goal distance (m)')
    ax_2d.set_xlim(left=0)
    ax_2d.set_ylim(bottom=0)
    ax_2d.legend(loc="upper right")

    #from matplotlib import rcParams
    #rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'

    plot_2d.set_size_inches((5, 3.5))
    plot_2d.tight_layout()
    plot_2d.savefig('results/local_controller_width/start_goal_dist.png')
    plot_2d.savefig('results/local_controller_width/start_goal_dist.pgf')
    plt.show()

if __name__ == "__main__":
    plot_figure(background_function)
