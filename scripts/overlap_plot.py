import matplotlib.pyplot as plt
import numpy as np
from system.controller.reachability_estimator.reachability_estimation import ViewOverlapReachabilityEstimator
from system.controller.reachability_estimator._types import PlaceInfo

env_model = 'obstacle_map_0'
env_variant = '1.0'
re = ViewOverlapReachabilityEstimator(env_model)

distance_values = [i / 8 for i in range(8*2 + 1)]

def get_overlap(start_distance, goal_distance):
    corner = (-0.75, 1.25)
    start = (corner[0], corner[1] - start_distance)
    goal = (corner[0] + goal_distance, corner[1])
    start_heading = np.pi / 2 # up
    goal_heading = 0 # left
    start = PlaceInfo(start, start_heading, None, None, None)
    goal = PlaceInfo(goal, goal_heading, None, None, None)
    return re.reachability_factor(start, goal)


x, y = np.meshgrid(distance_values, distance_values)
x, y = x.flatten(), y.flatten()
ratios = [get_overlap(xi, yi) for xi, yi in zip(x, y)]
distances = plt.scatter(x, y, c=ratios)

line_start = 'Maximum handle-able goal_distance is '
starts, stops = [], []
for x in distance_values:
    filename = f"system/tests/results/local_controller_width/adist={x}.log"
    with open(filename, 'r') as f:
        line = f.read().strip()
    assert line.startswith(line_start), line; line = line.removeprefix(line_start)
    start, stop = line.split(' ~ ')
    start, stop = [float(i) for i in (start, stop)]
    print(f'Between {get_overlap(x, start)} and {get_overlap(x, stop)}')
    starts.append((x, start))
    stops.append((x, stop))

line_start = 'Maximum handle-able start_distance is '
for x in distance_values:
    filename = f"system/tests/results/local_controller_width/zdist={x}.log"
    with open(filename, 'r') as f:
        line = f.read().strip()
    assert line.startswith(line_start), line; line = line.removeprefix(line_start)
    start, stop = line.split(' ~ ')
    start, stop = [float(i) for i in (start, stop)]
    print(f'Between {get_overlap(start, x)} and {get_overlap(stop, x)}')
    starts.append((start, x))
    stops.append((stop, x))

plt.scatter([s[0] for s in starts], [s[1] for s in starts], c='g')
plt.scatter([s[0] for s in stops], [s[1] for s in stops], c='r')

plt.show()
