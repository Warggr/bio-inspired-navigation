# coding: utf-8

import sys
map_name: str
print(sys.argv)

match sys.argv:
    case (_, map_name):
        nb_points = 1
    case (_, map_name, nb_points):
        nb_points = int(nb_points)
        pass
    case (_,):
        print("Error: map_name is required")
        sys.exit(1)

import matplotlib.pyplot as plt
import system.plotting.plotResults as plot
plot.plotTrajectoryInEnvironment(env_model=map_name, trajectory=False, show=False)
for i in range(nb_points):
    print(plt.ginput())
