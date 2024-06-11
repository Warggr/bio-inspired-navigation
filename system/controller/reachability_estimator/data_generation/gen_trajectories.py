''' This code has been adapted from:
***************************************************************************************
*    Title: "Scaling Local Control to Large Scale Topological Navigation"
*    Author: "Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox"
*    Date: 2020
*    Availability: https://github.com/xymeng/rmp_nav/tree/77a07393ccee77b0c94603642ed019268ce06640/rmp_nav/data_generation
*
***************************************************************************************
'''
import numpy as np
import time
import h5py

import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

from system.controller.simulation.pybullet_environment import PybulletEnvironment, Robot
from system.controller.simulation.environment_config import environment_dimensions
from system.controller.simulation.environment_cache import EnvironmentCache
from system.controller.local_controller.compass import AnalyticalCompass
from system.controller.local_controller.local_navigation import setup_gc_network, vector_navigation, WaypointInfo
from system.controller.simulation.environment.map_occupancy import MapLayout
import system.types as types

import system.plotting.plotResults as plot

from typing import List

def get_path():
    """ returns path to data storage folder """
    dirname = os.path.join(os.path.dirname(__file__), "..")
    return dirname


# Print debug statements
debug = False  # False


def print_debug(*params):
    if debug:
        print(*params)


plotting = True


def display_trajectories(filename, env_model):
    """ display all trajectories on one map, as well as a heatmap to check coverage """
    filename = filename + ".hd5"

    import matplotlib.pyplot as plt
    dirname = get_path()
    dirname = os.path.join(dirname, "data/trajectories")
    dirname = os.path.join(dirname, filename)
    filepath = os.path.realpath(dirname)
    hf = h5py.File(filepath, 'r')
    print("number of datasets: ", len(hf.keys()))

    # plot all trajectories in one map
    xy_coordinates = []
    for key in list(hf.keys()):
        ds_array = hf[key][()]
        coord = [i[0] for i in ds_array]
        xy_coordinates += coord
        print(key)

    #     # plot individual trajectories:
    #     # plot.plotTrajectoryInEnvironment(None,None,None,"title",xy_coordinates=coord,env_model="SMTP")
    # plot.plotTrajectoryInEnvironment(None,filename,xy_coordinates=xy_coordinates,env_model=env_model)

    # heatmap
    x = [i[0] for i in xy_coordinates]
    y = [i[1] for i in xy_coordinates]

    # get dimensions
    with PybulletEnvironment(env_model) as env:
        fig, ax = plt.subplots()
        hh = ax.hist2d(x, y, bins=[np.arange(env.dimensions[0], env.dimensions[1], 0.1)
            , np.arange(env.dimensions[2], env.dimensions[3], 0.1)], norm="symlog")
        fig.colorbar(hh[3], ax=ax)
        plt.show()

def valid_location(env_map : MapLayout):
    """ Sample valid location for agent in the environment """
    dimensions = environment_dimensions(env_map.name)
    while True:
        x = np.around(np.random.uniform(dimensions[0], dimensions[1]), 1)
        y = np.around(np.random.uniform(dimensions[2], dimensions[3]), 1)

        if env_map.suitable_position_for_robot([x, y]):
            return [x, y]

def gen_multiple_goals(env_map : MapLayout, nr_of_goals):
    ''' Generate start and multiple subgoals'''

    points = []
    for i in range(nr_of_goals):
        points.append(valid_location(env_map))

    start = valid_location(env_map)
    points.insert(0, start)
    return points


def waypoint_movement(env : PybulletEnvironment, cam_freq, traj_length, map_layout : MapLayout, gc_network) -> List[WaypointInfo]:
    ''' Calculates environment-specific waypoints from start to goal and creates
    trajectory by making agent follow them.
    
    arguments:
    env_model       -- environment name
    cam_freq        -- at what frequency is the state of the agent saved
    traj_length     -- how many timesteps should the agent run
    '''

    # initialize environment
    start = valid_location(map_layout)
    with Robot(env=env, base_position=start) as robot:

        samples : List['PlaceInfo'] = []

        while len(samples) < traj_length / cam_freq:
            goal = valid_location(map_layout)
            if (goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2 < 3:
                continue

            # calculate waypoints, if no path can be found return
            waypoints = map_layout.find_path(start, goal)
            print("---\nStart:", start)
            for waypoint in waypoints: print(waypoint)
            print("Goal:", goal, "\n---")
            if waypoints is None:
                print_debug("No path found!")
                continue

            from numbers import Number
            for g in waypoints:
                assert isinstance(g[0], Number), g

                # if trajectory_length has been reached the trajectory can be saved
                if len(samples) > traj_length / cam_freq:
                    break

                compass = AnalyticalCompass(robot.position, g)
                print(f"Vector navigation from {robot.position} to {g}")
                env.add_debug_line(start=robot.position, end=g, color=(1, 0, 0), width=2)
                goal_reached, data = vector_navigation(env, compass, gc_network, step_limit=5000, plot_it=False,
                                            obstacles=False, collect_data_freq=cam_freq)
                samples += data
                if not goal_reached:
                    print("Couldn't reach intermediate goal - breaking")
                    break
            if len(samples) > 0:
                start = samples[-1][0]

    if plotting:
        plot.plotTrajectoryInEnvironment(env)
    return samples


def generate_multiple_trajectories(out_hd5_obj, num_traj, trajectory_length, cam_freq, mapname : types.AllowedMapName):
    ''' Generate multiple trajectories
    
    arguments:
    out_hd5_obj         -- output file
    num_traj            -- number of trajectories that should be generated for the file
    trajectory_length   -- number of timesteps in generated trajectory
    cam_freq            -- frequency with which the agent state is saved
    mapname             -- name of environment
    '''
    dtype = np.dtype([
        ('xy_coordinates', (np.float32, 2)),
        ('orientation', np.float32),
        ('grid_cell_spiking', (np.float32, 9600)),
    ])

    seed = 123457
    rng_trajid = np.random.RandomState(seed)

    dt = 1e-2
    gc_network = setup_gc_network(dt)
    map_layout = MapLayout(mapname)

    i = 0
    while i < num_traj:
        traj_id = rng_trajid.randint(0xfffffff)
        dset_name = '/%08x' % traj_id

        print('processing trajectory %d id: %08x' % (i, traj_id))

        start_time = time.time()

        if dset_name in out_hd5_obj:
            print('dataset %s exists. skipped' % dset_name)
            continue

        samples = waypoint_movement(mapname, cam_freq, trajectory_length, map_layout, gc_network)
        # TODO: flatten each sample[:, 3]
        print(f"trajectory {samples[0]}-{samples[1]} with {len(samples)} steps")
        dset = out_hd5_obj.create_dataset(
            dset_name,
            data=np.array(samples, dtype=dtype),
            maxshape=(None,), dtype=dtype)

        out_hd5_obj.flush()
        i += 1

        print("--- %s seconds for one trajectory ---" % (time.time() - start_time))


def generate_and_save_trajectories(filename, mapname, num_traj, traj_length, cam_freq):
    ''' Generate and save trajectories.
    
    arguments:
    
    filename    -- filename for storing trajectories
    mapname     -- environment name
    num_traj    -- number of trajectories to be generated
    traj_length -- how many timesteps should the agent run
    cam_freq    -- at what frequency is the state of the agent saved
    '''
    dirname = get_path()
    directory = os.path.join(dirname, "data/trajectories")
    directory = os.path.realpath(directory)

    f = h5py.File(directory + "/" + filename + ".hd5", 'a')
    f.attrs.create('agent', "waypoint")
    f.attrs.create('map_type', mapname)

    generate_multiple_trajectories(f, num_traj, traj_length, cam_freq, mapname)


if __name__ == "__main__":
    """ 
    Testing:
    Generate/ load a few trajectories per map and display.
    
    Default:
    Generate 1000 trajectories of length 3000 with a saving frequency of 10 
    in the environment "Savinov_val3"
    
    Parameterized:
    Adjust filename, env_model, num_traj, traj_length and cam_freq 
    """
    test = True
    if len(sys.argv) == 6:
        _, filename, env_model, num_traj, trajectory_length, cam_freq = sys.argv
        print_debug(sys.argv)
        generate_and_save_trajectories(filename, str(env_model), int(num_traj), int(trajectory_length), int(cam_freq))
    elif test:
        print("Testing trajectory generation in available mazes.")
        print("Testing Savinov_val3")
        generate_and_save_trajectories("trajectories", "Savinov_val3", num_traj=1000, traj_length=3000, cam_freq=10)
        display_trajectories("trajectories", "Savinov_val3")
        # print("Testing Savinov_val2")
        # save_trajectories("test_2", "Savinov_val2", 1, 3000, 10)
        # display_trajectories("test_2", "Savinov_val2")
        # print("Testing Savinov_test7")
        # save_trajectories("test_3", "Savinov_test7", 1, 3000, 10)
        # display_trajectories("test_3", "Savinov_test7")
    else:
        num_traj = 1000
        trajectory_length = 3000
        cam_freq = 10
        env_model = "Savinov_val3"  # "Savinov_val2","Savinov_test7"

        display_trajectories("trajectories", env_model)
