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

from typing import List, Iterator, Tuple

def get_path():
    """ returns path to data storage folder """
    dirname = os.path.join(os.path.dirname(__file__), "..")
    return dirname


# Print debug statements
debug = os.getenv('DEBUG', False)


def print_debug(*params):
    if debug:
        print(*params)


plotting = os.getenv('PLOTTING', False)


def display_trajectories(filepath):
    """ display all trajectories on one map, as well as a heatmap to check coverage """

    import matplotlib.pyplot as plt
    hf = h5py.File(filepath, 'r')
    env_model = hf.attrs['map_type']
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

Dimensions = tuple[float, float, float, float]

def random_location(dimensions: Dimensions, generator: np.random.RandomState) -> types.Vector2D:
    x = np.around(generator.uniform(dimensions[0], dimensions[1]), 1)
    y = np.around(generator.uniform(dimensions[2], dimensions[3]), 1)
    return [x, y]

def waypoint_movement(env: PybulletEnvironment, cam_freq, traj_length, map_layout: MapLayout, gc_network, seed: int) -> list[WaypointInfo]:
    ''' Calculates environment-specific waypoints from start to goal and creates
    trajectory by making agent follow them.
    
    arguments:
    env_model       -- environment name
    cam_freq        -- at what frequency is the state of the agent saved
    traj_length     -- how many timesteps should the agent run
    '''

    random_state = np.random.RandomState(seed=seed)
    dimensions: Dimensions = environment_dimensions(env.env_model)
    valid_locations: Iterator[types.Vector2D] = filter( # TODO Pierre: I'm sorry for the unreadable code
        map_layout.suitable_position_for_robot,
        iter(lambda: random_location(dimensions, random_state), None),
    )

    # initialize environment
    start = next(valid_locations)
    robot = Robot(env=env, base_position=start)
    with robot:

        samples: list['PlaceInfo'] = []

        while len(samples) < traj_length / cam_freq:
            goal = next(valid_locations)
            if (goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2 < 3:
                continue

            # calculate waypoints, if no path can be found return
            waypoints = map_layout.find_path(start, goal)
            if waypoints is None:
                print_debug("No path found!")
                continue

            from tqdm import tqdm

            from numbers import Number
            for g in tqdm(waypoints):
                assert isinstance(g[0], Number), g

                # if trajectory_length has been reached the trajectory can be saved
                if len(samples) > traj_length / cam_freq:
                    break

                compass = AnalyticalCompass(robot.position, g)
                env.add_debug_line(start=robot.position, end=g, color=(1, 0, 0), width=2)
                goal_reached, data = vector_navigation(env, compass, gc_network, step_limit=5000, plot_it=False,
                                            collect_data_freq=cam_freq)
                samples += data
                if not goal_reached:
                    print("Couldn't reach intermediate goal - breaking")
                    break
            if len(samples) > 0:
                start = samples[-1][0]

    if plotting:
        plot.plotTrajectoryInEnvironment(env=None, env_model=env.env_model, xy_coordinates=robot.data_collector.xy_coordinates)
    return samples


def generate_multiple_trajectories(out_hd5_obj, num_traj, trajectory_length, cam_freq, mapname: types.AllowedMapName):
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
    with PybulletEnvironment(mapname, dt, visualize=False, build_data_set=True, contains_robot=False) as env:
        for i in range(num_traj):
            traj_id = rng_trajid.randint(0xfffffff)
            dset_name = '/%08x' % traj_id

            print('processing trajectory %d id: %08x' % (i, traj_id))

            start_time = time.time()

            if dset_name in out_hd5_obj:
                print('dataset %s exists. skipped' % dset_name)
                continue

            samples = waypoint_movement(env, cam_freq, trajectory_length, map_layout, gc_network, seed=traj_id)
            # TODO: flatten each sample[:, 3]
            print(f"trajectory {samples[0]}-{samples[1]} with {len(samples)} steps")
            dset = out_hd5_obj.create_dataset(
                dset_name,
                data=np.array(samples, dtype=dtype),
                maxshape=(None,), dtype=dtype)

            out_hd5_obj.flush()

            print("--- %s seconds for one trajectory ---" % (time.time() - start_time))


def generate_and_save_trajectories(filepath, mapname, num_traj, traj_length, cam_freq):
    ''' Generate and save trajectories.
    
    arguments:
    
    filepath    -- filename for storing trajectories
    mapname     -- environment name
    num_traj    -- number of trajectories to be generated
    traj_length -- how many timesteps should the agent run
    cam_freq    -- at what frequency is the state of the agent saved
    '''
    f = h5py.File(filepath, 'a')
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath')
    parser.add_argument('--extension')
    parser.add_argument('-e', '--env-model', choices=types.AllowedMapName.options, default='Savinov_val3')
    parser.add_argument('-n', '--num-traj', type=int, default=1000)
    parser.add_argument('-l', '--traj-length', type=int, help='Length of one trajectory in timesteps', default=3000)
    parser.add_argument('--cam-freq', type=int, default=10)
    parser.add_argument('--no-display', help='Do not display trajectories', action='store_true')
    parser.add_argument('--only-display', help='Do not create trajectory file, only display existing trajectories', action='store_true')
    args = parser.parse_args()

    if args.filepath:
        assert not args.extension, "Specifying full file path makes extension redundant"
        filepath = args.filepath
    else:
        filename = "trajectories"
        if args.env_model != "Savinov_val3":
            filename = args.env_model + "." + filename
        extension = ".hd5" if args.extension is None else args.extension
        filepath = os.path.join(get_path(), "data", "trajectories", filename + extension)

    assert not (args.no_display and args.only_display)
    if not args.only_display:
        print("Trajectory generation in maze", args.env_model)
        generate_and_save_trajectories(filepath, args.env_model, args.num_traj, args.traj_length, args.cam_freq)
    if not args.no_display:
        display_trajectories(filepath)
