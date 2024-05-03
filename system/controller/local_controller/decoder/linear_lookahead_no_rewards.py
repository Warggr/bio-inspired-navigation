''' This code has been adapted from:
***************************************************************************************
*    Title: "Biologically inspired spatial navigation using vector-based and topology-based path planning"
*    Author: "Tim Engelmann"
*    Date: 28.09.2021
*    Code version: 1.0
*    Availability: https://drive.google.com/file/d/1g7I-n9KVVulybh1YeElSC-fvm9_XDoez/view
*
***************************************************************************************
'''
import numpy as np

from system.bio_model.grid_cell_model import GridCellNetwork

active_threshold = 0.85

def perform_look_ahead_2xnr(gc_network: GridCellNetwork, robot : 'Robot'):
    """Performs a linear lookahead to find an offset in grid cell spiking in either x or y direction."""
    gc_network.reset_s_virtual()  # Resets virtual gc spiking to actual spiking

    dt = gc_network.dt * 10  # checks spiking only every nth step
    speed = 0.5  # match actual speed
    xy_speeds = np.array(([1, 0], [-1, 0], [0, 1], [0, -1])) * speed # define the four look-ahead velocity vectors
    goal_spiking = {}  # "axis": {"reward_value", "idx_place_cell", "distance", "step"}

    max_distance = 1.1 * robot.env.arena_size  # after this distance lookahead is aborted

    max_nr_steps = int(max_distance / (speed * dt))

    for idx, xy_speed in enumerate(xy_speeds):
        axis = int(idx / 2)  # either x or y
        reward_array = []  # save rewards during lookahead, to create lookahead video
        
        for i in range(max_nr_steps):

            # Compute reward firing
            # Only look for one specific place cell (goal location)
            s_vectors = gc_network.consolidate_gc_spiking(virtual=True)

            # computes projected pc firing
            #TODO Johanna: this can be directly connected to the place cell network but we wanted to make the local controller independent
            #firing = pc_network.place_cells[goal_pc_idx].compute_firing_2x(s_vectors, axis)
            firing = compute_firing_2x(gc_network.target_spiking,s_vectors, axis)

            # make sure that firing is strong enough
            reward = firing if firing > active_threshold else 0
            idx_place_cell = None
                
            reward_array.append(reward)

            distance = xy_speed[axis] * i * dt  # lookahead distance
            if axis not in goal_spiking or reward - goal_spiking[axis]["reward"] > 0:
                # First entrance or exceeds previous found value
                goal_spiking[axis] = {"reward": reward, "idx_place_cell": idx_place_cell,
                                      "distance": distance, "step": i}

            # Abort conditions to end lookahead earlier
            if axis in goal_spiking and i > 50 and reward < 0.85 * goal_spiking[axis]["reward"]\
                    and goal_spiking[axis]["reward"] > 0.9: 
                # To make sure that looks sufficiently in all 4 directions add this above
                # and np.sign(goal_spiking[axis]["distance"]) == np.sign(distance) \
                #print("Aborting")
                break

            gc_network.track_movement(xy_speed, virtual=True, dt_alternative=dt)  # track virtual movement

            # if i % 20 == 0:
            #     print_str = "Lookahead progress| Direction " + str(idx) + "/4 " \
            #                 "| time-step " + str(i) + "/" + str(max_nr_steps) + \
            #                 "| total " + str(int(100 * (idx * max_nr_steps + i) / (4 * max_nr_steps))) + "%"
            #     print(print_str)

        gc_network.reset_s_virtual()  # reset after lookahead in a direction

    goal_vector = np.array([axis["distance"] for axis in goal_spiking.values()])  # consolidate goal vector from dict

    print("------ Goal localization at time-step: ", len(robot.data_collector) - 1)
    if len(goal_vector) != 2:
        # Something went wrong and no goal vector was found
        goal_vector = np.random.rand(2) * 0.5
        print("Unable to find a goal_vector", goal_spiking)
        raise ValueError("No goal vector found.")
    else:
        print("Found goal vector", goal_vector, goal_spiking)

    return goal_vector

#TODO Johanna: this can be directly connected to the place cell network but we wanted to make the local controller independent
def compute_firing_2x(gc_connections, s_vectors, axis, plot=False):
    """Computes firing projected on one axis, based on current grid cell spiking"""
    new_dim = int(np.sqrt(len(s_vectors[0])))  # n

    s_vectors = np.where(s_vectors > 0.1, 1, 0)  # mute weak grid cell spiking, transform to binary vector
    gc_connections = np.where(gc_connections > 0.1, 1, 0)  # mute weak connections, transform to binary vector

    proj_s_vectors = np.empty((len(s_vectors[:, 0]), new_dim))
    for i, s in enumerate(s_vectors):
        s = np.reshape(s, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
        proj_s_vectors[i] = np.sum(s, axis=axis)  # sum over column/row

    proj_gc_connections = np.empty_like(proj_s_vectors)
    for i, gc_vector in enumerate(gc_connections):
        gc_vector = np.reshape(gc_vector, (new_dim, new_dim))  # reshape (n^2 x 1) vector to n x n vector
        proj_gc_connections[i] = np.sum(gc_vector, axis=axis)  # sum over column/row

    filtered = np.multiply(proj_gc_connections, proj_s_vectors)  # filter projected firing, by projected connections

    norm = np.sum(np.multiply(proj_s_vectors, proj_s_vectors), axis=1)  # compute unnormed firing at optimal case

    firing = 0
    modules_firing = 0
    for idx, filtered_vector in enumerate(filtered):
        # We have to distinguish between modules tuned for x direction and modules tuned for y direction
        if np.amin(filtered_vector) == 0:
            # If tuned for right direction there will be clearly distinguishable spikes
            firing = firing + np.sum(filtered_vector) / norm[idx]  # normalize firing and add to firing
            modules_firing = modules_firing + 1

    firing = firing/modules_firing  # divide by modules that we considered to get overall firing

    return firing