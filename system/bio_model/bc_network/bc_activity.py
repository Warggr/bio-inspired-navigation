''' Adapted from:
***************************************************************************************
*    Title: "Biologically Plausible Spatial Navigation Based on Border Cells"
*    Author: "Camillo Heye"
*    Date: 28.08.2021
*    Availability: https://drive.google.com/file/d/1RvmLd5Ee8wzNFMbqK-7jG427M8KlG4R0/view
*
***************************************************************************************
'''

from system.controller.simulation.pybullet_environment import types
from . import parameters as p
import math
import numpy as np
from typing import Optional

def radialDispersion(distance):    # linear function dependend of the distance a boundary object is encountered
    return (distance + 8) * 0.08

def bcCoordinates():
    polar_distances = np.array([ i*(i+1) for i in range(16) ]) * 0.05 + 1 # use very slowly increasing sizes of grid cells
    polar_angles = np.linspace(0, 2 * np.pi, p.nrDirections)
    grid_distances, grid_angles = np.meshgrid(polar_distances, polar_angles)
    grid_distances, grid_angles = grid_distances.flatten(), grid_angles.flatten() # vector of distances from each neuron to the origin
    grid_angles[grid_angles > np.pi] -= 2 * np.pi
    return grid_distances, grid_angles

grid_distances, grid_angles = bcCoordinates()
sigmaP = 0.2236
# Heye's thesis code was equivalent to:
# sigmaR = radialDispersion(distance)
# But I believe that's an error
sigmaR = radialDispersion(grid_distances)

# TODO: optimization: make all these functions accept a batch

def bcActivity(angle : types.Angle, distance : float, out : Optional[p.BoundaryCellActivity] = None) -> p.BoundaryCellActivity:
    if out is None:
        out = np.zeros_like(grid_distances)

    angDiff = abs(angle - grid_angles)
    angDiff = np.minimum(angDiff, -angDiff + 2*np.pi)

    # See Eqn. 4.1 in Heye's thesis
    out += 1 / distance * np.multiply(
        np.exp(-(angDiff / sigmaP) ** 2),
        np.exp(-((grid_distances - distance) / sigmaR) ** 2)
    )
    return out

def normalize(activity_vector: p.BoundaryCellActivity) -> p.BoundaryCellActivity:
    maximum = np.max(activity_vector)
    if maximum > 0.0:
        activity_vector = activity_vector / maximum
    return activity_vector

def bcActivityForLidar(lidar : types.LidarReading) -> p.BoundaryCellActivity:
    '''
    Calculates BC activity.

    :param thetaBndryPts: polar coordinate of boundary points representing angle
    :param rBndryPts: polar coordinate of boundary points representing distance
    :return: normalized activity vector containing each cell's activity
    '''

    activity_vector = None
    for angle, distance in zip(lidar.angles, lidar.distances):
        activity_vector = bcActivity(angle, distance, out=activity_vector)

    activity_vector = normalize(activity_vector)
    return activity_vector

# cartesian grid points that cover the region covered by the neurons
min_x = -p.maxR
max_x = p.maxR
min_y = -p.maxR
max_y = p.maxR
nr_x = round((max_x - min_x) / p.resolution)
nr_y = round((max_y - min_y) / p.resolution)
x_points = np.arange(min_x + p.resolution / 2, min_x + (nr_x - 0.5) * p.resolution, p.resolution)
y_points = np.arange(min_y + p.resolution / 2, min_y + (nr_y - 0.5) * p.resolution, p.resolution)
x, y = np.meshgrid(x_points, y_points)

def bcForWall(start : types.Vector2D, end : types.Vector2D) -> p.BoundaryCellActivity:
    '''
    :param boundary_location: start and endpoints of a certain boundary given as function parameters.
    :return: activity vector which contains each cell's activity
    '''
    # boundary start and endpoints
    x_start, y_start = start
    x_end, y_end = end

    boundary_length = np.linalg.norm(start-end)
    nr_x = (x_end - x_start) / boundary_length
    nr_y = (y_end - y_start) / boundary_length

    # perpendicular displacements
    perpDispFromGrdPts_x = -(x - x_start) * (1 - math.pow(nr_x, 2)) + (y - y_start) * nr_y * nr_x
    perpDispFromGrdPts_y = -(y - y_start) * (1 - math.pow(nr_y, 2)) + (x - x_start) * nr_x * nr_y

    if x_end != x_start:
        t = (x + perpDispFromGrdPts_x - x_start) / (x_end - x_start)
    else:
        t = (y + perpDispFromGrdPts_y - y_start) / (y_end - y_start)

    condition = (
        (t >= 0) & (t <= 1) &
        (perpDispFromGrdPts_x >= (-p.resolution / 2)) & (perpDispFromGrdPts_x < (p.resolution / 2)) &
        (perpDispFromGrdPts_y >= (-p.resolution / 2)) & (perpDispFromGrdPts_y < (p.resolution / 2))
    )

    xBndryPts = x[condition]
    yBndryPts = y[condition]

    # transform from cartesian to polar coordinates
    thetaBndryPts = np.arctan2(yBndryPts, xBndryPts)
    rBndryPts = np.sqrt(xBndryPts ** 2 + yBndryPts ** 2)

    activity_vector = np.zeros((816,))
    for distance, angle in zip(rBndryPts, thetaBndryPts):
        activity_vector = bcActivity(angle, distance, out=activity_vector)

    activity_vector = normalize(activity_vector)
    return activity_vector
