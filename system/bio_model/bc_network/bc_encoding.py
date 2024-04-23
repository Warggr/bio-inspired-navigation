''' Adapted from:
***************************************************************************************
*    Title: "Biologically Plausible Spatial Navigation Based on Border Cells"
*    Author: "Camillo Heye"
*    Date: 28.08.2021
*    Availability: https://drive.google.com/file/d/1RvmLd5Ee8wzNFMbqK-7jG427M8KlG4R0/view
*
***************************************************************************************
'''

import numpy as np
from system.controller.simulation.pybullet_environment import types
from .parameters import angularDispersion

BoundaryCellActivity = np.ndarray

def radialDispersion(distance):    # linear function dependend of the distance a boundary object is encountered
    return (distance + 8) * 0.08

def boundaryCellCoordinates():
    polar_distances = np.array([ i*(i+1) for i in range(16) ]) * 0.05 + 1 # use very slowly increasing sizes of grid cells
    polar_angles = np.linspace(0, 2 * math.pi, 51) # - parametersBC.polarAngularResolution
    grid_distance, grid_angle = np.meshgrid(polar_distances, polar_angles)
    grid_distance, grid_angle = grid_distance.flatten(), grid_angle.flatten() # vector of distances from each neuron to the origin
    grid_angles[grid_angles > math.pi] -= 2 * math.pi
    return grid_distance, grid_angle

def boundaryCellActivitySimulation(lidar : types.LidarReading) -> BoundaryCellActivity:
    '''
    Calculates BC activity.

    :param thetaBndryPts: polar coordinate of boundary points representing angle
    :param rBndryPts: polar coordinate of boundary points representing distance
    :return: normalized activity vector containing each cell's activity
    '''
    distances, angles = boundaryCellCoordinates()
    activity_vector = np.zeros_like(distances)

    for angle, distance in zip(lidar.angles, lidar.distances):
        angDiff = abs(grid_angle - theta)
        angDiff = np.min( angDiff, 2 * math.pi - angDiff )

        sigmaP = 0.2236
        # Heye's thesis code was equivalent to:
        # sigmaR = radialDispersion(distance)
        # But I believe that's an error
        sigmaR = radialDispersion(grid_distance)

        # See Eqn. 4.1 in Heye's thesis
        activity_vector += 1 / distance * np.multiply(
            np.exp(-(angDiff / sigmaP) ** 2),
            np.exp(-((grid_distance - r) / sigmaR) ** 2)
        )
    maximum = max(activity_vector)
    if maximum > 0.0:
        activity_vector = activity_vector / maximum
    return activity_vector
