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

import numpy as np
from matplotlib import pyplot as plt

from system.controller.simulation.environment.map_occupancy import MapLayout
from system.plotting.plotResults import plotStartGoalPair
from system.controller.reachability_estimator.types import PlaceInfo, ReachabilityController

from typing import Optional, override

def in_fov(src_position, src_heading, src_fov, dst_position):
    # Calculate the vector from the agent to the target
    target_vector = (dst_position[0] - src_position[0], dst_position[1] - src_position[1])

    # Calculate the angle between the agent's heading and the target vector
    angle_to_target = math.atan2(target_vector[1], target_vector[0]) - src_heading
    angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi

    # Check if the angle to the target is within the field of view
    return abs(angle_to_target) <= src_fov / 2


class ViewOverlapReachabilityController(ReachabilityController):
    def __init__(self):
        self.fov = 120 * np.pi / 180
        self.L_min = 0.5
        self.R_max = 1
        self.E_max_squared = 0.8
        self.theta_max = self.fov / 2

    def compute_overlap(self, map_name, pos1, heading1, pos2, heading2):
        map_layout = MapLayout(map_name)

        overlap_ratios = map_layout.view_overlap(pos1, heading1,
                                                 pos2, heading2,
                                                 self.fov, mode='plane')
        return overlap_ratios

    @override
    def reachable(self, env : 'PybulletEnvironment', src: PlaceInfo, dst: PlaceInfo, path_l : Optional[float] = None) -> bool:
        visual_overlap = min(self.compute_overlap(env.env_model,
                                                  src.pos, src.angle,
                                                  dst.pos, dst.angle))
        dst_in_fov = in_fov(src.pos, src.angle, self.fov, dst.pos)
        distance_squared = (src.pos[0] - dst.pos[0]) ** 2 + (src.pos[1] - dst.pos[1]) ** 2
        delta_theta = dst.angle - src.angle
        if delta_theta >= math.pi:
            delta_theta -= math.pi * 2

        # print(f"vis overlap {visual_overlap}, path {path_l}, in fov {dst_in_fov}, eu dist {math.sqrt(distance_squared)}, ∆Theta {delta_theta * 180 / math.pi}")
        return visual_overlap >= self.L_min and \
            (path_l or 0) <= self.R_max and \
            dst_in_fov and \
            distance_squared < self.E_max_squared and \
            abs(delta_theta) < self.theta_max


def display_sample(env_model, src_sample, dst_sample, src_image, dst_image):
    """ Display information about dataset file

    if imageplot: plot the stored images
    if startgoalplot: plot first 5000 start,goal pairs on a map of the environment
    Calculate the percentage of reached/failed samples.
    """
    if src_image is not None:
        plt.imshow(np.reshape(src_image, (64, 64, 4)))
        plt.show()
    if dst_image is not None:
        plt.imshow(np.reshape(dst_image, (64, 64, 4)))
        plt.show()

    plotStartGoalPair(env_model, src_sample[0], src_sample[1], dst_sample[0], dst_sample[1])