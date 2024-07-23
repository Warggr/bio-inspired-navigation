""" This code has been adapted from:
***************************************************************************************
*    Title: "Neurobiologically Inspired Navigation for Artificial Agents"
*    Author: "Johanna Latzel"
*    Date: 12.03.2024
*    Availability: https://nextcloud.in.tum.de/index.php/s/6wHp327bLZcmXmR
*
***************************************************************************************
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple
from system.controller.simulation.environment_config import environment_dimensions

TUM_colors = {
                'TUMBlue': '#0065BD',
                'TUMSecondaryBlue': '#005293',
                'TUMSecondaryBlue2': '#003359',
                'TUMBlack': '#000000',
                'TUMWhite': '#FFFFFF',
                'TUMDarkGray': '#333333',
                'TUMGray': '#808080',
                'TUMLightGray': '#CCCCC6',
                'TUMAccentGray': '#DAD7CB',
                'TUMAccentOrange': '#E37222',
                'TUMAccentGreen': '#A2AD00',
                'TUMAccentLightBlue': '#98C6EA',
                'TUMAccentBlue': '#64A0C8'
}

def add_environment(ax, env_model: str):

    #load topview of maze
    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname, "..")
    filename = os.path.join(dirname, "controller/simulation/environment/"+env_model+"/maze_topview_binary.png")
    filename = os.path.realpath(filename)

    if not env_model == "plane" and not "obstacle" in env_model:
        topview = plt.imread(filename)
        dimensions = environment_dimensions(env_model)
        ax.imshow(topview,cmap="gray",extent=dimensions,origin="upper")
    elif env_model == "obstacle_map_2":
        box1 = plt.Rectangle((-1.75, -1.5), 0.5, 3, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box1)
        box2 = plt.Rectangle((-2.5, -0.75), 1, 0.5, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box2)
        wall = plt.Rectangle((-4, -3), 0.1, 6, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((2, -3), 0.1, 6, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall) 
        wall = plt.Rectangle((-4, 3), 6, 0.1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((-4, -3), 6, 0.1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall) 
        ax.set_xlim(-4, 2.1)
        ax.set_ylim(-3, 3.1)
    elif env_model == "obstacle_map_3":
        box1 = plt.Rectangle((-0.75, -2.25), 0.5, 2.5, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box1)
        box2 = plt.Rectangle((-2.5, -0.75), 3, 0.5, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box2)
        wall = plt.Rectangle((-4, -3), 0.1, 6, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((2, -3), 0.1, 6, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall) 
        wall = plt.Rectangle((-4, 3), 6, 0.1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((-4, -3), 6, 0.1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        ax.set_xlim(-4, 2.1)
        ax.set_ylim(-3, 3.1)
    elif env_model == "obstacle_map_1":
        box1 = plt.Rectangle((-1.75, -1.5), 0.5, 3, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box1)
        box2 = plt.Rectangle((-2.5, -0.75), 1, 0.5, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box2)
        box3 = plt.Rectangle((-0.25, -2), 0.5, 3, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box3)        
        wall = plt.Rectangle((-4, -3), 0.1, 6, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((2, -3), 0.1, 6, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall) 
        wall = plt.Rectangle((-4, 3), 6.1, 0.1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((-4, -3), 6, 0.1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)   
        ax.set_xlim(-4, 2.1)
        ax.set_ylim(-3, 3.1)
    elif env_model == "obstacle_map_0":
        box1 = plt.Rectangle((-1.75, -1.5), 0.5, 4, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box1)
        box2 = plt.Rectangle((-2.5, -0.75), 1, 0.5, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box2)
        box3 = plt.Rectangle((-0.25, -2), 0.5, 3, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box3)
        box4 = plt.Rectangle((0.25, 0), 1.5, 1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box4)
        box5 = plt.Rectangle((-1.25, 1.75), 2.5, 0.5, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box5)

        wall = plt.Rectangle((-4, -3), 0.1, 6, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((2, -3), 0.1, 6, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((-4, 3), 6.1, 0.1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        wall = plt.Rectangle((-4, -3), 6, 0.1, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(wall)
        ax.set_xlim(-4, 2.1)
        ax.set_ylim(-3, 3.1)
    else:
        ax.set_xlim(-9, 6)
        ax.set_ylim(-5, 4)


def add_robot(ax, xy: tuple[float, float], angle: float):
    circle = plt.Circle((xy[0], xy[1]), 0.2, color=TUM_colors['TUMDarkGray'], alpha=1)
    ax.add_artist(circle)

    ax.quiver(xy[0], xy[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4, color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)

def add_goal(ax, goal: tuple[float, float]):
    circle = plt.Circle(goal, 0.2, color=TUM_colors['TUMAccentOrange'], alpha=1)
    ax.add_artist(circle)
