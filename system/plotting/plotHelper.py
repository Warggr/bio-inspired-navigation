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
from matplotlib.patches import Rectangle, Circle
import numpy as np
import os
from system.controller.simulation.environment_config import environment_dimensions
from system.types import AllowedMapName

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

def add_environment(ax, env_model: AllowedMapName, variant: str|None = None):
    maze_filename = 'maze_topview_binary'
    ax.axis('off')

    if env_model == "obstacle_map_0":
        distance = float(variant)-1 if variant is not None else 0

        # TODO: read these values from moveable_wall.urdf?
        box3 = Rectangle((-0.25+distance, -4.25-distance), 3.5, 5, color=TUM_colors['TUMDarkGray'])
        ax.add_artist(box3)
    elif env_model == "final_layout":
        # TODO: remove duplication with pybullet_environment
        if variant == "walls":
            variant = '11100'
        if variant is not None:
            maze_filename += '+empty'
    else:
        if variant is not None:
            maze_filename += f'+{variant}'

    # load topview of maze
    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname, "..")
    filename = os.path.join(dirname, "controller/simulation/environment", env_model, maze_filename+".png")
    filename = os.path.realpath(filename)

    if env_model == "final_layout" and variant is not None:
        unit_walls: list[tuple[Vector2D, Orientation]] = [
            ((-3, 2), 'vertical'),
            ((-3, 3), 'vertical'),
            ((-4, -2), 'vertical'),
            ((3, -5), 'vertical'),
            ((-7, 2), 'vertical'),
        ]
        # TODO: I could directly set unit_walls = [wall_rectangle(wall) for wall in unit_walls]
        def wall_rectangle(position, orientation):
            x0, y0 = position
            x1, y1 = position
            if orientation == 'horizontal':
                x1 += 1
                y0 -= 0.05; y1 += 0.05
            else:
                y1 += 1
                x0 -= 0.05; x1 += 0.05
            return x0, y0, x1, y1
        assert all(on in '01' for on in variant)
        for ((x, y), orientation), on in zip(unit_walls, variant):
            if on == '1':
                x0, y0, x1, y1 = wall_rectangle((x, y), orientation)
                wall = plt.Rectangle((x0, y0), x1-x0, y1-y0, color='#888888')
                ax.add_artist(wall)

    if env_model == "plane":
        pass
    else:
        topview = plt.imread(filename)
        dimensions = environment_dimensions(env_model)
        ax.imshow(topview, cmap="gray", extent=dimensions, origin="upper")


def environment_plot(env_model: AllowedMapName, variant: str|None = None):
    _, ax = plt.subplots()
    add_environment(ax, env_model, variant)
    return ax


def add_robot(ax, xy: tuple[float, float], angle: float, color=TUM_colors['TUMDarkGray']):
    circle = Circle((xy[0], xy[1]), 0.2, color=color, alpha=1)
    ax.add_artist(circle)

    ax.quiver(xy[0], xy[1], np.cos(angle) * 0.4, np.sin(angle) * 0.4, color=TUM_colors['TUMDarkGray'], headwidth=7,
              angles='xy', scale_units='xy', scale=1)


def add_goal(ax, goal: tuple[float, float]):
    circle = Circle(goal, 0.2, color=TUM_colors['TUMAccentOrange'], alpha=1)
    ax.add_artist(circle)
