import math
import numpy as np
from .types import Vector2D, Angle


def normalize(v: Vector2D) -> Vector2D:
    """ Normalize a vector so it has length 1 """
    v = np.array(v, dtype=float)
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    else:
        return np.array([0.0, 0.0])

def angle_abs_difference(a: Angle, b: Angle):
    """
    vectorizable
    """
    diff = np.abs(a - b) % (2*math.pi)
    diff = np.minimum(diff, 2*math.pi - diff)
    if type(a) == np.ndarray:
        return diff
    else:
        return diff.item()

def average(li):
    return sum(li) / len(li)
