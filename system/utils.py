import numpy as np
from .types import Vector2D

def normalize(v : Vector2D) -> Vector2D:
    """ Normalize a vector so it has length 1 """
    v = np.array(v, dtype=float)
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    else:
        return np.array([0.0, 0.0])
