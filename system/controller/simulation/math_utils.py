import numpy as np
from typing import Tuple

from system.types import Vector2D, Angle

def vectors_in_one_direction(v1 : Vector2D, v2 : Vector2D) -> bool:
    dot_product = np.dot(v1, v2)
    return dot_product >= 0

def intersect(p1 : Vector2D, v1 : Vector2D, p2 : Vector2D, v2 : Vector2D) -> bool:
    # Calculate determinant
    det = v1[0] * v2[1] - v1[1] * v2[0]

    # Parallel rays do not intersect
    if det == 0:
        return False

    # Calculate the relative position of intersection
    t = ((p2[0] - p1[0]) * v2[1] - (p2[1] - p1[1]) * v2[0]) / det
    u = ((p2[0] - p1[0]) * v1[1] - (p2[1] - p1[1]) * v1[0]) / det

    # Check if the intersection is in the direction of both rays
    return t >= 0 and u >= 0

def compute_angle(vec_1 : Vector2D, vec_2 : Vector2D) -> Angle:
    length_vector_1 = np.linalg.norm(vec_1)
    length_vector_2 = np.linalg.norm(vec_2)
    if length_vector_1 == 0 or length_vector_2 == 0:
        raise ValueError(f"Angle with the 0 vector is not defined: <{vec_1}, {vec_2}>")
    unit_vector_1 = vec_1 / length_vector_1
    unit_vector_2 = vec_2 / length_vector_2
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    dot_product = np.clip(dot_product, -1, 1) # it should be already, but due to numerical instabilities it might be e.g. -1.00002
    angle = np.arccos(dot_product)

    vec = np.cross([vec_1[0], vec_1[1], 0], [vec_2[0], vec_2[1], 0])

    return angle * np.sign(vec[2])
