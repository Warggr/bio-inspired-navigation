# coding: utf-8
from system.controller.simulation.environment.map_occupancy import random_points
import random
import math

rng = random.Random()

def format(p: 'PositionAndOrientation') -> str:
    return ','.join(map(str, [p[0][0], p[0][1], p[1]]))

for _ in range(100):
    p0, p1 = random_points(env_model="Savinov_val3", rng=rng)
    a0, a1 = [2*math.pi*rng.random() for _ in range(2)]
    print("Savinov_val3", format((p0, a0)), format((p1, a1)))
