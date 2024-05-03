import math
import numpy as np
from . import parameters as p
from system.types import Angle

def headingCellsActivityTraining(heading : Angle) -> p.HeadingCellActivity:
    '''
    calculates HDC activity from current decoded heading direction
    :param heading: heading direction
    :return: activity vector of HDCs
    '''

    sig = 0.25 #0.1885  #higher value means less deviation and makes the boundray rep more accurate
    sig = p.nrHDC * sig / (2 * math.pi)
    amp = 1
    heading_vector = np.repeat(heading, p.nrHDC)

    tuning_vector = np.linspace(0, 2 * math.pi, p.nrHDC)     # nrHdc = 100  np.arange(0, 2 * math.pi, hdRes)
    # normal gaussian for hdc activity profile
    activity_vector = np.exp( ((heading_vector - tuning_vector) / 2 * (math.pow(sig, 2)) ** 2) ) * amp
    return np.around(activity_vector, 5)
