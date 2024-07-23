import math
import numpy as np
from typing import Tuple, Any, Self, Optional
import sys, os
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from system.types import Angle, Vector2D
import system.bio_model.bc_network.parameters as p
import system.bio_model.bc_network.HDCActivity as HDCActivity
from system.bio_model.bc_network.bc_activity import bcForWall
from system.bio_model.utils import DATA_PATH

DEFAULT_FILENAME = os.path.join(DATA_PATH, "bc_model", "transformations.npz")

class BoundaryCellNetwork:
    def __init__(self,
        ego2trans: np.ndarray,
        heading2trans: np.ndarray,
        trans2BVC: np.ndarray,
    ):
        self.ego2trans = ego2trans
        self.heading2trans = heading2trans
        self.trans2BVC = trans2BVC

        # clipping small weights makes activities sharper
        #ego2trans = np.where(ego2trans >= np.max(ego2trans * 0.3), ego2trans, 0)

        # rescaling as in BB-Model
        #ego2trans = ego2trans * 50
        #trans2BVC = trans2BVC * 35
        #heading2trans = heading2trans * 15

    @staticmethod
    def load(filename: str = DEFAULT_FILENAME) -> Self:
        arrays = np.load(filename)
        return BoundaryCellNetwork(**arrays)

    def save(self, filename: str = DEFAULT_FILENAME):
        np.savez(filename, ego2trans=self.ego2trans, heading2trans=self.heading2trans, trans2BVC=self.trans2BVC)

    def calculateActivities(self, egocentricActivity: p.BoundaryCellActivity, heading: p.HeadingCellActivity) -> tuple[Any, p.BoundaryCellActivity]:
        '''
        calculates activity of all transformation layers and the BVC layer by multiplying with respective weight tensors
        :param egocentricActivity: egocentric activity which was previously calculated in @BCActivity
        :param heading: HDC networks activity
        :return: activity of all TR layers and BVC layer
        '''
        assert type(heading) == np.ndarray, f"Expected array, got {type(heading)}"

        egocentricActivity = egocentricActivity.flatten()
        transformationLayers = np.einsum('i,ijk -> jk', egocentricActivity, self.ego2trans)

        maxTRLayers = np.amax(transformationLayers)
        transformationLayers = transformationLayers / maxTRLayers
        headingIntermediate = np.einsum('i,jik -> jk ', heading, self.heading2trans)
        headingScaler = headingIntermediate[0, :]
        scaledTransformationLayers = np.ones((816, 20))
        for i in range(20):
            scaledTransformationLayers[:, i] = transformationLayers[:, i] * headingScaler[i]
        bvcActivity = np.sum(scaledTransformationLayers, 1)
        assert not np.isnan(np.max(bvcActivity))
        maxBVC = np.amax(bvcActivity)
        if maxBVC > 0:
            bvcActivity = bvcActivity/maxBVC
        assert not np.isnan(np.min(bvcActivity))

        return transformationLayers, bvcActivity

random = np.random.default_rng()

def random_angle(n=None) -> Angle:
    return 2 * math.pi * random.random(n)

def to_cartesian(angle: Angle, distance: float) -> Vector2D:
    x = distance * np.cos(angle)  # convert to cartesian x,y in allocentric reference frame
    y = distance * np.sin(angle)
    return np.array([x, y])

def random_segment(maxSize = p.maxR, n=None) -> tuple[Vector2D, Vector2D]:
    angle_start = random_angle(n)
    distance = p.maxR * random.random(n)

    angle_end = random_angle(n)

    return to_cartesian(angle_start, distance), to_cartesian(angle_end, angle_end)

def transformation_matrix(heading: Angle):
    return np.array([
        [np.cos(heading), np.sin(heading)],
        [-np.sin(heading), np.cos(heading)],
    ])

def train(
    nrSteps = 10000,
) -> BoundaryCellNetwork:
    ego2TransformationWts = np.zeros((p.nrBVC, p.nrBVC, p.nrTransformationLayers,))
    transformation2BVCWts = np.eye(p.nrBVC, p.nrTransformationLayers)
    heading2TransformationWts = np.zeros((p.nrBVC, p.nrHDC, p.nrTransformationLayers))

    for count in tqdm(range(nrSteps)):
        start, end = random_segment()
        BVCrate = bcForWall(start, end)

        transformationLayer = random.integers(0, len(p.transformationAngles))
        heading = p.transformationAngles[transformationLayer]
        # create egocentric point of view from the allocentric point of view by rotating the edges
        rotation = transformation_matrix(heading)
        rstart, rend = start @ rotation, end @ rotation

        egocentricRate = bcForWall(rstart, rend)
        ego2TransformationWts[:, :, transformationLayer] += np.outer(egocentricRate, np.transpose(BVCrate))

    assert not np.isnan(np.sum(ego2TransformationWts))

    for index in range(p.nrTransformationLayers):
        heading = p.transformationAngles[index]
        HDCrate = HDCActivity.headingCellsActivityTraining(heading)
        HDCrate = np.where(HDCrate < 0.01, 0, HDCrate)
        # HDCrate = sparse(HDCrate)     for better computational costs
        heading2TransformationWts[:, :, index] = np.outer(np.ones(p.nrBVC), HDCrate)

    assert not np.isnan(np.sum(heading2TransformationWts))

    #rescaling
    divtmp = np.zeros((p.nrBVC, p.nrBVC))
    for index in range(20):
        ego2TransformationWts[:, :, index] = ego2TransformationWts[:, :, index] / np.amax(ego2TransformationWts[:, :, index])
        divtmp = np.outer(np.sum(ego2TransformationWts[:, :, index], 1), np.ones(p.nrBVC))
        ego2TransformationWts[:, :, index] = np.divide(ego2TransformationWts[:, :, index], divtmp)

    assert not np.isnan(np.sum(ego2TransformationWts))

    return BoundaryCellNetwork(ego2TransformationWts, heading2TransformationWts, transformation2BVCWts)

if __name__ == "__main__":
    import sys
    kwargs = {}
    if len(sys.argv) > 1:
       kwargs['nrSteps'] = int(sys.argv[1])
    network = train(**kwargs)
    network.save()
