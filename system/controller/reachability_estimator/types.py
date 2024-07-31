import system.types as types
from system.types import LidarReading
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from system.polyfill import Self

def img_reshape(img_array) -> types.Image:
    """ image stored in array form to image in correct shape for nn """
    img = np.reshape(img_array, (64, 64, 4))

    return img

@dataclass
class PlaceInfo:
    """
    All the info that can be extracted about a certain place.
    """

    pos: types.Vector2D
    angle: types.Angle
    spikings: types.Spikings
    img: types.Image
    lidar: types.LidarReading

    def __repr__(self):
        return f"PlaceInfo(pos={self.pos}, angle={self.angle=}, { 'lidar=...' if hasattr(self, 'lidar') else 'no lidar' })"

@dataclass
class Sample:
    """ One sample containing two places. This can be taken as an input to evaluate reachability of both places. """
    src: PlaceInfo
    dst: PlaceInfo

    def to_tuple(self, reachable: bool) -> tuple:
        """ Returns a tuple which can be put into a Numpy array of type Sample.dtype """
        return (
            self.src.img.flatten(), self.dst.img.flatten(),
            reachable,
            self.src.pos, self.dst.pos,
            self.src.angle, self.dst.angle,
            self.src.spikings.flatten(), self.dst.spikings.flatten(),
            self.src.lidar.distances, self.dst.lidar.distances, # assuming default angle config
        )

    @staticmethod
    def from_tuple(tup: tuple) -> tuple[Self, bool]:
        (
            src_img, dst_img,
            reachable,
            src_pos, dst_pos,
            src_angle, dst_angle,
            src_spikings, dst_spikings,
            src_lidar, dst_lidar,
        ) = tup
        return Sample(
            PlaceInfo(
                src_pos, src_angle, src_spikings, img_reshape(src_img), LidarReading(src_lidar, LidarReading.angle_range(src_angle))
            ),
            PlaceInfo(
                dst_pos, dst_angle, dst_spikings, img_reshape(dst_img), LidarReading(dst_lidar, LidarReading.angle_range(dst_angle))
            ),
        ), reachable

    dtype = np.dtype([
            ('start_observation', (np.int32, 64*64*4)),
            ('goal_observation', (np.int32, 64*64*4)), # using (64, 64, 4) would be more elegant but H5py doesn't accept it
            ('reached', bool),
            ('start', (np.float32, 2)),  # x, y
            ('goal', (np.float32, 2)),  # x, y
            ('start_orientation', np.float32),  # theta
            ('goal_orientation', np.float32),  # theta
            ('start_spikings', (np.float32, 40*40*6)),
            ('goal_spikings', (np.float32, 40*40*6)),
            ('start_lidar', (np.float32, LidarReading.DEFAULT_NUMBER_OF_ANGLES)),
            ('goal_lidar', (np.float32, LidarReading.DEFAULT_NUMBER_OF_ANGLES)),
        ])


class ReachabilityController(ABC):
    """ Any algorithm that decides whether one place is reachable from another. """

    @abstractmethod
    def reachable(self, env: 'PybulletEnvironment', src: PlaceInfo, dst: PlaceInfo, path_l: Optional[float]=None) -> bool:
        """
        Decides whether dst is reachable from dst.

        Arguments:
        env       -- environment used
        src       -- source
        dst       -- destination
        path_l    -- the path arc length from src to dst, if it is known that both lie on the same trajectory
        """
        ...

    @staticmethod
    def factory(controller_type) -> Self:
        if controller_type == "view_overlap":
            from .reachability_utils import ViewOverlapReachabilityController
            return ViewOverlapReachabilityController()
        
        from .reachability_estimation import reachability_estimator_factory
        try:
            return reachability_estimator_factory(controller_type)
        except KeyError:
            pass
        raise ValueError("Controller type not found: " + controller_type)

ModelInput = list['torch.Tensor']
Prediction = tuple[float, types.Vector2D, types.Angle]

ImageForTorch = 'np.array[float, (4, 64, 64)]'
def transpose_image(img: types.Image) -> ImageForTorch:
    assert img.shape[1:] == (64, 64, 4) # that's the format returned by env.camera() and used in the rest of the code
    try: # for torch.Tensors
        img = img.transpose(1, 3).transpose(2, 3) # reorder 0123 -> 0321 -> 0312, i.e. channels first
    except ValueError: # for np.ndarrays
        img = img.transpose(0, 3, 1, 2)
    assert img.shape[1:] == (4, 64, 64)
    return img

def untranspose_image(img: ImageForTorch) -> types.Image:
    """ Inverse of transpose_image """
    assert img.shape[1:] == (4, 64, 64)
    try: # for torch.Tensors
        img = img.transpose(2, 3).transpose(1, 3) # 0312 -> 0321 -> 0123, i.e. channels last
    except ValueError: # for np.arrays
        img = img.transpose(0, 2, 3, 1)
    assert img.shape[1:] == (64, 64, 4)
    return img

Batch = list # only used for type hints - doesn't actually do anything
