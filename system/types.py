from typing import Tuple, Iterable, List
import numpy as np

Angle = float # assumed to be in radians

class LidarReading:
    def __init__(self, distances: List[float], angles: List[Angle]):
        self.distances = distances
        self.angles = angles

    def __getitem__(self, index):
        return self.distances[index]

    @staticmethod
    def angles(
        start_angle,
        tactile_cone = np.radians(310),
        num_ray_dir = 62, # number of directions to check (e.g. 16,51,71)
        blind_spot_cone = np.radians(50),
    ) -> Iterable[Angle]:
        max_angle = tactile_cone / 2
        for angle_offset in np.linspace(-max_angle, max_angle, num=num_ray_dir):
            if abs(angle_offset) < blind_spot_cone / 2:
                continue
            yield start_angle + angle_offset

    DEFAULT_NUMBER_OF_ANGLES = 52

assert LidarReading.DEFAULT_NUMBER_OF_ANGLES == len(list(LidarReading.angles(0)))


Vector2D = Tuple[float, float]
Vector3D = Tuple[float, float, float]

class types:
    DepthImage = np.ndarray
    Vector2D = Vector2D
    Vector3D = Vector3D
    Quaternion = Tuple[float, float, float, float]
    Spikings = 'np.ndarray[float, (40, 40, 6)]' # TODO: actual non-string type hint
    Angle = Angle
    Image = 'np.ndarray[float, (64, 64, 4)]'
    PositionAndOrientation = Tuple[Vector3D, Angle]
    AllowedMapName = str # TODO: an enumeration of the actual map names, i.e. Literal['Savinov_val3', ...]
    LidarReading = LidarReading

FlatSpikings = 'np.ndarray[float, 9600]'
WaypointInfo = Tuple[types.Vector2D, types.Angle, FlatSpikings]
