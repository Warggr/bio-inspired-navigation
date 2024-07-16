from typing import Sized, Tuple, Iterable, List, Protocol, Literal
import numpy as np

Angle = float # assumed to be in radians

class LidarReading:
    def __init__(self, distances: List[float], angles: Iterable[Angle]):
        self.distances = distances
        self.angles = list(angles)

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

class Vector2D(Sized, Iterable[float], Protocol):
    #def __getitem__(self, index : int) -> float: ...
    pass

Vector3D = Tuple[float, float, float]
# (grid cell) spikings
Spikings = 'np.ndarray[float, (40, 40, 6)]' # TODO: actual non-string type hint
Image = 'np.ndarray[float, (64, 64, 4)]'

allowed_map_names = [
    'Savinov_val3', 'linear_sunburst'
]
AllowedMapName = Literal[*allowed_map_names]
AllowedMapName.options = allowed_map_names

class types:
    Vector2D = Vector2D
    Vector3D = Vector3D
    Quaternion = Tuple[float, float, float, float]
    Spikings = Spikings
    Angle = Angle
    Image = Image
    PositionAndOrientation = Tuple[Vector2D, Angle]
    AllowedMapName = AllowedMapName
    LidarReading = LidarReading

FlatSpikings = 'np.ndarray[float, 9600]'
WaypointInfo = Tuple[types.Vector2D, types.Angle, FlatSpikings]
