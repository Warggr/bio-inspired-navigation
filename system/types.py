from typing import Sized, Iterable, Protocol, Literal
import numpy as np

Angle = float  # assumed to be in radians


class LidarReading:
    def __init__(self, distances: list[float], angles: Iterable[Angle]):
        self.distances = distances
        self.angles = list(angles)

    def __getitem__(self, index):
        return self.distances[index]

    @staticmethod
    def angle_range(
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


assert LidarReading.DEFAULT_NUMBER_OF_ANGLES == len(list(LidarReading.angle_range(0)))


class Vector2D(Sized, Iterable[float], Protocol):
    # def __getitem__(self, index: int) -> float: ...
    pass


Vector3D = tuple[float, float, float]
PositionAndOrientation = tuple[Vector2D, Angle]
Quaternion = tuple[float, float, float, float]
# (grid cell) spikings
Spikings = 'np.ndarray[float, (40, 40, 6)]' # TODO: actual non-string type hint
Image = 'np.ndarray[float, (64, 64, 4)]'

allowed_map_names = [
    'Savinov_val3', 'linear_sunburst', 'obstacle_map_0'
]
AllowedMapName = Literal['Savinov_val3', 'Savinov_val2', 'Savinov_test7', 'linear_sunburst', 'obstacle_map_0', 'obstacle_map_1', 'plane']
AllowedMapName.options = allowed_map_names


class types:
    Vector2D = Vector2D
    Vector3D = Vector3D
    Quaternion = Quaternion
    Spikings = Spikings
    Angle = Angle
    Image = Image
    PositionAndOrientation = PositionAndOrientation
    AllowedMapName = AllowedMapName
    LidarReading = LidarReading


FlatSpikings = 'np.ndarray[float, 9600]'
WaypointInfo = tuple[types.Vector2D, types.Angle, FlatSpikings]
