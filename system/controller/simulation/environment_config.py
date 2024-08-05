from .environment.map_occupancy import environment_dimensions as _environment_dims_as_tuple
from system.types import AllowedMapName

def environment_dimensions(map_name: AllowedMapName) -> tuple[float, float, float, float]:
    """
    :return: x_min, x_max, y_min, y_max
    """
    origin, corner = _environment_dims_as_tuple(map_name)
    return (origin[0], corner[0], origin[1], corner[1])
