from .environment.map_occupancy import environment_dimensions as _environment_dims_as_tuple

def environment_dimensions(map_name):
    origin, corner = _environment_dims_as_tuple(map_name)
    return [ origin[0], corner[0], origin[1], corner[1] ]
