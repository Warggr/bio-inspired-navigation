# About
This file is intended for future developers to get familiar with the program.

# Code style

Many functions are annotated with type hints. Common types are declared in [system/types.py](system/types.py) and [system/controller/reachability_estimator/types.py](system/controller/reachability_estimator/types.py).
This has two goals: First, to make the code more readable / self-documented. Secondly, so that IDEs can read the type hints and extract e.g. autocompletion and warning messages for wrong types.
However, some types are given only for the developer's convenience and would emit warnings if used with a real type checker. For example, the type hint `Vector2D` (intuitively, a 2D vector) can be either a tuple, a list, or a Numpy array; some functions expect it to be one of those specifically. (Or `Batch` can be either a Torch tensor, a list, or a Numpy array).

# Misc
There's a type `AllowedMapName` in system.types. Variable of that type are usually called either `env_model` or `map_name`, there's no logic behind when they're called what.

# Feature backlog

- During the exploration phase, we create a trajectories dataset. Couldn't we reuse this for future exploration phases,
instead of actually doing the simulation?
- CognitiveMaps should have an `env_model` attribute
