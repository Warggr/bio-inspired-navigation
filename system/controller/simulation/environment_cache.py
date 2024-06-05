from .pybullet_environment import PybulletEnvironment
from system.types import types
from typing import Dict, Self, Optional

# TODO Pierre: this is not compatible with PybulletEnvironment as a context manager
# nor with having a shared memory server
# which is rather good news, because we don't need the context management if we're not using a shmem server
class EnvironmentCache:
    class DONT_CARE: # Sentinel value
        pass

    default_env_kwargs = { 'build_data_set': True, 'contains_robot': False }

    def __init__(self, override_env_kwargs={}):
        self.envs : Dict[types.AllowedMapName, PybulletEnvironment] = {}
        self.override_env_kwargs = override_env_kwargs
        self.env_kwargs = { key: value for key, value in override_env_kwargs.items() if value is not EnvironmentCache.DONT_CARE }
        self.env_kwargs = self.env_kwargs | self.default_env_kwargs

    # Singleton pattern
    _instance : Optional[Self] = None

    @classmethod
    def create_instance(cls, override_env_kwargs) -> Self:
        assert cls._instance is None
        cls._instance = cls(override_env_kwargs)
        return cls._instance

    @classmethod
    def getinstance(cls, override_env_kwargs) -> Self:
        if cls._instance is None:
            return cls.create_instance(override_env_kwargs)
        elif cls.is_compatible_with(override_env_kwargs, cls._instance.override_env_kwargs):
            return cls._instance
        elif len(cls._instance.envs) == 0:
            if cls.is_compatible_with(cls._instance.override_env_kwargs, override_env_kwargs):
                # No envs have been created yet and new conf is backwards compatible -> let's silently migrate to the new conf
                cls._instance = cls(override_env_kwargs)
                return cls._instance
            else:
                raise ValueError(f"Could not reconcile configurations: old={cls._instance.override_env_kwargs}, new={override_env_kwargs}")
        else:
            raise ValueError(f"Could not reuse already-launched configuration {cls._instance.override_env_kwargs} for requested {override_env_kwargs}")

    def load(self, env_name : types.AllowedMapName):
        if env_name in self.envs:
            raise KeyError(f"Environment {env_name} already loaded")
        self.envs[env_name] = PybulletEnvironment(env_name, **self.env_kwargs)

    def __getitem__(self, env_name : types.AllowedMapName) -> PybulletEnvironment:
        if env_name not in self.envs:
            self.load(env_name)
        return self.envs[env_name]

    @classmethod
    def is_compatible_with(cls, new_conf, previous_conf):
        # Make sure that the env_kwargs are the same, 
        # except those that are default for PybulletEnvironment - for those it makes no difference whether they're set to the default or unset
        default_env_kwargs = { 'dt': 1e-2 } # TODO read all of them with inspect.signature
        previous_conf = previous_conf | EnvironmentCache.default_env_kwargs | default_env_kwargs
        new_conf = new_conf | EnvironmentCache.default_env_kwargs | default_env_kwargs

        dont_care_keys = [ key for key, value in new_conf.items() if value == EnvironmentCache.DONT_CARE ]
        for key in dont_care_keys:
            del new_conf[key]
            if key in previous_conf: del previous_conf[key]
        return previous_conf == new_conf
