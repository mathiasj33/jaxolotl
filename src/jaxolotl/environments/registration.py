from functools import partial

from jaxolotl.environments.environment import Environment, EnvParams
from jaxolotl.environments.letter_world.letter_world import LetterWorld
from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxolotl.environments.zone_env.zone_env import ZoneEnv

_name_to_env = {
    "ZoneEnv": ZoneEnv,
    "ZoneEnv-NM": partial(ZoneEnv, non_myopic=True),
    "LetterWorld": LetterWorld,
    "WarehouseEnv": WarehouseEnv,
}


def make(name: str, **kwargs) -> tuple[Environment, EnvParams]:
    """Create an environment by name.

    Returns:
        A tuple of the environment instance and its default parameters."""
    env_class = _name_to_env.get(name)
    if not env_class:
        raise ValueError(f"Unknown environment name: {name}")
    env = env_class(**kwargs)
    return env, env.default_params  # type: ignore
