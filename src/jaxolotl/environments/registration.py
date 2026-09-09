from functools import partial

from jaxolotl.environments.conveyor_world.conveyor_world import ConveyorWorld
from jaxolotl.environments.environment import Environment, EnvParams
from jaxolotl.environments.franka_zone_env.franka_zone_env import FrankaZoneEnv
from jaxolotl.environments.letter_world.letter_world import LetterWorld
from jaxolotl.environments.reset import (
    RESET_SOURCES,
    ResetSource,
    precomputed_reset_path,
)
from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxolotl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.environments.zone_env.zone_env import ZoneEnv

_name_to_env = {
    "ZoneEnv": ZoneEnv,
    "ZoneEnv-NM": partial(ZoneEnv, non_myopic=True),
    "LetterWorld": LetterWorld,
    "WarehouseEnv": WarehouseEnv,
    "ConveyorWorld": ConveyorWorld,
    **{
        f"FrankaZoneEnv-{num_colors}": partial(FrankaZoneEnv, num_colors=num_colors)
        for num_colors in range(4, 17)
    },
}


def make(
    name: str, *, reset_source: ResetSource = "native", **kwargs
) -> tuple[Environment | EnvWrapper, EnvParams]:
    """Create an environment by name.

    Args:
        name: Registered environment name.
        reset_source: Use native resets or sample from the precomputed training or
            test reset pool.
        **kwargs: Arguments forwarded to the environment constructor.

    Returns:
        A tuple of the environment instance and its default parameters."""
    if reset_source not in RESET_SOURCES:
        raise ValueError(
            f"Unknown reset source {reset_source!r}; expected one of {RESET_SOURCES}."
        )

    env_class = _name_to_env.get(name)
    if not env_class:
        raise ValueError(f"Unknown environment name: {name}")
    env = env_class(**kwargs)
    params = env.default_params

    if reset_source != "native":
        path = precomputed_reset_path(name, reset_source)
        if not path.is_file():
            raise FileNotFoundError(
                f"No precomputed {reset_source!r} resets found for {name!r} at "
                f"{path}. Generate them with `pixi run -e gpu python "
                f"scripts/precompute_resets.py env=<config>."
            )
        env = PrecomputedResetWrapper(env, params, path)

    return env, params
