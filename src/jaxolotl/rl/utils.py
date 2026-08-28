"""RL utility functions."""

import jax
from jaxtyping import PyTree

from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitState
from jaxolotl.environments.wrappers.wrapper import WrapperState


def stagger_initial_episodes(
    env_state: WrapperState, key: jax.Array, max_steps_in_episode: int
) -> PyTree:
    """Shorten each environment's first episode at a uniform random point. This
    decorrelates the initial episodes of each environment."""

    offsets = jax.random.randint(
        key,
        env_state.timestep.shape,
        0,
        max_steps_in_episode,
        dtype=env_state.timestep.dtype,
    )
    return env_state.replace_in_chain(TimeLimitState, timestep=offsets)
