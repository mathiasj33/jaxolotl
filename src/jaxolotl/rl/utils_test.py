import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitState
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.rl.utils import stagger_initial_episodes

MAX_STEPS = 100


class _Base(eqx.Module):
    value: jax.Array


class _Outer(WrapperState):
    pass


def _stack() -> _Outer:
    return _Outer(
        state=TimeLimitState(
            state=_Base(value=jnp.arange(8)),
            timestep=jnp.zeros((8,), dtype=jnp.int32),
        )
    )


def test_stagger_offsets_declared_timestep() -> None:
    out = stagger_initial_episodes(_stack(), jax.random.key(0), MAX_STEPS)

    assert out.timestep.shape == (8,)
    assert bool(jnp.all(out.timestep >= 0))
    assert bool(jnp.all(out.timestep < MAX_STEPS))
    assert bool(jnp.any(out.timestep != 0))


def test_stagger_preserves_unrelated_state() -> None:
    state = _stack()
    out = stagger_initial_episodes(state, jax.random.key(0), MAX_STEPS)

    assert jnp.array_equal(out.value, state.value)
