import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.environments.wrappers.wrapper import WrapperState


class _Base(eqx.Module):
    value: jax.Array


class _Inner(WrapperState):
    x: jax.Array


class _Outer(WrapperState):
    pass


def test_replace_in_chain() -> None:
    stack = _Outer(state=_Inner(state=_Base(value=jnp.arange(8)), x=jnp.asarray(5)))
    stack = stack.replace_in_chain(_Inner, x=jnp.asarray(10))
    assert isinstance(stack, _Outer)
    assert isinstance(stack.state, _Inner)
    assert isinstance(stack.state.state, _Base)
    assert jnp.array_equal(stack.state.x, jnp.asarray(10))
    assert jnp.array_equal(stack.state.state.value, jnp.arange(8))
