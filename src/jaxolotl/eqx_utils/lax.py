"""Equinox-compatible lax utilities."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax


def filter_scan[Carry, X, Y](
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
) -> tuple[Carry, Y]:
    """A wrapper around jax.lax.scan that supports equinox modules."""
    carry_params, carry_static = eqx.partition(init, eqx.is_array)

    def aux(carry_params, x):
        carry = eqx.combine(carry_params, carry_static)
        carry, y = f(carry, x)
        carry_params, _ = eqx.partition(carry, eqx.is_array)
        return carry_params, y

    carry_params, y = jax.lax.scan(
        aux,
        carry_params,
        xs,
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
    )
    carry = eqx.combine(carry_params, carry_static)
    return carry, y


def filter_map(f, xs, *, batch_size: int | None = None):
    """A wrapper around jax.lax.map that supports equinox modules. Does not support
    equinox modules as outputs."""
    params, static = eqx.partition(xs, eqx.is_array)

    def aux(params):
        x = eqx.combine(params, static)
        return f(x)

    return jax.lax.map(aux, params, batch_size=batch_size)


def filter_while_loop[T](
    cond_fun: Callable[[T], Any], body_fun: Callable[[T], T], init_val: T
) -> T:
    """A wrapper around jax.lax.while_loop that supports equinox modules."""
    params, static = eqx.partition(init_val, eqx.is_array)

    def aux(params):
        val = eqx.combine(params, static)
        return cond_fun(val)

    def body_aux(params):
        val = eqx.combine(params, static)
        new_val = body_fun(val)
        new_params, _ = eqx.partition(new_val, eqx.is_array)
        return new_params

    final_params = jax.lax.while_loop(aux, body_aux, params)
    final_val = eqx.combine(final_params, static)
    return final_val
