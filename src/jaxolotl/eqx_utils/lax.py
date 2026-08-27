"""Equinox-compatible lax utilities."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax


def batch_map[P, T](f: Callable[[P], T], xs, *, batch_size: int | None = None) -> T:
    """Map ``f`` over leading axes, including outputs with zero-sized dimensions.

    This matches :func:`jax.lax.map`'s ``batch_size`` behaviour, but avoids its
    internal ``reshape(-1, ...)``. That reshape is ambiguous when a mapped result
    contains a zero-sized non-batch axis.
    """
    if batch_size is None:
        return jax.lax.map(f, xs)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    leaves = jax.tree.leaves(xs)
    if not leaves:
        return jax.lax.map(f, xs, batch_size=batch_size)
    num_items = leaves[0].shape[0]
    if num_items == 0:
        return jax.vmap(f)(xs)
    num_batches, remainder = divmod(num_items, batch_size)
    batch_items = num_batches * batch_size

    def split_batches(leaf):
        return leaf[:batch_items].reshape((num_batches, batch_size, *leaf.shape[1:]))

    if num_batches:
        batched_xs = jax.tree.map(split_batches, xs)
        _, batched_ys = jax.lax.scan(
            lambda _, x: (None, jax.vmap(f)(x)), None, batched_xs
        )
        # Specify the leading dimension directly. Inferring it with ``-1`` fails
        # when an output has a zero-sized dimension.
        ys = jax.tree.map(lambda x: x.reshape((batch_items, *x.shape[2:])), batched_ys)
    if remainder:
        remainder_xs = jax.tree.map(lambda leaf: leaf[batch_items:], xs)
        remainder_ys = jax.vmap(f)(remainder_xs)
        if num_batches:
            ys = jax.tree.map(
                lambda full, tail: jax.lax.concatenate([full, tail], dimension=0),
                ys,
                remainder_ys,
            )
        else:
            ys = remainder_ys
    return ys  # type: ignore[return-value]


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

    return batch_map(aux, params, batch_size=batch_size)


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
