"""Generic batching utilities for array PyTrees."""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def pad_and_stack(pytrees: Sequence[PyTree], padding_values: PyTree) -> PyTree:
    """Pad corresponding array leaves to their maximum shapes and stack them.

    Args:
        pytrees: Sequence of PyTrees to pad and stack. Each PyTree must have the same
            structure and corresponding leaves must have the same rank.
        padding_values: PyTree of scalar values to use for padding. Must have the same
            structure as the PyTrees in `pytrees`.

    Returns:
        A single PyTree with the same structure as the input PyTrees, where each leaf is
        a stacked array of the corresponding leaves from the input PyTrees, padded to the
        maximum shape along each axis with the specified padding values.
    """
    if not pytrees:
        raise ValueError("Cannot pad and stack an empty PyTree batch.")

    structure = jax.tree.structure(padding_values)
    if any(jax.tree.structure(pytree) != structure for pytree in pytrees):
        raise ValueError("PyTrees and padding values must have matching structures.")

    def maximum_shape(*leaves: jax.Array) -> tuple[int, ...]:
        ranks = {leaf.ndim for leaf in leaves}
        if len(ranks) != 1:
            raise ValueError("Corresponding PyTree leaves must have matching ranks.")
        return tuple(
            max(leaf.shape[axis] for leaf in leaves) for axis in range(leaves[0].ndim)
        )

    target_shapes = jax.tree.map(maximum_shape, *pytrees)

    def pad_pytree(pytree: PyTree) -> PyTree:
        def pad_leaf(
            leaf: jax.Array, target_shape: tuple[int, ...], padding_value: Any
        ) -> jax.Array:
            padding = tuple(
                (0, target - current)
                for current, target in zip(leaf.shape, target_shape, strict=True)
            )
            return jnp.pad(leaf, padding, constant_values=padding_value)

        return jax.tree.map(pad_leaf, pytree, target_shapes, padding_values)

    padded = [pad_pytree(pytree) for pytree in pytrees]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *padded)
