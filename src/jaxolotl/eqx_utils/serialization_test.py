from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.eqx_utils.serialization import (
    load_from_template,
    load_metadata,
    save_with_template,
)


class NestedArrays(NamedTuple):
    mask: jax.Array


class ArrayModule(eqx.Module):
    values: jax.Array
    nested: NestedArrays


def test_template_serialization_round_trip(tmp_path):
    path = tmp_path / "tree.eqx"
    tree = ArrayModule(
        values=jnp.arange(6, dtype=jnp.int32).reshape(2, 3),
        nested=NestedArrays(mask=jnp.array([[True, False, True]])),
    )

    save_with_template(path, tree, metadata={"num_samples": 3})
    loaded = load_from_template(path)

    assert isinstance(loaded, ArrayModule)
    assert isinstance(loaded.nested, NestedArrays)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, tree, loaded))
    assert load_metadata(path)["num_samples"] == 3


def test_template_serialization_does_not_mutate_metadata(tmp_path):
    path = tmp_path / "tree.eqx"
    metadata = {"num_samples": 1}
    tree = ArrayModule(jnp.ones(1), NestedArrays(jnp.ones(1, dtype=bool)))

    save_with_template(path, tree, metadata)

    assert metadata == {"num_samples": 1}
