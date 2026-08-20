import jax.numpy as jnp
import numpy.testing as npt
import pytest

from jaxolotl.eqx_utils.batching import pad_and_stack


def test_pad_and_stack_uses_per_leaf_shapes_and_semantic_values():
    first = {
        "indices": jnp.asarray([[1], [2]], dtype=jnp.int32),
        "mask": jnp.asarray([True]),
    }
    second = {
        "indices": jnp.asarray([[3, 4, 5]], dtype=jnp.int32),
        "mask": jnp.asarray([True, True, False]),
    }

    batched = pad_and_stack(
        [first, second],
        {"indices": jnp.asarray(-1), "mask": jnp.asarray(False)},
    )

    assert batched["indices"].shape == (2, 2, 3)
    npt.assert_array_equal(batched["indices"][0, 1], [2, -1, -1])
    npt.assert_array_equal(batched["indices"][1, 1], [-1, -1, -1])
    npt.assert_array_equal(batched["mask"], [[True, False, False], [True, True, False]])


def test_pad_and_stack_rejects_empty_batches_and_rank_mismatches():
    with pytest.raises(ValueError, match="empty"):
        pad_and_stack([], {})

    with pytest.raises(ValueError, match="matching ranks"):
        pad_and_stack(
            [jnp.ones((2,)), jnp.ones((1, 2))],
            jnp.asarray(0),
        )
