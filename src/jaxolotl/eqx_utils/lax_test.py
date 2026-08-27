import jax
import jax.numpy as jnp

from jaxolotl.eqx_utils.lax import batch_map


def test_map_batched_preserves_zero_sized_output_axes():
    def f(x):
        return x + 1, jnp.zeros((0, 2), dtype=jnp.int32)

    values, empty = batch_map(f, jnp.arange(3), batch_size=2)

    assert values.tolist() == [1, 2, 3]
    assert empty.shape == (3, 0, 2)


def test_map_matches_lax_map():
    def f(x):
        return x + 1, jnp.zeros((5, 2), dtype=jnp.int32)

    values, empty = batch_map(f, jnp.arange(3), batch_size=2)
    values_lax, empty_lax = jax.lax.map(f, jnp.arange(3))

    assert jnp.all(values == values_lax)
    assert jnp.all(empty == empty_lax)
