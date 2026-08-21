import jax.numpy as jnp
import numpy.testing as npt

from jaxolotl.alg.common.reach_avoid.jax_sequence import JaxReachAvoidSequence
from jaxolotl.alg.deep_ltl.eval.preprocessing import _batch_sequences


def _sequence(shape: tuple[int, int, int], repeat_last: int) -> JaxReachAvoidSequence:
    reach = jnp.zeros(shape, dtype=jnp.int32)
    return JaxReachAvoidSequence(
        reach=reach,
        avoid=reach,
        repeat_last=jnp.full(shape[:2], repeat_last, dtype=jnp.int32),
        last_index=jnp.full(shape[:2], 2, dtype=jnp.int32),
    )


def test_batch_sequences_uses_semantic_padding_values():
    short = _sequence((1, 1, 1), repeat_last=3)
    long = _sequence((2, 1, 2), repeat_last=5)

    batched = _batch_sequences([short, long])

    assert batched.reach.shape == (2, 2, 1, 2)
    npt.assert_array_equal(batched.reach[0, 0, 0], [0, -1])
    npt.assert_array_equal(batched.reach[0, 1], -1)
    npt.assert_array_equal(batched.repeat_last[0, :, 0], [3, 1])
    npt.assert_array_equal(batched.last_index[0, :, 0], [2, 0])
