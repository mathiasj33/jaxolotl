import jax.numpy as jnp
import numpy.testing as npt

from jaxolotl.alg.genz_ltl.eval.preprocessing import _batch_subgoals
from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)


def _subgoals(num_states: int) -> JaxReachAvoidSubgoal:
    return JaxReachAvoidSubgoal(
        reach=jnp.zeros((num_states, 1), dtype=jnp.int32),
        avoid=jnp.zeros((num_states, 1, 2), dtype=jnp.int32),
        reach_one_hot=jnp.ones((num_states, 1, 2), dtype=jnp.int32),
        avoid_one_hot=jnp.ones((num_states, 1, 2), dtype=jnp.int32),
    )


def test_batch_subgoals_zero_pads_one_hot_features():
    batched = _batch_subgoals([_subgoals(1), _subgoals(2)])

    npt.assert_array_equal(batched.reach[0, 1], [-1])
    npt.assert_array_equal(batched.avoid[0, 1], [[-1, -1]])
    npt.assert_array_equal(batched.reach_one_hot[0, 1], [[0, 0]])
    npt.assert_array_equal(batched.avoid_one_hot[0, 1], [[0, 0]])
