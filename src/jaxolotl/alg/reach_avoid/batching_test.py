import jax.numpy as jnp
import numpy.testing as npt
import pytest

from jaxolotl.alg.reach_avoid.batching import (
    batch_assignments,
    batch_state_sequences,
)
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.reach_avoid.sequence import EPSILON, ReachAvoidSequence


def test_batch_assignments_preserves_epsilon_and_repeat_last():
    red = Assignment("red")
    green = Assignment("green")
    sequence = ReachAvoidSequence(
        [(EPSILON, frozenset({red})), (frozenset({green}), frozenset())],
        repeat_last=7,
    )

    encoded = batch_assignments([sequence], [red, green])

    npt.assert_array_equal(encoded.reach[0, :, 0], [2, 1])
    npt.assert_array_equal(encoded.avoid[0, 0], [0, -1])
    npt.assert_array_equal(encoded.repeat_last, [7])


def test_batch_state_sequences_supports_sparse_state_indices_and_padding():
    state_to_values = {0: [10, 11], 2: [20]}

    encoded = batch_state_sequences(
        state_to_values,
        lambda values: {
            "value": jnp.asarray(values),
            "valid": jnp.ones(len(values), dtype=bool),
        },
        {"value": -1, "valid": False},
    )

    npt.assert_array_equal(encoded["value"], [[10, 11], [-1, -1], [20, -1]])
    npt.assert_array_equal(
        encoded["valid"], [[True, True], [False, False], [True, False]]
    )


def test_batch_state_sequences_validates_num_states():
    with pytest.raises(ValueError, match="cannot contain state index"):
        batch_state_sequences({2: [20]}, jnp.asarray, jnp.asarray(-1), num_states=2)
