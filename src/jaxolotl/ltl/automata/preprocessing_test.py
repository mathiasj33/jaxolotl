from unittest.mock import Mock

import jax.numpy as jnp
import numpy.testing as npt
import pytest

from jaxolotl.ltl.automata.jax_ldba import JaxLDBA
from jaxolotl.ltl.automata.preprocessing import batch_ldbas, build_ldba


def _jax_ldba(num_states: int, initial_state: int, *, finite: bool) -> JaxLDBA:
    transitions = jnp.arange(num_states * 3, dtype=jnp.int32).reshape(num_states, 3)
    return JaxLDBA(
        num_states=jnp.asarray(num_states, dtype=jnp.int32),
        initial_state=jnp.asarray(initial_state, dtype=jnp.int32),
        transitions=transitions,
        accepting=transitions[:, :2] % 2 == 0,
        sink_states=jnp.arange(num_states) == num_states - 1,
        finite=jnp.asarray(finite),
    )


def test_batch_ldbas_preserves_values_and_pads_states():
    first = _jax_ldba(2, 1, finite=False)
    second = _jax_ldba(1, 0, finite=True)

    batched = batch_ldbas([first, second])

    npt.assert_array_equal(batched.num_states, [2, 1])
    npt.assert_array_equal(batched.initial_state, [1, 0])
    npt.assert_array_equal(batched.transitions[0], first.transitions)
    npt.assert_array_equal(batched.transitions[1, 0], second.transitions[0])
    npt.assert_array_equal(batched.transitions[1, 1], [-1, -1, -1])
    npt.assert_array_equal(batched.accepting[1, 1], [False, False])
    assert not batched.sink_states[1, 1]
    npt.assert_array_equal(batched.finite, [False, True])


def test_batch_ldbas_rejects_empty_or_incompatible_batches():
    with pytest.raises(ValueError, match="empty"):
        batch_ldbas([])

    incompatible = _jax_ldba(1, 0, finite=False)._replace(
        transitions=jnp.zeros((1, 4), dtype=jnp.int32)
    )
    with pytest.raises(ValueError, match="assignment space"):
        batch_ldbas([_jax_ldba(1, 0, finite=False), incompatible])


def test_build_ldba_runs_the_complete_preprocessing_chain(monkeypatch):
    ldba = Mock()
    converter = Mock(return_value=ldba)
    monkeypatch.setattr(
        "jaxolotl.ltl.automata.preprocessing.ltl2ldba",
        converter,
    )
    env = Mock(propositions=("red", "green"))
    env.assignments.return_value = ["red", "green"]

    result = build_ldba("F red", env)

    assert result is ldba
    converter.assert_called_once_with("F red", ("red", "green"))
    ldba.prune.assert_called_once_with(["red", "green"])
    ldba.complete_sink_state.assert_called_once_with()
    ldba.compute_sccs.assert_called_once_with()
