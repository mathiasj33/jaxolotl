import numpy as np
import pytest

from jaxolotl.alg.ltl2action.utils.formula_closure import FormulaClosureGraph
from jaxolotl.alg.ltl2action.utils.jax_formula_closure import JaxFormulaClosureGraph
from jaxolotl.ltl.logic.assignment import Assignment


class _StubEnv:
    propositions = ("a", "b")

    @staticmethod
    def assignments():
        return (Assignment(), Assignment("a"), Assignment("b"))


def _build_closure(formula: str) -> FormulaClosureGraph:
    closure = FormulaClosureGraph(formula)
    closure.build(_StubEnv.assignments())
    return closure


def test_safety_residual_accepts_under_finite_semantics():
    # G !b & F a never progresses to a literal True: once a is reached the
    # residual is G !b, which is accepting under truncated semantics.
    env = _StubEnv()
    assignments = env.assignments()
    empty_idx = assignments.index(Assignment())
    a_idx = assignments.index(Assignment("a"))
    b_idx = assignments.index(Assignment("b"))

    batched = JaxFormulaClosureGraph.from_closure_graphs(
        [_build_closure("G !b & F a")],
        env,  # type: ignore
        finite=True,
    )

    init = int(batched.initial_state[0])
    true_state = int(batched.true_state[0])
    false_state = int(batched.false_state[0])
    transitions = np.asarray(batched.transitions[0])

    assert true_state >= 0
    assert transitions[init, a_idx] == true_state
    assert transitions[init, empty_idx] == init
    assert transitions[init, b_idx] == false_state
    assert transitions[true_state, empty_idx] == true_state


def test_finite_flag_is_a_no_op_for_co_safe_formulas():
    env = _StubEnv()

    default = JaxFormulaClosureGraph.from_closure_graphs([_build_closure("F a")], env)  # type: ignore
    finite = JaxFormulaClosureGraph.from_closure_graphs(
        [_build_closure("F a")],
        env,  # type: ignore
        finite=True,
    )

    assert np.array_equal(
        np.asarray(default.transitions), np.asarray(finite.transitions)
    )
    assert int(default.true_state[0]) == int(finite.true_state[0])


def test_missing_true_state_raises_without_finite():
    env = _StubEnv()

    with pytest.raises(AssertionError, match="True state not found"):
        JaxFormulaClosureGraph.from_closure_graphs([_build_closure("G !b & F a")], env)  # type: ignore
