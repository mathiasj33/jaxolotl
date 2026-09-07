import random

import pytest

from jaxolotl.ltl.automata import LDBA, ltl2ldba
from jaxolotl.ltl.automata.rabinizer import RABINIZER_PATH
from jaxolotl.ltl.logic import Assignment

requires_rabinizer = pytest.mark.skipif(
    not RABINIZER_PATH.exists(),
    reason=f"Rabinizer binary not found at {RABINIZER_PATH}",
)


def _ldba(propositions: set[str], edges, initial: int = 0) -> LDBA:
    """An LDBA from `(source, target, label_or_None, accepting)` tuples, pruned over the
    full assignment space so transition assignment sets are materialized."""
    ldba = LDBA(propositions)
    states = sorted({s for e in edges for s in (e[0], e[1])})
    for s in states:
        ldba.add_state(s, initial=(s == initial))
    for source, target, label, accepting in edges:
        ldba.add_transition(source, target, label, accepting)
    ldba.prune(list(Assignment.all_possible_assignments(tuple(sorted(propositions)))))
    return ldba


def _num_epsilons(ldba: LDBA) -> int:
    return sum(
        1 for s in ldba.states for t in ldba.state_to_transitions[s] if t.is_epsilon()
    )


def test_forced_pad_is_fused_into_incoming_letters():
    ldba = _ldba(
        {"a"},
        [
            (0, 0, "!a", False),
            (0, 1, "a", False),
            (1, 2, None, False),
            (2, 2, "t", True),
        ],
    )
    ldba.eliminate_forced_epsilons()
    assert ldba.num_states == 2
    assert _num_epsilons(ldba) == 0
    assert ldba.initial_state == 0
    assert ldba.get_next_state(0, {"a"}) == (1, False)
    assert ldba.get_next_state(0, set()) == (0, False)
    assert ldba.get_next_state(1, {"a"}) == (1, True)


def test_genuine_guess_is_kept():
    ldba = _ldba(
        {"a"},
        [
            (0, 0, "t", False),
            (0, 1, None, False),
            (1, 1, "a", True),
            (1, 1, "!a", False),
        ],
    )
    ldba.eliminate_forced_epsilons()
    assert ldba.num_states == 2
    assert _num_epsilons(ldba) == 1


def test_epsilon_only_nondeterministic_choice_is_kept():
    ldba = _ldba(
        {"a"},
        [
            (0, 1, None, False),
            (0, 2, None, False),
            (1, 1, "t", True),
            (2, 2, "t", False),
        ],
    )
    ldba.eliminate_forced_epsilons()
    assert ldba.num_states == 3
    assert ldba.get_ordered_epsilon_transitions(0) == [1, 2]


def test_forced_initial_state_moves_to_epsilon_target():
    ldba = _ldba(
        {"a"},
        [
            (0, 1, None, False),
            (1, 1, "a", True),
            (1, 1, "!a", False),
        ],
    )
    ldba.eliminate_forced_epsilons()
    assert ldba.num_states == 1
    assert ldba.initial_state == 0
    assert _num_epsilons(ldba) == 0
    assert ldba.get_next_state(0, {"a"}) == (0, True)


def test_compaction_remaps_state_info():
    ldba = _ldba(
        {"a"},
        [
            (0, 1, "a", False),
            (0, 0, "!a", False),
            (1, 2, None, False),
            (2, 2, "t", True),
        ],
    )
    ldba.state_to_info = {
        0: {"embedding": "initial"},
        1: {"embedding": "forced"},
        2: {"embedding": "accepting"},
    }

    ldba.eliminate_forced_epsilons()

    assert ldba.state_to_info == {
        0: {"embedding": "initial"},
        1: {"embedding": "accepting"},
    }


def test_retargeted_edge_merges_with_existing_sibling():
    ldba = _ldba(
        {"a", "b"},
        [
            (0, 2, "a & !b", False),
            (0, 1, "b & !a", False),
            (1, 2, None, False),
            (2, 2, "t", True),
        ],
    )
    ldba.eliminate_forced_epsilons()
    assert ldba.num_states == 2
    assert len(ldba.state_to_transitions[0]) == 1
    merged = ldba.state_to_transitions[0][0]
    assert merged.target == 1
    assert merged.valid_assignments == {
        Assignment(frozenset({"a"})),
        Assignment(frozenset({"b"})),
    }
    assert ldba.check_deterministic_transitions(0)


def test_forced_pad_chain_is_fully_collapsed():
    ldba = _ldba(
        {"a"},
        [
            (0, 1, "a", False),
            (0, 0, "!a", False),
            (1, 2, None, False),
            (2, 3, None, False),
            (3, 3, "t", True),
        ],
    )
    ldba.eliminate_forced_epsilons()
    assert ldba.num_states == 2
    assert _num_epsilons(ldba) == 0
    assert ldba.get_next_state(0, {"a"}) == (1, False)
    assert ldba.get_next_state(1, set()) == (1, True)


def test_raises_after_sink_completion():
    ldba = _ldba(
        {"a"},
        [
            (0, 1, "a", False),
            (1, 2, None, False),
            (2, 2, "t", True),
        ],
    )
    ldba.complete_sink_state()
    with pytest.raises(ValueError, match="complete_sink_state"):
        ldba.eliminate_forced_epsilons()


_PROPOSITIONS = ("a", "b", "c", "d")

_FORMULAS = [
    "G F a & G F b & F c & G !d",  # forced pad: co-safe F mixed into recurrence
    "G (d => F a) & F (b & F c)",  # forced pads in a response formula
    "F G !d & G F a",  # genuine guess: must stay nondeterministic
    "F (a & F (b & F c))",  # co-safe chain: no epsilon at all
]


def _build(formula: str, eliminate: bool) -> LDBA:
    assignments = list(Assignment.all_possible_assignments(_PROPOSITIONS))
    ldba = ltl2ldba(formula, _PROPOSITIONS, assignments)
    ldba.prune(assignments)
    if eliminate:
        ldba.eliminate_forced_epsilons()
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


def _has_forced_epsilon(ldba: LDBA, state: int) -> bool:
    """Whether `state` (in a sink-completed LDBA) offers an epsilon whose every
    letter alternative goes to the sink."""
    transitions = ldba.state_to_transitions[state]
    if not any(t.is_epsilon() for t in transitions):
        return False
    return all(t.target == ldba.sink_state for t in transitions if not t.is_epsilon())


def _accepting_events(ldba: LDBA, trace: list[set[str]]) -> list[bool]:
    """Steps `ldba` over `trace`, taking forced epsilons eagerly, and returns the
    per-step accepting events."""
    state = ldba.initial_state
    assert state is not None
    events = []
    for propositions in trace:
        while _has_forced_epsilon(ldba, state):
            state, accepting = ldba.get_next_state(state, set(), take_epsilon=True)
            assert not accepting
        state, accepting = ldba.get_next_state(state, propositions)
        events.append(accepting)
    return events


@requires_rabinizer
@pytest.mark.parametrize("formula", _FORMULAS)
def test_elimination_preserves_accepted_traces(formula):
    before = _build(formula, eliminate=False)
    after = _build(formula, eliminate=True)
    assert not any(_has_forced_epsilon(after, s) for s in after.states)
    rng = random.Random(0)
    for _ in range(100):
        trace = [{p for p in _PROPOSITIONS if rng.random() < 0.3} for _ in range(30)]
        assert _accepting_events(before, trace) == _accepting_events(after, trace)


@requires_rabinizer
def test_forced_formulas_become_epsilon_free():
    for formula, expected_epsilons in [
        ("G F a & G F b & F c & G !d", 0),
        ("G (d => F a) & F (b & F c)", 0),
        ("F G !d & G F a", 1),
    ]:
        ldba = _build(formula, eliminate=True)
        assert _num_epsilons(ldba) == expected_epsilons, formula
