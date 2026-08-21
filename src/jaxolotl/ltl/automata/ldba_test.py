from jaxolotl.ltl.automata.ldba import LDBA, SCC
from jaxolotl.ltl.logic import Assignment


def _add_states(ldba: LDBA, count: int) -> None:
    for state in range(count):
        ldba.add_state(state, initial=state == 0)


def test_prune_restricts_transitions_to_possible_assignments() -> None:
    ldba = LDBA({"a", "b"})
    _add_states(ldba, 3)
    retained = ldba.add_transition(0, 1, "a | b", False)
    removed = ldba.add_transition(0, 2, "a & b", False)
    epsilon = ldba.add_transition(1, 2, None, False)
    possible_assignments = [Assignment(), Assignment("a"), Assignment("b")]

    ldba.prune(possible_assignments)

    assert retained.valid_assignments == {Assignment("a"), Assignment("b")}
    assert retained in ldba.state_to_transitions[0]
    assert retained in ldba.state_to_incoming_transitions[1]
    assert removed not in ldba.state_to_transitions[0]
    assert removed not in ldba.state_to_incoming_transitions[2]
    assert epsilon in ldba.state_to_transitions[1]
    assert ldba.num_transitions == 2
    assert ldba.possible_assignments == possible_assignments


def test_complete_sink_state_makes_a_pruned_ldba_total_and_is_idempotent() -> None:
    ldba = LDBA({"a", "b"})
    _add_states(ldba, 2)
    ldba.add_transition(0, 1, "a", False)
    ldba.add_transition(1, 1, "t", True)
    possible_assignments = [Assignment(), Assignment("a"), Assignment("b")]
    ldba.prune(possible_assignments)

    ldba.complete_sink_state()

    assert ldba.complete
    assert ldba.sink_state == 2
    assert ldba.num_states == 3
    assert ldba.num_transitions == 4

    all_assignments = set(possible_assignments)
    for state in ldba.states:
        transitions = ldba.state_to_transitions[state]
        covered = [
            assignment
            for transition in transitions
            for assignment in transition.valid_assignments
        ]
        assert set(covered) == all_assignments
        assert len(covered) == len(all_assignments)

    assert ldba.get_next_state(0, set()) == (ldba.sink_state, False)
    assert ldba.get_next_state(0, {"a"}) == (1, False)
    assert ldba.get_next_state(0, {"b"}) == (ldba.sink_state, False)
    assert ldba.get_next_state(ldba.sink_state, {"a"}) == (
        ldba.sink_state,
        False,
    )

    snapshot = (
        ldba.num_states,
        ldba.num_transitions,
        {
            state: tuple(transitions)
            for state, transitions in ldba.state_to_transitions.items()
        },
    )
    ldba.complete_sink_state()
    assert (
        ldba.num_states,
        ldba.num_transitions,
        {
            state: tuple(transitions)
            for state, transitions in ldba.state_to_transitions.items()
        },
    ) == snapshot


def test_compute_sccs_classifies_accepting_and_bottom_components() -> None:
    ldba = LDBA({"a"})
    _add_states(ldba, 4)
    ldba.add_transition(0, 0, "t", False)
    ldba.add_transition(0, 1, None, False)
    ldba.add_transition(0, 3, "a", False)
    ldba.add_transition(1, 2, "t", False)
    ldba.add_transition(2, 1, "t", True)
    ldba.add_transition(3, 3, "t", False)

    ldba.compute_sccs()

    initial = SCC(frozenset({0}), accepting=False, bottom=False)
    accepting = SCC(frozenset({1, 2}), accepting=True, bottom=True)
    rejecting = SCC(frozenset({3}), accepting=False, bottom=True)
    assert ldba.state_to_scc == {
        0: initial,
        1: accepting,
        2: accepting,
        3: rejecting,
    }

    state_to_scc = ldba.state_to_scc.copy()
    ldba.compute_sccs()
    assert ldba.state_to_scc == state_to_scc
