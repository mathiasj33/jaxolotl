from pathlib import Path

from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxolotl.ltl.automata import LDBA
from jaxolotl.ltl.hoa import HOAParser
from jaxolotl.ltl.logic import Assignment
from jaxolotl.ltl.reach_avoid.path_search import compute_sequences
from jaxolotl.ltl.reach_avoid.sequence import ReachAvoidSequence

TESTDATA = Path(__file__).with_name("testdata")

FINITE_FORMULA = "!a U (b & (!c U (d & (!e U f))))"
INFINITE_FORMULA = "GF a & GF b & GF c & GF d & G (!e & !f)"
WAREHOUSE_FORMULA = (
    "((crate => (crate U region_b)) U "
    "(crate & region_b & (region_b U (!crate & region_b)))) "
    "| F (door & F (region_b))"
)


def _load_preprocessed_ldba(
    fixture: str,
    formula: str,
    assignments: list[Assignment] | None = None,
) -> LDBA:
    hoa = (TESTDATA / fixture).read_text()
    ldba = HOAParser(formula, hoa).parse_hoa()
    assignments = assignments or Assignment.zero_or_one_propositions(
        set(ldba.propositions)
    )
    ldba.prune(assignments)
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


def _assignments(*propositions: str) -> frozenset[Assignment]:
    return frozenset(Assignment(proposition) for proposition in propositions)


def test_compute_sequences_for_configured_finite_formula() -> None:
    # conf/formulas/letter/finite.yaml
    ldba = _load_preprocessed_ldba("letter_finite.hoa", FINITE_FORMULA)

    state_to_sequences = compute_sequences(ldba, num_loops=3)

    expected = ReachAvoidSequence(
        [
            (_assignments("b"), _assignments("a")),
            (_assignments("d"), _assignments("c")),
            (_assignments("f"), _assignments("e")),
        ]
    )
    assert ldba.initial_state is not None
    assert ldba.sink_state is not None
    assert state_to_sequences[ldba.initial_state] == [expected]
    assert state_to_sequences[1] == [ReachAvoidSequence([])]
    assert state_to_sequences[ldba.sink_state] == []
    # Finite specifications do not repeat their accepting loop.
    assert len(state_to_sequences[ldba.initial_state][0]) == 3


def test_compute_sequences_for_configured_infinite_formula() -> None:
    # conf/formulas/letter/infinite.yaml
    ldba = _load_preprocessed_ldba("letter_infinite.hoa", INFINITE_FORMULA)
    unsafe = _assignments("e", "f")

    state_to_sequences = compute_sequences(ldba, num_loops=2)

    accepting_cycle = [
        (_assignments("b"), unsafe),
        (_assignments("d"), unsafe),
        (_assignments("a"), unsafe),
        (_assignments("c"), unsafe),
    ]
    expected = ReachAvoidSequence(accepting_cycle * 2)
    assert ldba.initial_state is not None
    assert ldba.sink_state is not None
    assert state_to_sequences[ldba.initial_state] == [expected]
    assert state_to_sequences[ldba.sink_state] == []
    assert all(
        len(sequences) == 1
        for state, sequences in state_to_sequences.items()
        if state != ldba.sink_state
    )


def test_compute_sequences_for_warehouse_disjunctive_formula() -> None:
    assignments = WarehouseEnv.assignments()
    ldba = _load_preprocessed_ldba(
        "warehouse_disjunctive.hoa",
        WAREHOUSE_FORMULA,
        assignments,
    )

    def matching(
        *, present: tuple[str, ...] = (), absent: tuple[str, ...] = ()
    ) -> frozenset[Assignment]:
        return frozenset(
            assignment
            for assignment in assignments
            if all(proposition in assignment for proposition in present)
            and all(proposition not in assignment for proposition in absent)
        )

    empty = frozenset()
    crate_only_region = matching(present=("crate",), absent=("door", "region_b"))
    no_crate_or_target_region = matching(absent=("crate", "door", "region_b"))
    crate_in_region_b = matching(present=("crate", "region_b"))
    no_crate_in_region_b = matching(present=("region_b",), absent=("crate",))
    door = matching(present=("door",))
    door_without_crate = matching(present=("door",), absent=("crate",))
    crate_at_door = matching(present=("crate", "door"))
    region_b = matching(present=("region_b",))

    def sequence(*steps) -> ReachAvoidSequence:
        return ReachAvoidSequence(steps)

    state_to_sequences = compute_sequences(ldba)

    assert ldba.sink_state is None
    assert state_to_sequences == {
        0: [
            sequence(
                (crate_only_region, empty),
                (no_crate_or_target_region, no_crate_in_region_b),
                (door, empty),
                (region_b, empty),
            ),
            sequence(
                (crate_in_region_b, empty),
                (no_crate_in_region_b, no_crate_or_target_region),
            ),
            sequence((door_without_crate, empty), (region_b, empty)),
            sequence((crate_at_door, empty), (region_b, empty)),
        ],
        1: [sequence()],
        2: [
            sequence(
                (no_crate_or_target_region, empty),
                (door, empty),
                (region_b, empty),
            ),
            sequence(
                (crate_in_region_b, empty),
                (no_crate_in_region_b, crate_only_region),
            ),
            sequence((door_without_crate, empty), (region_b, empty)),
            sequence((crate_at_door, empty), (region_b, empty)),
        ],
        3: [
            sequence(
                (crate_only_region, empty),
                (no_crate_or_target_region, crate_in_region_b),
                (door, empty),
                (region_b, empty),
            ),
            sequence((no_crate_in_region_b, empty)),
            sequence((door_without_crate, empty), (region_b, empty)),
            sequence((crate_at_door, empty), (region_b, empty)),
        ],
        4: [sequence((region_b, empty))],
        5: [sequence((region_b, empty))],
        6: [sequence((door, empty), (region_b, empty))],
        7: [sequence((region_b, empty))],
    }
