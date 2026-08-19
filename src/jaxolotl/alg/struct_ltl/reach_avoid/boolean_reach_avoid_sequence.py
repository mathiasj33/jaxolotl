from collections.abc import Iterable
from typing import TYPE_CHECKING

from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import BooleanNode, MultiOrNode, OrNode
from jaxolotl.ltl.logic.utils import (
    Clause,
    compute_sat,
    formula_to_clauses,
    synthesize_formula,
)
from jaxolotl.ltl.reach_avoid.sequence import (
    EPSILON,
    AssignmentSet,
    EpsilonType,
    ReachAvoidSequence,
)

if TYPE_CHECKING:
    from jaxolotl.environments.environment import Environment
    from jaxolotl.environments.wrappers.wrapper import EnvWrapper


class BooleanReachAvoidSequence(ReachAvoidSequence):
    """A reach-avoid sequence of boolean formulas to reach and avoid. Stores the
    formula graphs, sets of clauses, and satisfying assignment sets."""

    def __init__(
        self,
        reach_avoid: list[tuple[BooleanNode | EpsilonType | None, BooleanNode | None]],
        assignments: Iterable[Assignment],
        repeat_last: int = 0,
    ):
        super().__init__(
            self._compute_sat_assignments(reach_avoid, tuple(assignments)), repeat_last
        )
        self.reach_avoid_formulas = tuple(reach_avoid)
        clauses: list[tuple[list[Clause] | EpsilonType, list[Clause]]] = []
        for reach_graph, avoid_graph in reach_avoid:
            if isinstance(reach_graph, EpsilonType):
                reach_clauses = EPSILON
            else:
                reach_clauses = formula_to_clauses(reach_graph)
            avoid_clauses = formula_to_clauses(avoid_graph)
            clauses.append((reach_clauses, avoid_clauses))
        self.clauses = tuple(clauses)
        self.assignments = tuple(assignments)

    @staticmethod
    def _compute_sat_assignments(
        reach_avoid: list[tuple[BooleanNode | EpsilonType | None, BooleanNode | None]],
        assignments: tuple[Assignment, ...],
    ) -> list[tuple[AssignmentSet | EpsilonType, AssignmentSet]]:
        results = []
        for reach, avoid in reach_avoid:
            if reach is None:
                reach_set = frozenset()
            elif isinstance(reach, EpsilonType):
                reach_set = EPSILON
            else:
                reach_set = compute_sat(reach, assignments)
            avoid_set = (
                frozenset() if avoid is None else compute_sat(avoid, assignments)
            )
            results.append((reach_set, avoid_set))
        return results

    # TODO: write test for this!!!
    def expand_clauses(self) -> list["BooleanReachAvoidSequence"]:
        """Expands the reach-avoid sequence into multiple sequences based on clause combinations."""
        formulas = list(self.reach_avoid_formulas)
        if len(formulas) == 0:
            return [
                BooleanReachAvoidSequence(
                    [], self.assignments, repeat_last=self.repeat_last
                )
            ]
        rest = BooleanReachAvoidSequence(
            formulas[1:], self.assignments, repeat_last=self.repeat_last
        )
        rec = rest.expand_clauses()
        expanded = []
        reach, avoid = formulas[0]
        if not isinstance(reach, MultiOrNode | OrNode):
            new_ra = [(reach, avoid)]
            for r in rec:
                expanded.append(
                    BooleanReachAvoidSequence(
                        new_ra + list(r.reach_avoid_formulas),
                        self.assignments,
                        repeat_last=self.repeat_last,
                    )
                )
            return expanded

        assert not isinstance(reach, OrNode), "OrNode not yet supported"

        for clause in reach.operands:
            new_ra = [(clause, avoid)]
            for r in rec:
                expanded.append(
                    BooleanReachAvoidSequence(
                        new_ra + list(r.reach_avoid_formulas),
                        self.assignments,
                        repeat_last=self.repeat_last,
                    )
                )
        return expanded

    @classmethod
    def from_reach_avoid_sequence(
        cls,
        sequence: ReachAvoidSequence,
        env: "Environment | EnvWrapper",
    ) -> "BooleanReachAvoidSequence":
        """Creates a BooleanReachAvoidSequence from a ReachAvoidSequence. Synthesises
        formulas from assignment sets with Quine-McCluskey's algorithm."""
        reach_avoid = []
        assignments = frozenset(env.assignments())
        props = env.propositions
        for reach_set, avoid_set in sequence.reach_avoid:
            if isinstance(reach_set, EpsilonType):
                reach_graph = EPSILON
            else:
                reach_graph = synthesize_formula(reach_set, assignments, props)
            avoid_graph = synthesize_formula(avoid_set, assignments, props)
            reach_avoid.append((reach_graph, avoid_graph))

        return cls(
            reach_avoid,
            assignments=tuple(env.assignments()),
            repeat_last=sequence.repeat_last,
        )

    def __hash__(self):
        return hash((self.reach_avoid, self.repeat_last, self.assignments))

    def __eq__(self, other):
        if not isinstance(other, BooleanReachAvoidSequence):
            return False
        return (
            self.reach_avoid == other.reach_avoid
            and self.repeat_last == other.repeat_last
            and self.assignments == other.assignments
        )

    def __iter__(self):
        return iter(self.reach_avoid)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start >= len(self.reach_avoid):
                if self.repeat_last <= 0:
                    return []
                return [self.reach_avoid[-1]]
            return [self.reach_avoid[item]]
        if item >= len(self.reach_avoid):
            if self.repeat_last <= 0:
                raise IndexError
            return self.reach_avoid[-1]
        return self.reach_avoid[item]
