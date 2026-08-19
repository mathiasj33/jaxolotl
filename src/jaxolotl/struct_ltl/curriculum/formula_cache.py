import itertools
from collections.abc import Sequence

from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import (
    BooleanNode,
    MultiAndNode,
    MultiOrNode,
    NotNode,
    VarNode,
)
from jaxolotl.ltl.logic.utils import compute_sat


class FormulaCache:
    """
    Cache that stores pre-computed Boolean formulas as graphs. Useful for Boolean samplers.
    """

    def __init__(
        self,
        propositions: Sequence[str],
        assignments: Sequence[Assignment],
        conj_length: int = 3,
        disj_length: int = 2,
    ):
        """Initializes the cache and pre-computes common formulas.

        Args:
            propositions: List of proposition names.
            assignments: List of all possible assignments.
            conj_length: Maximum length of conjunctions to pre-compute.
            disj_length: Maximum length of disjunctions to pre-compute.
        """
        self.propositions = propositions
        self.assignments = tuple(assignments)
        # stores sets of assignments for which a formula has already been computed
        self.cache: set[frozenset[Assignment]] = set()
        self.props, self.ands, self.and_nots, self.ors = self._compute_formulas(
            conj_length, disj_length
        )

    def _compute_formulas(self, conj_length: int, disj_length: int):
        """Pre-computes and caches common Boolean formulas."""
        props = self._filter_and_cache([VarNode(p) for p in self.propositions])
        ands = self._filter_and_cache(
            [
                MultiAndNode(nodes)
                for r in range(2, conj_length + 1)
                for nodes in itertools.combinations(props, r)
            ]
        )
        and_nots = self._filter_and_cache(
            [MultiAndNode([x, NotNode(y)]) for x, y in itertools.combinations(props, 2)]
        )
        ors = self._filter_and_cache(
            [
                MultiOrNode(nodes)
                for r in range(2, disj_length + 1)
                for nodes in itertools.combinations(props + ands + and_nots, r)
            ]
        )
        return props, ands, and_nots, ors

    def _filter_and_cache(self, graphs: Sequence[BooleanNode]) -> list[BooleanNode]:
        """Filters out graphs that are already in the cache or have no satisfying
        assignments. Caches the satisfying assignments of new graphs.
        """
        filtered = []
        for g in graphs:
            sats = compute_sat(g, self.assignments)
            if sats and sats not in self.cache:
                self.cache.add(sats)
                filtered.append(g)
        return filtered
