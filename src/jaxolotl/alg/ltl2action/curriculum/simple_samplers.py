import random
from collections.abc import Sequence
from typing import override

from jaxolotl.alg.curriculum import Sampler
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import BooleanNode, FalseNode
from jaxolotl.ltl.logic.utils import compute_sat


class SimpleReachAvoidFormulaSampler(Sampler[str]):
    """Samples simple reach-avoid formulas."""

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(depth, int):
            depth = (depth, depth)
        if isinstance(reach, int):
            reach = (reach, reach)
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.depth = depth
        self.reach = reach
        self.avoid = avoid
        self.propositions = propositions

    def sample(self) -> str:
        depth = random.randint(self.depth[0], self.depth[1])
        props = []
        last_props = set()
        for _ in range(depth):
            nr = random.randint(self.reach[0], self.reach[1])
            na = random.randint(self.avoid[0], self.avoid[1])
            available_props = [p for p in self.propositions if p not in last_props]
            reach_props = random.sample(available_props, min(nr, len(available_props)))
            available_props = [
                p
                for p in available_props
                if p not in reach_props and p not in last_props
            ]
            avoid_props = random.sample(available_props, min(na, len(available_props)))
            props.append((reach_props, avoid_props))
            last_props = set(reach_props)
        formula = "true"
        for reach_props, avoid_props in reversed(props):
            if not avoid_props:
                formula = f"F(({' | '.join(reach_props)}) & {formula})"
            else:
                formula = f"(!({' | '.join(avoid_props)}) U ({' | '.join(reach_props)} & {formula}))"
        return formula

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SimpleReachAvoidFormulaSampler):
            return False
        return (
            self.depth == value.depth
            and self.reach == value.reach
            and self.avoid == value.avoid
            and set(self.propositions) == set(value.propositions)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.depth,
                self.reach,
                self.avoid,
                tuple(sorted(self.propositions)),
            )
        )


class BooleanReachAvoidFormulaSampler(Sampler[str]):
    """Samples reach-avoid formulae of complex boolean formulae. Ensures feasibility."""

    depth: tuple[int, int]
    reach_formulas: Sequence[BooleanNode]
    avoid_formulas: Sequence[BooleanNode]
    avoid_prob: float

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach_formulas: Sequence[BooleanNode],
        avoid_formulas: Sequence[BooleanNode],
        assignments: Sequence[Assignment],
        avoid_prob: float = 0.5,
    ):
        if isinstance(depth, int):
            depth = (depth, depth)
        self.depth = depth
        self.reach_formulas = reach_formulas
        self.avoid_formulas = avoid_formulas
        self.avoid_prob = avoid_prob
        if not reach_formulas:
            raise ValueError("At least one reach formula must be provided.")
        self.assignments = tuple(assignments)

    @override
    def sample(self) -> str:
        depth = random.randint(self.depth[0], self.depth[1])

        last_reach_sat = None
        reach_avoid = []

        for _ in range(depth):
            # 1. Sample Reach Formula
            available_reach = [
                f
                for f in self.reach_formulas
                if not last_reach_sat
                or not compute_sat(f, self.assignments).issubset(last_reach_sat)
            ]
            available_reach = (
                available_reach if available_reach else self.reach_formulas
            )
            reach = random.choice(available_reach)
            reach_sat = compute_sat(reach, self.assignments)

            # 2. Sample Avoid Formula
            available_avoid = [
                f
                for f in self.avoid_formulas
                if not reach_sat.issubset(compute_sat(f, self.assignments))
                and (
                    not last_reach_sat
                    or not last_reach_sat.issubset(compute_sat(f, self.assignments))
                )
            ]
            if not available_avoid or random.random() > self.avoid_prob:
                avoid = FalseNode()
            else:
                avoid = random.choice(available_avoid)

            last_reach_sat = reach_sat
            reach_avoid.append((reach, avoid))

        formula = "true"
        for reach, avoid in reversed(reach_avoid):
            if isinstance(avoid, FalseNode):
                formula = f"F(({str(reach)}) & {formula})"
            else:
                formula = f"(!({str(avoid)}) U ({str(reach)} & {formula}))"
        return formula
