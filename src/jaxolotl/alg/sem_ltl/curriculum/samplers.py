import random
from collections.abc import Sequence
from typing import override

from jaxolotl.alg.curriculum.curriculum import Sampler
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import BooleanNode, FalseNode
from jaxolotl.ltl.logic.utils import compute_sat


class BooleanReachStayFormulaSampler(Sampler[str]):
    """Sample feasible Boolean reach-and-remain objectives as LTL formulae."""

    def __init__(
        self,
        reach_formulas: Sequence[BooleanNode],
        avoid_formulas: Sequence[BooleanNode],
        assignments: Sequence[Assignment],
        avoid_prob: float = 0.5,
    ):
        if not reach_formulas:
            raise ValueError("At least one reach formula must be provided.")
        self.reach_formulas = reach_formulas
        self.avoid_formulas = avoid_formulas
        self.assignments = tuple(assignments)
        self.avoid_prob = avoid_prob

    @override
    def sample(self, rng: random.Random) -> str:
        reach = rng.choice(self.reach_formulas)
        reach_sat = compute_sat(reach, self.assignments)
        available_avoid = [
            formula
            for formula in self.avoid_formulas
            if not reach_sat.issubset(compute_sat(formula, self.assignments))
        ]
        if not available_avoid or rng.random() > self.avoid_prob:
            avoid = FalseNode()
        else:
            avoid = rng.choice(available_avoid)

        if isinstance(avoid, FalseNode):
            return f"FG ({reach})"
        return f"(!({avoid}) U G({reach}))"


class SimpleGFSampler(Sampler[str]):
    """Samples GF formulas."""

    def __init__(
        self,
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(reach, int):
            reach = (reach, reach)
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.reach = reach
        self.avoid = avoid
        self.propositions = propositions

    def sample(self, rng: random.Random) -> str:
        reach = rng.randint(self.reach[0], self.reach[1])
        avoid = rng.randint(self.avoid[0], self.avoid[1])
        reach_props = rng.sample(self.propositions, reach)
        available_props = [p for p in self.propositions if p not in reach_props]
        avoid_props = rng.sample(available_props, min(avoid, len(available_props)))
        formula = " & ".join(f"GF {p}" for p in reach_props)
        if avoid_props:
            formula += " & G(!(" + " | ".join(avoid_props) + "))"
        return formula

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SimpleGFSampler):
            return False
        return (
            self.reach == value.reach
            and self.avoid == value.avoid
            and set(self.propositions) == set(value.propositions)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.reach,
                self.avoid,
                tuple(sorted(self.propositions)),
            )
        )


class SimpleFGSampler(Sampler[str]):
    """Samples FG formulas."""

    def __init__(
        self,
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.avoid = avoid
        self.propositions = propositions

    def sample(self, rng: random.Random) -> str:
        reach_prop = rng.choice(self.propositions)
        avoid = rng.randint(self.avoid[0], self.avoid[1])
        available_props = [p for p in self.propositions if p != reach_prop]
        avoid_props = rng.sample(available_props, min(avoid, len(available_props)))
        formula = f"FG {reach_prop}"
        if avoid_props:
            formula += " & G(!(" + " | ".join(avoid_props) + "))"
        return formula

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SimpleFGSampler):
            return False
        return self.avoid == value.avoid and set(self.propositions) == set(
            value.propositions
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.avoid,
                tuple(sorted(self.propositions)),
            )
        )
