"""Collection of samplers for sequences of Boolean formulae."""

import random
from collections.abc import Sequence
from typing import override

from jaxolotl.alg.curriculum import Sampler
from jaxolotl.alg.struct_ltl.curriculum.formula_cache import FormulaCache
from jaxolotl.alg.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.environments.zone_env.zone_env import ZoneEnv
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import (
    BooleanNode,
    FalseNode,
    NotNode,
)
from jaxolotl.ltl.logic.utils import compute_sat, push_down_nots
from jaxolotl.ltl.reach_avoid.sequence import EPSILON


class BooleanReachAvoidSampler(Sampler[BooleanReachAvoidSequence]):
    """Samples reach-avoid sequences of boolean formulae. Ensures that sequences are feasible."""

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
    def sample(self) -> BooleanReachAvoidSequence:
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

        return BooleanReachAvoidSequence(reach_avoid, self.assignments)


class BooleanReachStaySampler(Sampler[BooleanReachAvoidSequence]):
    """Samples reach-stay sequences of boolean formulae. Ensures that sequences are feasible."""

    def __init__(
        self,
        num_stay: int,
        reach_formulas: Sequence[BooleanNode],
        avoid_formulas: Sequence[BooleanNode],
        assignments: Sequence[Assignment],
        avoid_prob: float = 0.5,
    ):
        self.num_stay = num_stay
        self.reach_formulas = reach_formulas
        self.avoid_formulas = avoid_formulas
        self.assignments = tuple(assignments)
        self.avoid_prob = avoid_prob
        if not avoid_formulas:
            raise ValueError("At least one avoid formula must be provided.")

    @override
    def sample(self) -> BooleanReachAvoidSequence:
        # 1. Sample Reach Formula
        reach = random.choice(self.reach_formulas)
        reach_sat = compute_sat(reach, self.assignments)

        # 2. Sample Avoid Formula
        available_avoid = [
            f
            for f in self.avoid_formulas
            if not reach_sat.issubset(compute_sat(f, self.assignments))
        ]
        if not available_avoid or random.random() > self.avoid_prob:
            avoid = FalseNode()
        else:
            avoid = random.choice(available_avoid)

        avoid_except_reach = push_down_nots(NotNode(reach))
        seq = [
            (EPSILON, avoid),
            (reach, avoid_except_reach),
            (reach, avoid_except_reach),
        ]
        return BooleanReachAvoidSequence(
            seq, self.assignments, repeat_last=self.num_stay
        )


if __name__ == "__main__":
    # propositions = WarehouseEnv.propositions
    # assignments = WarehouseEnv.assignments()
    # cache = FormulaCache(propositions, assignments)
    # print(cache.and_nots)
    # sampler = BooleanReachAvoidSampler(
    #     depth=(1, 3),
    #     reach_formulas=cache.props + cache.ands + cache.and_nots + cache.ors,
    #     avoid_formulas=cache.props + cache.ands + cache.and_nots,
    #     avoid_prob=0.5,
    #     assignments=assignments,
    # )
    # samples = [sampler.sample() for _ in range(100)]
    # num_unique = len(set(samples))
    # print(f"Num unique: {num_unique}")
    # for seq in samples[:10]:
    #     print(seq.reach_avoid_formulas)
    #     print(seq.clauses)

    props = ZoneEnv.propositions
    assignments = ZoneEnv.assignments()
    cache = FormulaCache(props, assignments)
    sampler = BooleanReachStaySampler(
        num_stay=5,
        reach_formulas=cache.props,
        avoid_formulas=cache.props + cache.ors,
        avoid_prob=0.5,
        assignments=assignments,
    )
    samples = [sampler.sample() for _ in range(100)]
    num_unique = len(set(samples))
    print(f"Num unique: {num_unique}")
    for seq in samples[:10]:
        print(seq.clauses)
