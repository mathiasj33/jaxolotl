import random
from typing import override

from jaxolotl.alg.deep_ltl.reach_avoid.reach_avoid_sequence import (
    EPSILON,
    AssignmentSet,
    EpsilonType,
    ReachAvoidSequence,
)
from jaxolotl.alg.ltl2action.curriculum.curriculum import Sampler
from jaxolotl.ltl.logic.assignment import Assignment


class SimpleReachAvoidSampler(Sampler[ReachAvoidSequence]):
    """Samples simple reach-avoid sequences. Assumes that no propositions can be true at
    the same time.
    """

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        assignments: list[Assignment],
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
        self.assignments = assignments

        for assignment in assignments:
            for other in assignments:
                if assignment != other and not assignment.true_propositions.isdisjoint(
                    other.true_propositions
                ):
                    raise ValueError(
                        "All assignments must be mutually exclusive for "
                        "SimpleReachAvoidSampler."
                    )

    @override
    def sample(self) -> ReachAvoidSequence:
        depth = random.randint(*self.depth)
        last_reach: set[Assignment] = set()
        seq: list[tuple[AssignmentSet, AssignmentSet]] = []

        for _ in range(depth):
            # sample reach
            available_reach = [
                a
                for a in self.assignments
                if a not in last_reach and a.true_propositions
            ]
            reach_lower = min(self.reach[0], len(available_reach))
            reach_upper = min(self.reach[1], len(available_reach))
            num_reach = random.randint(reach_lower, reach_upper)
            reach = set(random.sample(available_reach, num_reach))

            # sample avoid
            available_avoid = [
                a
                for a in self.assignments
                if a not in reach and a not in last_reach and a.true_propositions
            ]
            avoid_lower = min(self.avoid[0], len(available_avoid))
            avoid_upper = min(self.avoid[1], len(available_avoid))
            num_avoid = random.randint(avoid_lower, avoid_upper)
            avoid = set(random.sample(available_avoid, num_avoid))

            last_reach = reach
            seq.append((frozenset(reach), frozenset(avoid)))

        return ReachAvoidSequence(seq)


class SimpleReachStaySampler(Sampler[ReachAvoidSequence]):
    """Samples simple reach-stay sequences."""

    def __init__(
        self,
        num_stay: int,
        avoid: int | tuple[int, int],
        assignments: list[Assignment],
    ):
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.avoid = avoid
        self.num_stay = num_stay  # the number of timesteps to stay after reaching
        self.assignments = assignments

    def sample(self) -> ReachAvoidSequence:
        # sample reach
        available_reach = [a for a in self.assignments if a.true_propositions]
        reach = random.choice(available_reach)

        # sample avoid
        available_avoid = [
            a for a in self.assignments if a != reach and a.true_propositions
        ]
        avoid_lower = min(self.avoid[0], len(available_avoid))
        avoid_upper = min(self.avoid[1], len(available_avoid))
        num_avoid = random.randint(avoid_lower, avoid_upper)
        avoid = set(random.sample(available_avoid, num_avoid))

        # avoid everything except reach
        avoid_all_else = set(self.assignments) - {reach}

        # build sequence
        seq: list[tuple[AssignmentSet | EpsilonType, AssignmentSet]] = []
        seq.append((EPSILON, frozenset(avoid)))
        seq.append((frozenset({reach}), frozenset(avoid_all_else)))
        seq.append((frozenset({reach}), frozenset(avoid_all_else)))

        return ReachAvoidSequence(seq, repeat_last=self.num_stay)
