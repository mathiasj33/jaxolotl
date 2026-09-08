import random

from jaxolotl.alg.genz_ltl.curriculum.generic_curriculum import SubgoalSampler
from jaxolotl.ltl.logic.assignment import Assignment


def _assignments():
    return [
        Assignment("a"),
        Assignment("b"),
        Assignment("c"),
        Assignment("d"),
        Assignment(),
    ]


def test_subgoal_sampling_is_deterministic_and_excludes_empty_assignment():
    sampler = SubgoalSampler(_assignments())
    first_rng = random.Random(42)
    second_rng = random.Random(42)

    first = [sampler.sample(first_rng) for _ in range(100)]
    second = [sampler.sample(second_rng) for _ in range(100)]

    assert first == second
    assert all(sample.reach for sample in first)
    assert all(sample.reach not in sample.avoid for sample in first)
    assert all(all(assignment for assignment in sample.avoid) for sample in first)
