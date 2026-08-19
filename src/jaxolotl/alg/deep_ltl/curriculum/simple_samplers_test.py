from jaxolotl.alg.deep_ltl.curriculum.simple_samplers import (
    SimpleReachAvoidSampler,
    SimpleReachStaySampler,
)
from jaxolotl.alg.deep_ltl.reach_avoid.reach_avoid_sequence import EPSILON, EpsilonType
from jaxolotl.ltl.logic.assignment import Assignment


def test_simple_reach_avoid_sampler():
    sampler = SimpleReachAvoidSampler(
        depth=(3, 8),
        reach=(1, 3),
        avoid=(0, 2),
        assignments=Assignment.zero_or_one_propositions({"a", "b", "c", "d"}),
    )
    for j in range(1000):
        seq = sampler.sample()

        if j < 5:
            print(f"\nSampled Reach-Avoid Sequence ({j}):\n{seq}")

        # Check depth constraints
        assert sampler.depth[0] <= len(seq) <= sampler.depth[1]

        for i, (reach, avoid) in enumerate(seq.reach_avoid):
            assert not isinstance(reach, EpsilonType)
            assert not isinstance(avoid, EpsilonType)

            # Check reach set size constraints
            assert sampler.reach[0] <= len(reach) <= sampler.reach[1]

            # Check avoid set size constraints
            assert sampler.avoid[0] <= len(avoid) <= sampler.avoid[1]

            # Check reach-avoid disjointness
            assert reach.isdisjoint(avoid)

            # Check reach-last reach disjointness
            if i > 0:
                last_reach_set = seq.reach_avoid[i - 1][0]
                assert not isinstance(last_reach_set, EpsilonType) and reach.isdisjoint(
                    last_reach_set
                )

            # Check that empty assignment is not included
            assert all(len(a.true_propositions) > 0 for a in reach)
            assert all(len(a.true_propositions) > 0 for a in avoid)


def test_reach_stay_sampler():
    assignments = Assignment.zero_or_one_propositions({"a", "b", "c", "d"})
    sampler = SimpleReachStaySampler(
        num_stay=30,
        avoid=(0, 2),
        assignments=assignments,
    )
    for j in range(1000):
        seq = sampler.sample()

        if j < 5:
            print(f"\nSampled Reach-Stay Sequence ({j}):\n{seq}")

        assert seq.reach_avoid[0][0] == EPSILON

        # Check avoid set at first step
        avoid_first = seq.reach_avoid[0][1]
        assert sampler.avoid[0] <= len(avoid_first) <= sampler.avoid[1]

        # Check reach-stay steps
        reach_prop = seq.reach_avoid[1][0]
        for i in range(1, 3):
            reach_step, avoid_step = seq.reach_avoid[i]

            # Check reach proposition is maintained
            assert reach_step == reach_prop

            # Check avoid set excludes reached proposition
            assert set(avoid_step) == set(assignments) - reach_prop  # type: ignore
