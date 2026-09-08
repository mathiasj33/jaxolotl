"""Generic curriculum for GenZ-LTL.

GenZ-LTL samples random reach-avoid subgoals throughout training and does not require
multiple curriculum stages. We implement this approach as a single RandomCurriculumStage
that samples random reach-avoid subgoals.
"""

import random

from jaxolotl.alg.curriculum import (
    RandomCurriculumStage,
    Sampler,
)
from jaxolotl.alg.curriculum.curriculum import CurriculumStage
from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    ReachAvoidSubgoal,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.logic.assignment import Assignment


class SubgoalSampler(Sampler[ReachAvoidSubgoal]):
    def __init__(self, assignments: list[Assignment]):
        self.assignments = assignments

    def sample(self, rng: random.Random) -> ReachAvoidSubgoal:
        assignments = [assignment for assignment in self.assignments if assignment]
        reach = rng.choice(assignments)
        available = [assignment for assignment in assignments if assignment != reach]
        # Uniformly sample an avoid subset
        avoid = [assignment for assignment in available if rng.random() < 0.5]
        return ReachAvoidSubgoal(reach=reach, avoid=avoid)


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    return [
        RandomCurriculumStage(
            sampler=SubgoalSampler(env.assignments()),
            threshold=None,
        )
    ]
