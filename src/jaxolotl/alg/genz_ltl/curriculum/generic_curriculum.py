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

    def sample(self) -> ReachAvoidSubgoal:
        reach = random.choice(self.assignments)
        available = [a for a in self.assignments if a != reach]
        num_avoid = random.randint(0, len(available))
        avoid = random.sample(available, num_avoid)
        return ReachAvoidSubgoal(reach=reach, avoid=avoid)


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    return [
        RandomCurriculumStage(
            sampler=SubgoalSampler(env.assignments()),
            threshold=None,
        )
    ]
