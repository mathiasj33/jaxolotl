import random
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxolotl.alg.curriculum import (
    Curriculum,
    RandomCurriculumStage,
    SampleBatcher,
    Sampler,
)
from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
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


class SubgoalBatcher(SampleBatcher[ReachAvoidSubgoal, JaxReachAvoidSubgoal]):
    @staticmethod
    def batch(
        samples: list[ReachAvoidSubgoal],
        env: Environment | EnvWrapper,
    ) -> JaxReachAvoidSubgoal:
        assignment_to_idx = {
            assignment: idx for idx, assignment in enumerate(env.assignments())
        }
        reach = [assignment_to_idx[s.reach] for s in samples]
        avoid = [[assignment_to_idx[a] for a in s.avoid] for s in samples]
        for a in avoid:
            a.extend([-1] * (len(env.assignments()) - len(a)))
        reach_one_hot = np.zeros((len(samples), len(env.propositions)), dtype=np.int32)
        avoid_one_hot = np.zeros((len(samples), len(env.assignments())), dtype=np.int32)
        for i, sample in enumerate(samples):
            for j, prop in enumerate(env.propositions):
                reach_one_hot[i, j] = 1 if prop in sample.reach else 0
            for j, assignment in enumerate(env.assignments()):
                avoid_one_hot[i, j] = 1 if assignment in sample.avoid else 0
        reach = jnp.array(reach, dtype=jnp.int32)
        avoid = jnp.array(avoid, dtype=jnp.int32)
        reach_one_hot = jnp.array(reach_one_hot)
        avoid_one_hot = jnp.array(avoid_one_hot)
        return JaxReachAvoidSubgoal(reach, avoid, reach_one_hot, avoid_one_hot)


def make(
    env: Environment | EnvWrapper, load_path: Path | None = None
) -> Curriculum[ReachAvoidSubgoal, JaxReachAvoidSubgoal]:
    return Curriculum(
        [
            RandomCurriculumStage(
                sampler=SubgoalSampler(env.assignments()),
                threshold=None,
            )
        ],
        num_samples=int(1e5),
        batcher=SubgoalBatcher(),
        env=env,
        load_path=load_path,
    )
