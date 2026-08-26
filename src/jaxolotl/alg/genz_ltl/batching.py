import jax.numpy as jnp
import numpy as np

from jaxolotl.alg.curriculum import (
    SampleBatcher,
)
from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
    ReachAvoidSubgoal,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


class SubgoalBatcher(SampleBatcher[ReachAvoidSubgoal, JaxReachAvoidSubgoal]):
    @staticmethod
    def batch(
        samples: list[ReachAvoidSubgoal],
        env: Environment | EnvWrapper,
        num_parallel: int = 1,
    ) -> JaxReachAvoidSubgoal:
        del num_parallel
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
