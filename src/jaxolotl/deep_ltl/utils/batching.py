from typing import override

from jaxolotl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxolotl.deep_ltl.reach_avoid.reach_avoid_sequence import ReachAvoidSequence
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl2action.curriculum.curriculum import SampleBatcher


class ReachAvoidSequenceBatcher(
    SampleBatcher[ReachAvoidSequence, JaxReachAvoidSequence]
):
    """Batches reach-avoid sequences into a JaxReachAvoidSequence."""

    @override
    @staticmethod
    def batch(
        samples: list[ReachAvoidSequence],
        env: Environment | EnvWrapper,
    ) -> JaxReachAvoidSequence:
        return JaxReachAvoidSequence.from_reach_avoid_seqs(samples, env)
