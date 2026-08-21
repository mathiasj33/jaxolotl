from typing import override

from jaxolotl.alg.common.reach_avoid.jax_sequence import (
    JaxReachAvoidSequence,
)
from jaxolotl.alg.curriculum import SampleBatcher
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.reach_avoid.sequence import ReachAvoidSequence


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
