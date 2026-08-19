"""Batching utilities for tokenized Boolean reach-avoid sequences."""

from typing import override

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl2action.curriculum.curriculum import SampleBatcher
from jaxolotl.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.struct_ltl.reach_avoid.jax_tokenized_reach_avoid_sequence import (
    JaxTokenizedReachAvoidSequence,
)


class TokenizedSequenceBatcher(
    SampleBatcher[BooleanReachAvoidSequence, JaxTokenizedReachAvoidSequence]
):
    """Batches BooleanReachAvoidSequences into a JaxTokenizedReachAvoidSequence."""

    @override
    @staticmethod
    def batch(
        samples: list[BooleanReachAvoidSequence],
        env: Environment | EnvWrapper,
    ) -> JaxTokenizedReachAvoidSequence:
        return JaxTokenizedReachAvoidSequence.from_reach_avoid_seqs(samples, env)
