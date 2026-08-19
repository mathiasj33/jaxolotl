"""Batching utilities for graph-encoded Boolean reach-avoid sequences."""

from typing import override

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl2action.curriculum.curriculum import SampleBatcher
from jaxolotl.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.struct_ltl.reach_avoid.jax_clause_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)


class GraphSequenceBatcher(
    SampleBatcher[
        BooleanReachAvoidSequence,
        JaxGraphReachAvoidSequence,
    ]
):
    """Batches BooleanReachAvoidSequences into a JaxGraphReachAvoidSequence."""

    @override
    @staticmethod
    def batch(
        samples: list[BooleanReachAvoidSequence],
        env: Environment | EnvWrapper,
    ) -> JaxGraphReachAvoidSequence:
        return JaxGraphReachAvoidSequence.from_reach_avoid_seqs(samples, env)
