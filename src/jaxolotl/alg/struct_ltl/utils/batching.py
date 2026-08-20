from typing import override

from jaxolotl.alg.curriculum import SampleBatcher
from jaxolotl.alg.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.alg.struct_ltl.reach_avoid.jax_clause_reach_avoid_sequence import (
    JaxClauseReachAvoidSequence,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


class BooleanSequenceBatcher(
    SampleBatcher[BooleanReachAvoidSequence, JaxClauseReachAvoidSequence]
):
    """Batches BooleanReachAvoidSequences into a JaxClauseReachAvoidSequence."""

    @override
    @staticmethod
    def batch(
        samples: list[BooleanReachAvoidSequence],
        env: Environment | EnvWrapper,
    ) -> JaxClauseReachAvoidSequence:
        return JaxClauseReachAvoidSequence.from_reach_avoid_seqs(samples, env)
