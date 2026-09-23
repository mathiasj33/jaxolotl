import random

from jaxolotl.alg.curriculum import (
    CurriculumStage,
)
from jaxolotl.alg.curriculum.curriculum import RandomCurriculumStage, Sampler
from jaxolotl.alg.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import VarNode


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    return [
        RandomCurriculumStage(
            sampler=ConveyorSequenceSampler(
                propositions=env.propositions, assignments=env.assignments()
            ),
            threshold=None,
        )
    ]


class ConveyorSequenceSampler(Sampler[BooleanReachAvoidSequence]):
    """Samples the two ConveyorWorld reach-chains with equal probability.

    The chains share all propositions except the final one: the first
    ``len(propositions) - 2`` propositions form the shared prefix and the last
    two are the corridor-specific finals.
    """

    def __init__(
        self, propositions: tuple[str, ...], assignments: list[Assignment]
    ) -> None:
        self._shared = [VarNode(prop) for prop in propositions[:-2]]
        self._finals = [VarNode(prop) for prop in propositions[-2:]]
        self._assignments = assignments

    def sample(self, rng: random.Random) -> BooleanReachAvoidSequence:
        final = self._finals[0] if rng.random() < 0.5 else self._finals[1]
        seq = [(prop, None) for prop in [*self._shared, final]]
        return BooleanReachAvoidSequence(seq, assignments=self._assignments)
