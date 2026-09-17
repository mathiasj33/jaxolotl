"""Generic curriculum for GCRL-LTL.

This consists of a single RandomCurriculumStage that samples random propositions to reach.
"""

import random

from jaxolotl.alg.curriculum import (
    RandomCurriculumStage,
    Sampler,
)
from jaxolotl.alg.curriculum.curriculum import CurriculumStage
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


class PropositionSampler(Sampler[str]):
    def __init__(self, propositions: list[str]):
        self.propositions = propositions

    def sample(self, rng: random.Random) -> str:
        return rng.choice(self.propositions)


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    return [
        RandomCurriculumStage(
            sampler=PropositionSampler(list(env.propositions)),
            threshold=None,
        )
    ]
