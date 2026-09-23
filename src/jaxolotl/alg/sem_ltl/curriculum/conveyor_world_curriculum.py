import random

from jaxolotl.alg.curriculum.curriculum import (
    CurriculumStage,
    RandomCurriculumStage,
    Sampler,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    return [
        RandomCurriculumStage(
            sampler=ConveyorFormulaSampler(propositions=env.propositions),
            threshold=None,
        ),
    ]


def chain_formula(propositions: list[str] | tuple[str, ...]) -> str:
    """Builds the nested reach-chain formula ``F (p0 & F (p1 & ... F pn))``."""
    if len(propositions) == 1:
        return f"F {propositions[0]}"
    return f"F ({propositions[0]} & {chain_formula(propositions[1:])})"


class ConveyorFormulaSampler(Sampler[str]):
    """Samples the two ConveyorWorld reach-chain formulas with equal probability.

    The chains share all propositions except the final one: the first
    ``len(propositions) - 2`` propositions form the shared prefix and the last
    two are the corridor-specific finals.
    """

    def __init__(self, propositions: tuple[str, ...]) -> None:
        shared = list(propositions[:-2])
        self._formulas = [
            chain_formula([*shared, final]) for final in propositions[-2:]
        ]

    def sample(self, rng: random.Random) -> str:
        if rng.random() < 0.5:
            return self._formulas[0]
        else:
            return self._formulas[1]
