import random

from jaxolotl.alg.curriculum.curriculum import (
    CurriculumStage,
    RandomCurriculumStage,
    Sampler,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    del env  # Unused.
    return [
        RandomCurriculumStage(
            sampler=ConveyorFormulaSampler(),
            threshold=None,
        ),
    ]


class ConveyorFormulaSampler(Sampler[str]):
    """Samples formulas specific to the conveyor world."""

    def sample(self, rng: random.Random) -> str:
        if rng.random() < 0.5:
            return "F(parcel & (F wrench))"
        else:
            return "F(parcel & (F hammer))"
