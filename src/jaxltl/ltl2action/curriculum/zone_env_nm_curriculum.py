from pathlib import Path

import jaxltl
from jaxltl import eqx_utils
from jaxltl.environments.environment import Environment
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.ltl2action.curriculum.curriculum import (
    Curriculum,
    RandomCurriculumStage,
)
from jaxltl.ltl2action.curriculum.simple_samplers import (
    SimpleReachAvoidFormulaSampler,
)
from jaxltl.ltl2action.eval.batching import FormulaClosureBatcher


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    return Curriculum(
        stages=[
            RandomCurriculumStage(
                SimpleReachAvoidFormulaSampler(
                    depth=1,
                    reach=1,
                    avoid=(0, 1),
                    propositions=list(env.propositions),
                ),
                threshold=0.9,
            ),
            RandomCurriculumStage(
                SimpleReachAvoidFormulaSampler(
                    depth=(1, 2),
                    reach=1,
                    avoid=(0, 1),
                    propositions=list(env.propositions),
                ),
                threshold=0.85,
            ),
            RandomCurriculumStage(
                SimpleReachAvoidFormulaSampler(
                    depth=(1, 2),
                    reach=(1, 2),
                    avoid=(0, 1),
                    propositions=list(env.propositions),
                ),
                threshold=None,
            ),
            # RandomCurriculumStage(
            #     ZoneReachAvoidFormulaSampler(
            #         depth=2,
            #         reach=1,
            #         avoid=1,
            #         propositions=list(env.propositions),
            #     ),
            #     threshold=0.9,
            # ),
            # RandomCurriculumStage(
            #     ZoneReachAvoidFormulaSampler(
            #         depth=(1, 2),
            #         reach=(1, 2),
            #         avoid=(0, 2),
            #         propositions=list(env.propositions),
            #     ),
            #     threshold=None,
            # ),
        ],
        num_samples=10_000,
        batcher=FormulaClosureBatcher(),
        env=env,
        load_path=load_path,
    )


if __name__ == "__main__":
    env, _ = jaxltl.make("ZoneEnv")
    curriculum = make(env)
    print(curriculum)
    print(eqx_utils.compute_size(curriculum) / 2**20, "MB")
