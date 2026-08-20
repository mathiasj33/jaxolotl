from pathlib import Path

import jaxolotl
from jaxolotl import eqx_utils
from jaxolotl.alg.curriculum import (
    Curriculum,
    RandomCurriculumStage,
)
from jaxolotl.alg.ltl2action.curriculum.simple_samplers import (
    SimpleReachAvoidFormulaSampler,
)
from jaxolotl.alg.ltl2action.eval.batching import FormulaClosureBatcher
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


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
        ],
        num_samples=10_000,
        batcher=FormulaClosureBatcher(),
        env=env,
        load_path=load_path,
    )


if __name__ == "__main__":
    env, _ = jaxolotl.make("ZoneEnv")
    curriculum = make(env)
    print(curriculum)
    print(eqx_utils.compute_size(curriculum) / 2**20, "MB")
