from pathlib import Path

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl2action.curriculum.curriculum import (
    Curriculum,
    MultiRandomStage,
    RandomCurriculumStage,
)
from jaxolotl.ltl2action.curriculum.simple_samplers import (
    SimpleReachAvoidFormulaSampler,
)
from jaxolotl.ltl2action.eval.batching import FormulaClosureBatcher


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    return Curriculum(
        stages=[
            MultiRandomStage(
                [
                    RandomCurriculumStage(
                        SimpleReachAvoidFormulaSampler(
                            depth=(1, 3),
                            reach=1,
                            avoid=0,
                            propositions=list(env.propositions),
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        SimpleReachAvoidFormulaSampler(
                            depth=(1, 2),
                            reach=1,
                            avoid=1,
                            propositions=list(env.propositions),
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.5, 0.5],
                threshold=None,
            )
        ],
        num_samples=10_000,
        batcher=FormulaClosureBatcher(),
        env=env,
        load_path=load_path,
    )
