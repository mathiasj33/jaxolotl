from jaxolotl.alg.curriculum import (
    MultiRandomStage,
    RandomCurriculumStage,
)
from jaxolotl.alg.curriculum.curriculum import CurriculumStage
from jaxolotl.alg.ltl2action.curriculum.simple_samplers import (
    SimpleReachAvoidFormulaSampler,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    return [
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
    ]
