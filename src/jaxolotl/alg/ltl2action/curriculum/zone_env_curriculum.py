from jaxolotl.alg.curriculum import (
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
    ]
