from jaxolotl.alg.curriculum.curriculum import (
    CurriculumStage,
    MultiRandomStage,
    RandomCurriculumStage,
)
from jaxolotl.alg.ltl2action.curriculum.simple_samplers import (
    SimpleReachAvoidFormulaSampler,
)
from jaxolotl.alg.sem_ltl.curriculum.samplers import SimpleGFSampler
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    return [
        # 1. Simple reach tasks
        RandomCurriculumStage(
            sampler=SimpleReachAvoidFormulaSampler(
                depth=1,
                reach=1,
                avoid=0,
                propositions=list(env.propositions),
            ),
            threshold=0.9,
        ),
        # 2. Reach tasks of depth 2
        RandomCurriculumStage(
            sampler=SimpleReachAvoidFormulaSampler(
                depth=2,
                reach=1,
                avoid=0,
                propositions=list(env.propositions),
            ),
            threshold=0.95,
        ),
        # 3. Simple reach-avoid tasks
        RandomCurriculumStage(
            sampler=SimpleReachAvoidFormulaSampler(
                depth=1,
                reach=1,
                avoid=1,
                propositions=list(env.propositions),
            ),
            threshold=0.95,
        ),
        # 4. Reach-avoid tasks of depth 2
        RandomCurriculumStage(
            sampler=SimpleReachAvoidFormulaSampler(
                depth=2,
                reach=1,
                avoid=1,
                propositions=list(env.propositions),
            ),
            threshold=0.9,
        ),
        # 5. Final mix
        MultiRandomStage(
            [
                RandomCurriculumStage(
                    sampler=SimpleReachAvoidFormulaSampler(
                        depth=(1, 2),
                        reach=(1, 2),
                        avoid=(0, 2),
                        propositions=list(env.propositions),
                    ),
                    threshold=None,
                ),
                RandomCurriculumStage(
                    sampler=SimpleGFSampler(
                        reach=(2, 3),
                        avoid=(0, 2),
                        propositions=list(env.propositions),
                    ),
                    threshold=None,
                ),
            ],
            probs=[2 / 3, 1 / 3],
            threshold=None,
        ),
    ]
