from jaxolotl.alg.curriculum.curriculum import CurriculumStage, RandomCurriculumStage
from jaxolotl.alg.deep_ltl.curriculum.simple_samplers import SimpleReachAvoidSampler
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    return [
        # 1. Simple reach-avoid tasks
        RandomCurriculumStage(
            sampler=SimpleReachAvoidSampler(
                depth=1,
                reach=1,
                avoid=(1, 2),
                assignments=env.assignments(),
            ),
            threshold=0.9,
        ),
        # 2. Depth 2 tasks
        RandomCurriculumStage(
            sampler=SimpleReachAvoidSampler(
                depth=2,
                reach=1,
                avoid=(0, 2),
                assignments=env.assignments(),
            ),
            threshold=0.9,
        ),
        # 3. Depth 3 tasks
        RandomCurriculumStage(
            sampler=SimpleReachAvoidSampler(
                depth=3,
                reach=(1, 2),
                avoid=(0, 3),
                assignments=env.assignments(),
            ),
            threshold=None,
        ),
    ]
