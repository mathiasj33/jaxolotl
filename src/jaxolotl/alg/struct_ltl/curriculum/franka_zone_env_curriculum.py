from jaxolotl.alg.curriculum import (
    CurriculumStage,
    RandomCurriculumStage,
)
from jaxolotl.alg.struct_ltl.curriculum.boolean_samplers import (
    BooleanReachAvoidSampler,
)
from jaxolotl.alg.struct_ltl.curriculum.formula_cache import FormulaCache
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
    propositions = env.propositions
    assignments = env.assignments()
    cache = FormulaCache(propositions, assignments)

    return [
        # 1. Simple reach tasks
        RandomCurriculumStage(
            sampler=BooleanReachAvoidSampler(
                depth=1,
                reach_formulas=cache.props,
                avoid_formulas=[],
                avoid_prob=0.0,
                assignments=env.assignments(),
            ),
            threshold=0.9,
        ),
        # 2. Reach tasks of depth 2
        RandomCurriculumStage(
            sampler=BooleanReachAvoidSampler(
                depth=2,
                reach_formulas=cache.props,
                avoid_formulas=[],
                avoid_prob=0.0,
                assignments=env.assignments(),
            ),
            threshold=0.80,
        ),
        # 3. Simple reach-avoid tasks
        RandomCurriculumStage(
            sampler=BooleanReachAvoidSampler(
                depth=1,
                reach_formulas=cache.props,
                avoid_formulas=cache.props,
                avoid_prob=1.0,
                assignments=env.assignments(),
            ),
            threshold=0.95,
        ),
        # 4. Reach-avoid tasks of depth 2
        RandomCurriculumStage(
            sampler=BooleanReachAvoidSampler(
                depth=2,
                reach_formulas=cache.props,
                avoid_formulas=cache.props,
                avoid_prob=1.0,
                assignments=env.assignments(),
            ),
            threshold=0.80,
        ),
        # 5. Final mixture of reach-avoid tasks
        RandomCurriculumStage(
            sampler=BooleanReachAvoidSampler(
                depth=(1, 2),
                reach_formulas=cache.props,
                avoid_formulas=cache.props + cache.ors,
                avoid_prob=0.7,
                assignments=env.assignments(),
            ),
            threshold=None,
        ),
    ]
