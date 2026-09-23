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
    cache = FormulaCache(propositions, assignments, disj_length=3)
    return [
        RandomCurriculumStage(
            sampler=BooleanReachAvoidSampler(
                depth=1,
                reach_formulas=cache.props,
                avoid_formulas=cache.props + cache.ors,
                avoid_prob=0.9,
                assignments=env.assignments(),
            ),
            threshold=0.9,
        ),
        RandomCurriculumStage(
            sampler=BooleanReachAvoidSampler(
                depth=2,
                reach_formulas=cache.props,
                avoid_formulas=cache.props + cache.ors,
                avoid_prob=0.6,
                assignments=env.assignments(),
            ),
            threshold=0.80,
        ),
        RandomCurriculumStage(
            sampler=BooleanReachAvoidSampler(
                depth=3,
                reach_formulas=cache.props,
                avoid_formulas=cache.props + cache.ors,
                avoid_prob=0.6,
                assignments=env.assignments(),
            ),
            threshold=None,
        ),
    ]
