from jaxolotl.alg.curriculum import (
    CurriculumStage,
    MultiRandomStage,
    RandomCurriculumStage,
)
from jaxolotl.alg.ltl2action.curriculum.simple_samplers import (
    BooleanReachAvoidFormulaSampler,
)
from jaxolotl.alg.sem_ltl.curriculum.samplers import BooleanReachStayFormulaSampler
from jaxolotl.alg.struct_ltl.curriculum.formula_cache import FormulaCache
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv

propositions = WarehouseEnv.propositions
assignments = WarehouseEnv.assignments()
cache = FormulaCache(propositions, assignments)
all_except_or_formulas = cache.props + cache.ands + cache.and_nots
all_formulas = all_except_or_formulas + cache.ors


def make_stages(_env: Environment) -> list[CurriculumStage]:
    """Build the SemLTL warehouse curriculum as LTL formula samplers.

    The stages and sampling probabilities mirror StructLTL's warehouse curriculum.
    Reach-stay sequences are expressed as infinite LTL reach-and-remain objectives.
    """
    return [
        # 1. Reach propositions individually
        RandomCurriculumStage(
            sampler=BooleanReachAvoidFormulaSampler(
                depth=1,
                reach_formulas=cache.props,
                avoid_formulas=[],
                avoid_prob=0.0,
                assignments=assignments,
            ),
            threshold=0.9,
        ),
        # 2. Reach combinations of propositions (no avoids)
        RandomCurriculumStage(
            sampler=BooleanReachAvoidFormulaSampler(
                depth=1,
                reach_formulas=all_except_or_formulas,
                avoid_formulas=[],
                avoid_prob=0.0,
                assignments=assignments,
            ),
            threshold=0.95,
        ),
        # 3. Reach combinations of depth 2
        RandomCurriculumStage(
            sampler=BooleanReachAvoidFormulaSampler(
                depth=2,
                reach_formulas=all_except_or_formulas,
                avoid_formulas=[],
                avoid_prob=0.0,
                assignments=assignments,
            ),
            threshold=0.9,
        ),
        # 4. Introduce avoids
        RandomCurriculumStage(
            sampler=BooleanReachAvoidFormulaSampler(
                depth=1,
                reach_formulas=all_except_or_formulas,
                avoid_formulas=all_except_or_formulas,
                avoid_prob=0.5,
                assignments=assignments,
            ),
            threshold=0.95,
        ),
        # 5. Reach depth 2 with avoids
        RandomCurriculumStage(
            sampler=BooleanReachAvoidFormulaSampler(
                depth=(1, 2),
                reach_formulas=all_except_or_formulas,
                avoid_formulas=all_except_or_formulas,
                avoid_prob=0.5,
                assignments=assignments,
            ),
            threshold=0.95,
        ),
        # 6. Mixed reach-avoid and reach-stay
        MultiRandomStage(
            stages=[
                RandomCurriculumStage(
                    sampler=BooleanReachAvoidFormulaSampler(
                        depth=(1, 2),
                        reach_formulas=all_except_or_formulas,
                        avoid_formulas=all_formulas,
                        avoid_prob=0.5,
                        assignments=assignments,
                    ),
                    threshold=None,
                ),
                RandomCurriculumStage(
                    sampler=BooleanReachStayFormulaSampler(
                        reach_formulas=all_except_or_formulas,
                        avoid_formulas=all_except_or_formulas,
                        avoid_prob=0.2,
                        assignments=assignments,
                    ),
                    threshold=None,
                ),
            ],
            probs=[0.4, 0.6],
            threshold=0.9,
        ),
        # 7. More complex mixed stage
        MultiRandomStage(
            stages=[
                RandomCurriculumStage(
                    sampler=BooleanReachAvoidFormulaSampler(
                        depth=(1, 2),
                        reach_formulas=all_except_or_formulas,
                        avoid_formulas=all_formulas,
                        avoid_prob=0.5,
                        assignments=assignments,
                    ),
                    threshold=None,
                ),
                RandomCurriculumStage(
                    sampler=BooleanReachStayFormulaSampler(
                        reach_formulas=all_except_or_formulas,
                        avoid_formulas=all_formulas,
                        avoid_prob=0.5,
                        assignments=assignments,
                    ),
                    threshold=None,
                ),
            ],
            probs=[0.8, 0.2],
            threshold=None,
        ),
    ]
