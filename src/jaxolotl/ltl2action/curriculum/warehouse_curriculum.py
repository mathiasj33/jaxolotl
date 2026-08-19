from pathlib import Path

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl2action.curriculum.curriculum import (
    Curriculum,
    RandomCurriculumStage,
)
from jaxolotl.ltl2action.curriculum.simple_samplers import (
    BooleanReachAvoidFormulaSampler,
    SimpleReachAvoidFormulaSampler,
)
from jaxolotl.ltl2action.eval.batching import FormulaClosureBatcher
from jaxolotl.struct_ltl.curriculum.formula_cache import FormulaCache

propositions = WarehouseEnv.propositions
assignments = WarehouseEnv.assignments()
cache = FormulaCache(propositions, assignments)
all_except_or_formulas = cache.props + cache.ands + cache.and_nots
all_formulas = all_except_or_formulas + cache.ors


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
                BooleanReachAvoidFormulaSampler(
                    depth=(1, 2),
                    reach_formulas=all_except_or_formulas,
                    avoid_formulas=all_except_or_formulas,
                    assignments=assignments,
                    avoid_prob=0.2,
                ),
                threshold=0.95,
            ),
            RandomCurriculumStage(
                BooleanReachAvoidFormulaSampler(
                    depth=(1, 2),
                    reach_formulas=all_except_or_formulas,
                    avoid_formulas=all_except_or_formulas,
                    assignments=assignments,
                    avoid_prob=0.5,
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
    sampler = BooleanReachAvoidFormulaSampler(
        depth=(1, 2),
        reach_formulas=all_except_or_formulas,
        avoid_formulas=all_except_or_formulas,
        assignments=assignments,
        avoid_prob=0.5,
    )
    formulas = [sampler.sample() for _ in range(10)]
    for f in formulas:
        print(f)
