from pathlib import Path

from jaxolotl.alg.ltl2action.curriculum.curriculum import (
    Curriculum,
    MultiRandomStage,
    RandomCurriculumStage,
)
from jaxolotl.alg.struct_ltl.curriculum.boolean_samplers import (
    BooleanReachAvoidSampler,
    BooleanReachStaySampler,
)
from jaxolotl.alg.struct_ltl.curriculum.formula_cache import FormulaCache
from jaxolotl.alg.struct_ltl.utils.batching import BooleanSequenceBatcher
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxolotl.environments.wrappers.wrapper import EnvWrapper

propositions = WarehouseEnv.propositions
assignments = WarehouseEnv.assignments()
cache = FormulaCache(propositions, assignments)
all_except_or_formulas = cache.props + cache.ands + cache.and_nots
all_formulas = all_except_or_formulas + cache.ors


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    return Curriculum(
        [
            # 7. More complex mixed stage
            MultiRandomStage(
                stages=[
                    RandomCurriculumStage(
                        sampler=BooleanReachAvoidSampler(
                            depth=(1, 2),
                            reach_formulas=all_except_or_formulas,
                            avoid_formulas=all_formulas,
                            avoid_prob=0.5,
                            assignments=assignments,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=BooleanReachStaySampler(
                            reach_formulas=all_except_or_formulas,
                            avoid_formulas=all_formulas,
                            avoid_prob=0.5,
                            num_stay=60,
                            assignments=assignments,
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.8, 0.2],
                threshold=None,
            ),
        ],
        num_samples=int(1e4),
        batcher=BooleanSequenceBatcher(),
        env=env,
        load_path=load_path,
    )


if __name__ == "__main__":
    sampler = BooleanReachAvoidSampler(
        depth=(1, 2),
        reach_formulas=all_except_or_formulas,
        avoid_formulas=all_formulas,
        avoid_prob=0.5,
        assignments=assignments,
    )
    formulas = [sampler.sample() for _ in range(10)]
    for f in formulas:
        print(f.clauses)
        print(f.reach_avoid)
