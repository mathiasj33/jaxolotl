"""Curriculum for the tokenized LTL ablation study on WarehouseEnv.

This uses the same samplers as the struct_ltl curriculum but with tokenized
sequence representations instead of structured clause representations.
"""

from pathlib import Path

from jaxolotl.alg.curriculum import (
    Curriculum,
    MultiRandomStage,
    RandomCurriculumStage,
)
from jaxolotl.alg.struct_ltl.curriculum.boolean_samplers import (
    BooleanReachAvoidSampler,
    BooleanReachStaySampler,
)
from jaxolotl.alg.struct_ltl.curriculum.formula_cache import FormulaCache
from jaxolotl.alg.struct_ltl.utils.tokenized_batching import TokenizedSequenceBatcher
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.warehouse_env.warehouse_env import WarehouseEnv
from jaxolotl.environments.wrappers.wrapper import EnvWrapper

propositions = WarehouseEnv.propositions
assignments = WarehouseEnv.assignments()
cache = FormulaCache(propositions, assignments)
all_except_or_formulas = cache.props + cache.ands + cache.and_nots
all_formulas = all_except_or_formulas + cache.ors


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    """Create a curriculum for the tokenized LTL ablation study on WarehouseEnv.

    This curriculum uses the same task distribution as the struct_ltl curriculum
    but represents formulas as token sequences rather than structured clauses.
    """
    return Curriculum(
        [
            # 1. Reach propositions individually
            RandomCurriculumStage(
                sampler=BooleanReachAvoidSampler(
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
                sampler=BooleanReachAvoidSampler(
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
                sampler=BooleanReachAvoidSampler(
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
                sampler=BooleanReachAvoidSampler(
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
                sampler=BooleanReachAvoidSampler(
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
                            avoid_formulas=all_except_or_formulas,
                            avoid_prob=0.2,
                            num_stay=30,
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
        batcher=TokenizedSequenceBatcher(),
        env=env,
        load_path=load_path,
    )
