from pathlib import Path

from jaxolotl.alg.curriculum import (
    Curriculum,
    CurriculumStage,
    MultiRandomStage,
    RandomCurriculumStage,
    SampleBatcher,
)
from jaxolotl.alg.struct_ltl.curriculum.boolean_samplers import (
    BooleanReachAvoidSampler,
    BooleanReachStaySampler,
)
from jaxolotl.alg.struct_ltl.curriculum.formula_cache import FormulaCache
from jaxolotl.alg.struct_ltl.utils.batching import BooleanSequenceBatcher
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.environments.zone_env.zone_env import ZoneEnv

propositions = ZoneEnv.propositions
assignments = ZoneEnv.assignments()
cache = FormulaCache(propositions, assignments)


def make_stages(env: Environment | EnvWrapper) -> list[CurriculumStage]:
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
        # 5. Reach-avoid / reach-stay tasks
        MultiRandomStage(
            stages=[
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
                RandomCurriculumStage(
                    sampler=BooleanReachStaySampler(
                        num_stay=30,
                        reach_formulas=cache.props,
                        avoid_formulas=cache.props,
                        avoid_prob=0.5,
                        assignments=env.assignments(),
                    ),
                    threshold=None,
                ),
            ],
            probs=[0.4, 0.6],
            threshold=0.85,
        ),
        # 6. More complex reach-avoid / reach-stay tasks
        MultiRandomStage(
            stages=[
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
                RandomCurriculumStage(
                    sampler=BooleanReachStaySampler(
                        num_stay=60,
                        reach_formulas=cache.props,
                        avoid_formulas=cache.props,
                        avoid_prob=0.5,
                        assignments=env.assignments(),
                    ),
                    threshold=None,
                ),
            ],
            probs=[0.8, 0.2],
            threshold=0.85,
        ),
        # 7. Final mixture of complex tasks
        MultiRandomStage(
            stages=[
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
                RandomCurriculumStage(
                    sampler=BooleanReachStaySampler(
                        num_stay=60,
                        reach_formulas=cache.props,
                        avoid_formulas=cache.props + cache.ors,
                        avoid_prob=0.5,
                        assignments=env.assignments(),
                    ),
                    threshold=None,
                ),
            ],
            probs=[0.8, 0.2],
            threshold=None,
        ),
    ]


def make(
    env: Environment | EnvWrapper,
    load_path: Path | None = None,
    batcher: SampleBatcher | None = None,
    ablation: bool = False,
) -> Curriculum:
    stages = make_stages(env)
    if ablation:
        stages = stages[-1:]
    return Curriculum(
        stages,
        num_samples=int(1e3),
        batcher=BooleanSequenceBatcher() if batcher is None else batcher,
        env=env,
        load_path=load_path,
    )
