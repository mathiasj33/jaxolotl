from pathlib import Path

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.environments.zone_env.zone_env import ZoneEnv
from jaxolotl.ltl2action.curriculum.curriculum import (
    Curriculum,
    MultiRandomStage,
    RandomCurriculumStage,
)
from jaxolotl.struct_ltl.curriculum.boolean_samplers import (
    BooleanReachAvoidSampler,
    BooleanReachStaySampler,
)
from jaxolotl.struct_ltl.curriculum.formula_cache import FormulaCache
from jaxolotl.struct_ltl.utils.batching import BooleanSequenceBatcher

propositions = ZoneEnv.propositions
assignments = ZoneEnv.assignments()
cache = FormulaCache(propositions, assignments)


def make(env: Environment | EnvWrapper, load_path: Path | None = None) -> Curriculum:
    return Curriculum(
        [
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
        ],
        num_samples=int(1e3),
        batcher=BooleanSequenceBatcher(),
        env=env,
        load_path=load_path,
    )
