import hydra
from omegaconf import DictConfig

from jaxolotl import DATA_DIR
from jaxolotl.alg.ltl2action.wrappers.curriculum_wrapper import CurriculumWrapper
from jaxolotl.alg.ltl2action.wrappers.formula_closure_wrapper import (
    FormulaClosureWrapper,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper:
    env = FormulaClosureWrapper(env)
    if training:
        precomputed_curriculum_path = (
            DATA_DIR / cfg.env.name / cfg.alg.name / "curriculum.eqx"
        )
        curriculum = hydra.utils.call(cfg.curriculum, env, precomputed_curriculum_path)
        env = CurriculumWrapper(env, curriculum, cfg.curriculum_wrapper.episode_window)
    return env
