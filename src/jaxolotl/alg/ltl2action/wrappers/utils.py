from omegaconf import DictConfig

from jaxolotl.alg.curriculum import wrap_for_training
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
        env = wrap_for_training(env, cfg)
    return env
