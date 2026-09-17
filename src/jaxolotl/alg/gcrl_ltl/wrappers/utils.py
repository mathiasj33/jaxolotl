from omegaconf import DictConfig

from jaxolotl.alg.curriculum.factory import wrap_for_training
from jaxolotl.alg.deep_ltl.wrappers.ldba_wrapper import LDBAWrapper
from jaxolotl.alg.gcrl_ltl.wrappers.goal_wrapper import GoalWrapper
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> Environment | EnvWrapper:
    if training:
        env = wrap_for_training(env, cfg, GoalWrapper)
    else:
        finite = cfg.get("eval", {}).get("finite", False)
        env = LDBAWrapper(env, overwrite_finite=finite)
    return env
