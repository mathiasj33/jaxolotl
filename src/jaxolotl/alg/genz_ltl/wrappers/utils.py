from omegaconf import DictConfig

from jaxolotl.alg.curriculum import wrap_for_training
from jaxolotl.alg.genz_ltl.wrappers.ldba_subgoal_wrapper import LDBASubgoalWrapper
from jaxolotl.alg.genz_ltl.wrappers.subgoal_wrapper import SubgoalWrapper
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def wrap_env(
    env: Environment | EnvWrapper,
    cfg: DictConfig,
    training: bool,
    terminate_on_success: bool = False,
) -> EnvWrapper | Environment:
    if training:
        env = wrap_for_training(
            env,
            cfg,
            lambda env: SubgoalWrapper(env, terminate_on_success=terminate_on_success),
        )
    else:
        finite = cfg.get("eval", {}).get("finite", False)
        env = LDBASubgoalWrapper(env, overwrite_finite=finite)
    return env
