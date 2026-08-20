from omegaconf import DictConfig

from jaxolotl.alg.curriculum import wrap_for_training
from jaxolotl.alg.deep_ltl.wrappers.ldba_wrapper import LDBAWrapper
from jaxolotl.alg.deep_ltl.wrappers.sequence_wrapper import SequenceWrapper
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper | Environment:
    if training:
        env = wrap_for_training(env, cfg, SequenceWrapper)
    else:
        finite = cfg.get("eval", {}).get("finite", False)
        env = LDBAWrapper(env, overwrite_finite=finite)
    return env
