from omegaconf import DictConfig

from jaxolotl.alg.curriculum import wrap_for_training
from jaxolotl.alg.sem_ltl.wrappers.semantic_ldba_wrapper import SemanticLDBAWrapper
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.normalize_reward_wrapper import (
    NormalizeRewardWrapper,
)
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper | Environment:
    if training:
        env = wrap_for_training(env, cfg, SemanticLDBAWrapper)
        env = NormalizeRewardWrapper(env, gamma=cfg.rl_alg.gamma)
    else:
        finite = cfg.get("eval", {}).get("finite", False)
        env = SemanticLDBAWrapper(env, overwrite_finite=finite)
    return env
