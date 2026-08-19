import hydra
from omegaconf import DictConfig

from jaxolotl import DATA_DIR
from jaxolotl.alg.deep_ltl.wrappers.ldba_wrapper import LDBAWrapper
from jaxolotl.alg.deep_ltl.wrappers.sequence_wrapper import SequenceWrapper
from jaxolotl.alg.ltl2action.wrappers.curriculum_wrapper import CurriculumWrapper
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper | Environment:
    if training:
        precomputed_curriculum_path = (
            DATA_DIR / cfg.env.name / cfg.alg.name / "curriculum.eqx"
        )
        curriculum = hydra.utils.call(cfg.curriculum, env, precomputed_curriculum_path)
        env = SequenceWrapper(env)
        env = CurriculumWrapper(env, curriculum, cfg.curriculum_wrapper.episode_window)
    else:
        finite = cfg.get("eval", {}).get("finite", False)
        env = LDBAWrapper(env, overwrite_finite=finite)
    return env
