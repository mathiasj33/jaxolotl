import hydra
from omegaconf import DictConfig

from jaxolotl import DATA_DIR
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.genz_ltl.wrappers.ldba_subgoal_wrapper import LDBASubgoalWrapper
from jaxolotl.genz_ltl.wrappers.subgoal_wrapper import SubgoalWrapper
from jaxolotl.ltl2action.wrappers.curriculum_wrapper import CurriculumWrapper


def wrap_env(
    env: Environment | EnvWrapper, cfg: DictConfig, training: bool
) -> EnvWrapper | Environment:
    if training:
        precomputed_curriculum_path = (
            DATA_DIR / cfg.env.name / cfg.alg.name / "curriculum.eqx"
        )
        curriculum = hydra.utils.call(cfg.curriculum, env, precomputed_curriculum_path)
        env = SubgoalWrapper(env)
        env = CurriculumWrapper(env, curriculum, cfg.curriculum_wrapper.episode_window)
    else:
        finite = cfg.get("eval", {}).get("finite", False)
        env = LDBASubgoalWrapper(env, overwrite_finite=finite)
    return env
