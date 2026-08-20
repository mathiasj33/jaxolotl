from collections.abc import Callable

import hydra
from omegaconf import DictConfig

from jaxolotl import DATA_DIR
from jaxolotl.alg.curriculum.wrapper import CurriculumWrapper
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def wrap_for_training(
    env: Environment | EnvWrapper,
    cfg: DictConfig,
    task_wrapper: Callable[[Environment | EnvWrapper], EnvWrapper] | None = None,
) -> CurriculumWrapper:
    """Load a configured curriculum and install the training wrappers.

    The curriculum is built against the incoming environment. An optional
    algorithm-specific task wrapper is then placed inside CurriculumWrapper.
    """
    load_path = DATA_DIR / cfg.env.name / cfg.alg.name / "curriculum.eqx"
    curriculum = hydra.utils.call(cfg.curriculum, env, load_path)
    if task_wrapper is not None:
        env = task_wrapper(env)
    return CurriculumWrapper(env, curriculum, cfg.curriculum_wrapper.episode_window)
