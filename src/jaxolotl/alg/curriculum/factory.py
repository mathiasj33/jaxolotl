"""Factory functions for building curriculum objects."""

from collections.abc import Callable
from pathlib import Path

import hydra
from omegaconf import DictConfig

from jaxolotl import DATA_DIR
from jaxolotl.alg.curriculum import Curriculum, SampleBatcher
from jaxolotl.alg.curriculum.curriculum import CurriculumStage
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
    stages: list[CurriculumStage] = hydra.utils.call(cfg.curriculum.make_stages, env)
    batcher: SampleBatcher = hydra.utils.instantiate(cfg.alg.batcher)
    curriculum = make_curriculum(
        env=env,
        stages=stages,
        num_samples=int(cfg.curriculum.num_samples),
        batcher=batcher,
        load_path=load_path,
        skip_curriculum=cfg.curriculum.get("skip_curriculum", False),
    )
    if task_wrapper is not None:
        env = task_wrapper(env)
    return CurriculumWrapper(env, curriculum, cfg.curriculum.episode_window)


def make_curriculum(
    env: Environment | EnvWrapper,
    stages: list[CurriculumStage],
    num_samples: int,
    batcher: SampleBatcher,
    load_path: Path | None = None,
    skip_curriculum: bool = False,
    num_parallel: int = 1,
) -> Curriculum:
    """Build a curriculum for the given environment."""
    if skip_curriculum:
        stages = stages[-1:]  # Only keep the last stage
    return Curriculum(
        stages=stages,
        batcher=batcher,
        env=env,
        num_samples=num_samples,
        load_path=load_path,
        num_parallel=num_parallel,
    )
