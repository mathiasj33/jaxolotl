"""Script to visualize an environment using a renderer. Supports teleoperation or random actions."""

import hydra
from omegaconf import DictConfig

import jaxolotl
from jaxolotl.environments.renderer.renderer import BaseRenderer
from jaxolotl.environments.wrappers.auto_reset_wrapper import (
    AutoResetWrapper,
    ResetStrategy,
)
from jaxolotl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxolotl.hydra_utils.utils import resolve_default_options


@hydra.main(version_base="1.3", config_path="../conf", config_name="visualize_env")
def main(cfg: DictConfig):
    default_options = resolve_default_options(cfg.env)

    env, params = jaxolotl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        env = PrecomputedResetWrapper(
            env,
            params,
            jaxolotl.DATA_DIR / cfg.env.name / cfg.env.precomputed_resets_path,
        )
    env = AutoResetWrapper(
        env, reset_strategy=ResetStrategy.FULL, auto_reset_options=default_options
    )

    renderer: BaseRenderer = env.get_renderer(params)
    renderer.run_render_loop(
        env,
        params,
        policy=cfg.policy,
        print_debug=cfg.print_debug,
    )


if __name__ == "__main__":
    main()
