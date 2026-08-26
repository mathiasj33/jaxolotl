"""Script to visualize an environment using a renderer. Supports teleoperation or random actions."""

import hydra
from omegaconf import DictConfig

import jaxolotl
from jaxolotl.environments.renderer.renderer import BaseRenderer
from jaxolotl.environments.wrappers.auto_reset_wrapper import (
    AutoResetWrapper,
    ResetStrategy,
)


@hydra.main(version_base="1.3", config_path="../conf", config_name="visualize_env")
def main(cfg: DictConfig):
    env, params = jaxolotl.make(
        cfg.env.name, reset_source=cfg.env.get("reset_source", "test")
    )
    env = AutoResetWrapper(env, reset_strategy=ResetStrategy.FULL)

    renderer: BaseRenderer = env.get_renderer(params)
    renderer.run_render_loop(
        env,
        params,
        policy=cfg.policy,
        print_debug=cfg.print_debug,
    )


if __name__ == "__main__":
    main()
