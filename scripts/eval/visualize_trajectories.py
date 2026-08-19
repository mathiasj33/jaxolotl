"""Visualise trajectories of trained models on specified LTL formulas."""

import logging
import time
from functools import partial

import equinox as eqx
import hydra
import jax
from jax import numpy as jnp
from jaxtyping import PyTree
from omegaconf import DictConfig

import jaxltl
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxltl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxltl.eval.utils import load_batched_models, make_eval_fn

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="visualize_traj")
def main(cfg: DictConfig):
    # build environment
    env, env_params = jaxltl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        resets_path = (
            f"{jaxltl.DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_resets_path}"
        )
        env = PrecomputedResetWrapper(env, env_params, resets_path)
    env = TimeLimitWrapper(env)
    env = hydra.utils.call(cfg.alg.wrap_env, env, cfg, training=False)
    env = VectorizeWrapper(env)

    formulas: PyTree = hydra.utils.call(
        cfg.alg.preprocess_formulas, [cfg.eval.formula], env
    )

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models, _ = load_batched_models(cfg, env, env_params, key=model_key)

    # select single model from ensemble (while keeping batch dimension)
    params, static = eqx.partition(models, eqx.is_array)
    params = jax.tree.map(lambda x: x[cfg.eval.model_index], params)
    params = jax.tree.map(lambda x: x[None, ...], params)  # add batch dim
    model = eqx.combine(params, static)

    agent = hydra.utils.instantiate(cfg.alg.agent, model)

    # set up evaluator
    eval_fn = make_eval_fn(cfg, num_models=1, num_formulas=1, return_trajs=True)

    # evaluate
    key, eval_key = jax.random.split(key)
    logger.info("Starting evaluation...")
    start = time.time()
    returns, disc_returns, lengths, _, trajs = eval_fn(
        agent,
        env,
        env_params,
        formulas,
        eval_key,
    )  # shape: (1, 1, num_episodes)
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # plot trajectories
    trajs = jax.tree.map(partial(jnp.squeeze, axis=[0, 1]), trajs)
    lengths = jax.tree.map(partial(jnp.squeeze, axis=[0, 1]), lengths)

    if cfg.replay:
        if cfg.render.backend == "threejs":
            if env.name != "WarehouseEnv":
                raise ValueError("Three.js replay is only available for WarehouseEnv.")
            from jaxltl.environments.warehouse_env.threejs.replay import (  # noqa
                replay_trajectories,
            )

            output_dir = replay_trajectories(
                trajs,
                lengths,
                env_params,  # type: ignore
                frames_per_step=cfg.render.frames_per_step,
                pause_between_episodes=cfg.render.pause_between_episodes,
            )
            serve_cmd = f"pixi run python -m http.server 8000 --directory {output_dir}"
            logger.info("Three.js replay assets written to %s", output_dir)
            logger.info("Serve them with: %s", serve_cmd)
            logger.info("Then open: http://localhost:8000/")
            logger.info(
                "Alternatively, call pixi run threejs to serve and open in one step."
            )
        else:
            renderer = env.get_renderer(env_params)
            renderer.replay_trajectories(
                trajs,
                lengths,
                frames_per_step=cfg.render.frames_per_step,
                pause_between_episodes=cfg.render.pause_between_episodes,
            )
            renderer.close()

    env.plot_trajectories(
        trajs,
        lengths,
        env_params,
        num_cols=cfg.plotting.cols,
        num_rows=cfg.plotting.rows,
        save_path=cfg.plotting.save_path,
    )


if __name__ == "__main__":
    main()
