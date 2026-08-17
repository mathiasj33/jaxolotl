"""Main training entry point.

Uses Hydra for configuration management. See conf/train.yaml, or run
`python train.py --help` for details.
"""

import datetime
import logging
import time
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import jaxltl
from jaxltl import DATA_DIR, eqx_utils
from jaxltl.environments.environment import EnvParams
from jaxltl.environments.spaces import Space
from jaxltl.environments.wrappers import AutoResetWrapper, LogWrapper, VectorizeWrapper
from jaxltl.environments.wrappers.auto_reset_wrapper import ResetStrategy
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxltl.eqx_utils.utils import compute_num_params
from jaxltl.hydra_utils.utils import resolve_default_options
from jaxltl.rl.actor_critic import ActorCritic
from jaxltl.rl.algorithm import RLAlgorithm

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    if not cfg.use_gpu:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        logger.info("Using CPU for training")

    default_options = resolve_default_options(cfg.env)

    env, env_params = jaxltl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        resets_path = f"{DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_resets_path}"
        env = PrecomputedResetWrapper(env, env_params, resets_path)
    env = TimeLimitWrapper(env)

    env = hydra.utils.call(cfg.alg.wrap_env, env, cfg, training=True)
    env = AutoResetWrapper(
        env, reset_strategy=ResetStrategy.FULL, auto_reset_options=default_options
    )
    env = LogWrapper(env)
    env = VectorizeWrapper(env)

    seeds = jnp.arange(cfg.start_seed, cfg.num_seeds)
    keys = jax.vmap(jax.random.key)(seeds)
    split = jax.vmap(jax.random.split)(keys)
    keys, model_keys = split[:, 0], split[:, 1]

    make_models = eqx.filter_vmap(
        build_model, in_axes=(None, None, None, None, None, None, 0)
    )
    models = make_models(
        cfg.model,
        env.observation_space(env_params).shape,
        env.action_space(env_params),
        len(env.assignments()),
        len(env.propositions),
        env_params,
        model_keys,
    )
    logger.info(
        f"Training model with {compute_num_params(models) / cfg.num_seeds / 1e3}k parameters."
    )

    rl_alg: RLAlgorithm = hydra.utils.instantiate(cfg.rl_alg)
    train = eqx.filter_vmap(
        rl_alg.train, in_axes=(eqx.if_array(0), None, None, 0, None, None, 0)
    )
    train = eqx.filter_jit(train)

    if cfg.wandb.use_wandb:
        import wandb  # noqa

        settings = wandb.Settings(silent=True)

        wandb_runs = [
            wandb.init(
                reinit="create_new",
                project=cfg.wandb.project,
                config=OmegaConf.to_container(cfg, resolve=True) | {"seed": seeds[i]},  # type: ignore
                name=f"{cfg.run}_{i}",
                settings=settings,
            )
            for i in range(cfg.num_seeds)
        ]
    else:
        wandb_runs = None

    logger.info("Compiling training function...")
    start_time = time.time()
    cb = make_callback(cfg, wandb_runs)
    compiled = train.lower(
        models, env, env_params, keys, cb, cfg.save_freq, seeds
    ).compile()
    logger.info(f"Compilation completed in {time.time() - start_time:.2f} seconds")

    logger.info("Starting training")
    cb = make_callback(cfg, wandb_runs)
    models = jax.block_until_ready(
        compiled(models, env, env_params, keys, cb, cfg.save_freq, seeds)
    )
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

    eqx_utils.save("models.eqx", models, metadata={"num_models": cfg.num_seeds})
    logger.info("Models saved to models.eqx")

    if wandb_runs is not None:
        for run in wandb_runs:
            run.finish()


def make_callback(cfg: DictConfig, wandb_runs: list | None = None):
    """Create a callback function to log progress and save model checkpoints."""

    start_time = time.time()

    def callback(
        metric: dict[str, jax.Array],
        model_params: jax.Array,
        seed: jax.Array,
        step: jax.Array,
    ):
        # estimate remaining training time
        seconds = time.time() - start_time
        sps = step / seconds
        remaining = int((cfg.rl_alg.total_timesteps - step) / sps)
        remaining = str(datetime.timedelta(seconds=remaining))

        # average returns
        window_returns = metric["episode_return"][metric["done"]][
            -cfg.curriculum_wrapper.episode_window :
        ]
        avg_returns = jnp.mean(window_returns)

        # positive returns
        window_pos_returns = metric["positive_return"][metric["done"]][
            -cfg.curriculum_wrapper.episode_window :
        ]
        avg_pos_returns = jnp.mean(window_pos_returns)

        window_stages = metric["curriculum_stage"][metric["done"]][
            -cfg.curriculum_wrapper.episode_window :
        ]
        avg_stage = jnp.mean(window_stages)
        min_stage = jnp.min(window_stages)
        max_stage = jnp.max(window_stages)

        # log progress
        logger.info(
            f"seed {seed} | step {step} | ret {avg_returns:.2f} | sr {avg_pos_returns:.2f} | stage {avg_stage:.2f} ({min_stage:}, {max_stage:}) | sps {int(sps)} | eta {remaining}"
        )

        # save checkpoint
        folder = Path("checkpoints")
        folder.mkdir(parents=True, exist_ok=True)
        filename = folder / f"model_seed{seed}_step{step}.eqx"
        eqx_utils.save(filename, model_params)

        # log to wandb
        if wandb_runs is not None:
            wandb = wandb_runs[int(seed)]
            wandb.log(
                {
                    "avg_return": float(avg_returns),
                    "avg_curriculum_stage": float(avg_stage),
                    "steps_per_second": int(sps),
                },
                step=int(step),
            )

        # log to csv
        return_values = metric["episode_return"][metric["done"]].tolist()
        pos_return_values = metric["positive_return"][metric["done"]].tolist()
        lengths = metric["episode_length"][metric["done"]].tolist()
        stages = metric["curriculum_stage"][metric["done"]].tolist()
        timesteps = (
            metric["total_step"][metric["done"]] * cfg.rl_alg.num_envs
        ).tolist()
        df = pd.DataFrame(
            {
                "timestep": timesteps,
                "return": return_values,
                "positive_return": pos_return_values,
                "length": lengths,
                "curriculum_stage": stages,
            }
        )
        df["seed"] = int(seed)
        df.to_csv(
            "logs.csv", mode="a", header=not Path("logs.csv").exists(), index=False
        )

    return callback


def build_model(
    model_cfg: DictConfig,
    obs_shape: tuple[int, ...],
    act_space: Space,
    num_assignments: int,
    num_propositions: int,
    env_params: EnvParams,
    key: jax.Array,
) -> ActorCritic:
    model_fn = hydra.utils.instantiate(
        model_cfg,
        obs_shape=obs_shape,
        num_assignments=num_assignments,
        num_propositions=num_propositions,
        env_params=env_params,
        key=key,
        _partial_=True,
    )
    # Ensure the space is not converted to an OmegaConf object
    model: ActorCritic = model_fn(act_space=act_space)
    return model


if __name__ == "__main__":
    main()
