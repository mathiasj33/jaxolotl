"""Compute evaluation curves by evaluating model checkpoints over time on a fixed set of
formulas. Saves results to a CSV file for later plotting (see scripts/plotting/plot_eval_curves.py).
"""

import csv
import logging
import os
import time

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from jaxtyping import PyTree
from omegaconf import DictConfig
from tqdm import tqdm

import jaxolotl
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_model_checkpoints, make_eval_fn

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../conf", config_name="eval_curves")
def main(cfg: DictConfig):
    # build environment
    env, env_params = jaxolotl.make(
        cfg.env.name, reset_source=cfg.get("reset_source", "test")
    )
    env = TimeLimitWrapper(env)
    env = hydra.utils.call(cfg.alg.wrap_env, env, cfg, training=False)
    env = VectorizeWrapper(env)

    # preprocess formulas
    formulas: PyTree = hydra.utils.call(cfg.alg.preprocess_formulas, cfg.formulas, env)
    logger.info(f"Processed {len(cfg.formulas)} formulas.")

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models, num_seeds, checkpoint_steps = load_model_checkpoints(
        cfg, env, env_params, key=model_key
    )
    agents = hydra.utils.instantiate(cfg.alg.agent, models)

    # set up evaluator
    eval_fn = make_eval_fn(cfg, num_seeds, len(cfg.formulas), return_trajs=False)

    # evaluate
    logger.info("Starting evaluation...")
    params, static = eqx.partition(agents, eqx.is_array)
    pbar = tqdm(total=len(checkpoint_steps), desc="Evaluating checkpoints", leave=False)

    def update_progress():
        pbar.update(1)  # important: do not return anything

    def eval_timestep(key, agent_params):
        agent = eqx.combine(agent_params, static)
        key, eval_key = jax.random.split(key)
        returns, disc_returns, lengths, _, _ = eval_fn(
            agent,
            env,
            env_params,
            formulas,
            eval_key,
        )  # shape: (num_formulas, num_seeds, num_episodes)
        io_callback(update_progress, None)
        return key, (returns, disc_returns, lengths)

    start = time.time()
    _, (returns, disc_returns, lengths) = jax.lax.scan(eval_timestep, key, params)
    # shape: (num_checkpoints, num_seeds, num_formulas, num_episodes)
    jax.block_until_ready(returns)
    pbar.close()
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # log to stdout and save to CSV
    save_results(cfg, returns, disc_returns, lengths, checkpoint_steps)


def save_results(
    cfg: DictConfig,
    returns: jax.Array,
    disc_returns: jax.Array,
    lengths: jax.Array,
    checkpoint_steps: list[int],
):
    """Saves averaged results to a CSV file."""

    csv_path = (
        f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.run}/eval_results_checkpoints.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    num_seeds = int(returns.shape[1])
    seeds = list(range(num_seeds))

    fieldnames = [
        "seed",
        "deterministic",
        "timestep",
        "metric",
        "return",
        "length",
    ]

    rows = []
    for i, step in enumerate(checkpoint_steps):
        # Compute per-seed stats for this timestep
        # shapes: (num_checkpoints, num_seeds, num_formulas, num_episodes)

        mask = returns[i] > 0  # (num_seeds, num_formulas, num_episodes)
        counts = jnp.sum(mask, axis=(1, 2))  # Shape: (num_seeds,)

        def compute_masked_seed_means(data_tensor, i=i, mask=mask, counts=counts):
            # Sum values of successful trajectories only
            sum_val = jnp.sum(data_tensor[i] * mask, axis=(1, 2))
            # Avoid division by zero if a seed had 0 successes
            return jnp.where(counts > 0, sum_val / counts, jnp.nan)

        mean_returns, mean_disc_returns = jax.tree.map(
            lambda x, i=i: x[i].mean(axis=-1).mean(axis=-1),
            (returns, disc_returns),
        )
        mean_lengths = compute_masked_seed_means(lengths)

        # CSV rows (per-seed)
        for seed in seeds:
            rows.append(
                {
                    "seed": seed,
                    "deterministic": bool(cfg.eval.deterministic),
                    "timestep": step,
                    "metric": float(mean_returns[seed]),
                    "return": float(mean_disc_returns[seed]),
                    "length": float(mean_lengths[seed]),
                }
            )

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote results to {csv_path}")


if __name__ == "__main__":
    main()
