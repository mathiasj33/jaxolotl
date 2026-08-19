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
from jaxolotl import DATA_DIR
from jaxolotl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eval.utils import load_model_checkpoints, make_eval_fn

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval_curves")
def main(cfg: DictConfig):
    # build environment
    env, env_params = jaxolotl.make(cfg.env.name)
    if cfg.env.use_precomputed_resets:
        resets_path = (
            f"{jaxolotl.DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_resets_path}"
        )
        env = PrecomputedResetWrapper(env, env_params, resets_path)
    env = TimeLimitWrapper(env)
    env = hydra.utils.call(cfg.alg.wrap_env, env, cfg, training=False)
    env = VectorizeWrapper(env)

    # load and preprocess formulas
    logger.info("Processing formulas...")
    if hasattr(cfg, "formulas"):
        formula_strings = cfg.formulas
    else:
        formulas_file = DATA_DIR / cfg.env.name / "eval_formulas.txt"
        with open(formulas_file) as f:
            formula_strings = [line.strip() for line in f.readlines() if line.strip()]
    formulas: PyTree = hydra.utils.call(
        cfg.alg.preprocess_formulas, formula_strings, env
    )
    logger.info(f"Processed {len(formula_strings)} formulas.")

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models, num_seeds, checkpoint_steps = load_model_checkpoints(
        cfg, env, env_params, key=model_key
    )
    agents = hydra.utils.instantiate(cfg.alg.agent, models)

    # set up evaluator
    eval_fn = make_eval_fn(cfg, num_seeds, len(formula_strings), return_trajs=False)

    # evaluate
    logger.info("Starting evaluation...")
    params, static = eqx.partition(agents, eqx.is_array)
    pbar = tqdm(total=len(checkpoint_steps), desc="Evaluating checkpoints", leave=False)

    def update_progress():
        pbar.update(1)  # important: do not return anything

    def eval_timestep(key, agent_params):
        agent = eqx.combine(agent_params, static)
        key, eval_key = jax.random.split(key)
        returns, disc_returns, lengths, violations, _ = eval_fn(
            agent,
            env,
            env_params,
            formulas,
            eval_key,
        )  # shape: (num_formulas, num_seeds, num_episodes)
        io_callback(update_progress, None)
        return key, (returns, disc_returns, lengths, violations)

    start = time.time()
    _, (returns, disc_returns, lengths, violations) = jax.lax.scan(
        eval_timestep, key, params
    )
    # shape: (num_checkpoints, num_seeds, num_formulas, num_episodes)
    jax.block_until_ready(returns)
    pbar.close()
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # log to stdout and save to CSV
    save_results(cfg, returns, disc_returns, violations, lengths, checkpoint_steps)


def save_results(
    cfg: DictConfig,
    returns: jax.Array,
    disc_returns: jax.Array,
    violations: jax.Array,
    lengths: jax.Array,
    checkpoint_steps: list[int],
):
    """Save one row of seed-averaged metrics for each checkpoint."""

    csv_path = f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.run}/violation_checkpoints.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "seed",
        "deterministic",
        "timestep",
        "metric",
        "return",
        "vr",
        "length",
    ]

    rows = []
    for i, step in enumerate(checkpoint_steps):
        # Returns, discounted returns, and lengths have shape
        # (num_checkpoints, num_seeds, num_formulas, num_episodes).
        # Violations omit the final axis because they are episode totals.

        mask = returns[i] > 0  # (num_seeds, num_formulas, num_episodes)
        counts = jnp.sum(mask, axis=(1, 2))  # Shape: (num_seeds,)

        def compute_masked_seed_means(data_tensor):
            # Sum values of successful trajectories only
            sum_val = jnp.sum(data_tensor[i] * mask, axis=(1, 2))
            # Avoid division by zero if a seed had 0 successes
            return jnp.where(counts > 0, sum_val / counts, jnp.nan)

        mean_returns, mean_disc_returns = jax.tree.map(
            lambda x, i=i: x[i].mean(axis=(-1, -2)),
            (returns, disc_returns),
        )
        mean_lengths = compute_masked_seed_means(lengths)

        # ``violations`` has no episode axis: Evaluator.eval accumulates the
        # violations over all episodes for each (seed, formula) pair. Convert
        # those counts to per-seed rates before averaging across seeds.
        violations_per_seed = violations[i].mean(axis=-1) / cfg.eval.num_episodes

        # Save one seed-averaged row for each checkpoint.
        num_seeds = int(returns.shape[1])
        for seed in range(num_seeds):
            rows.append(
                {
                    "seed": seed,
                    "deterministic": bool(cfg.eval.deterministic),
                    "timestep": step,
                    "metric": float(mean_returns[seed]),
                    "return": float(mean_disc_returns[seed]),
                    "vr": float(violations_per_seed[seed]),
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
