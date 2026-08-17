"""Evaluate trained models on a specified set of LTL formulas.

Evaluates all models on all formulas. Saves results to a CSV
file and prints to stdout. Use the batch_size config options
to trade off speed and memory usage during evaluation.
"""

import csv
import logging
import os
import time

import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from jaxtyping import PyTree
from omegaconf import DictConfig

import jaxltl
from jaxltl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxltl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxltl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxltl.eqx_utils.utils import compute_num_params
from jaxltl.eval.utils import (
    load_batched_models,
    make_eval_fn,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../../conf", config_name="eval")
def main(cfg: DictConfig):
    # build environment
    env_params = cfg.get("env_params", {})
    env, env_params = jaxltl.make(cfg.env.name, **env_params)
    if cfg.env.use_precomputed_resets:
        resets_path = (
            f"{jaxltl.DATA_DIR}/{cfg.env.name}/{cfg.env.precomputed_resets_path}"
        )
        env = PrecomputedResetWrapper(env, env_params, resets_path)
    env = TimeLimitWrapper(env)
    env = hydra.utils.call(cfg.alg.wrap_env, env, cfg, training=False)
    env = VectorizeWrapper(env)

    # preprocess formulas
    logger.info("Preprocessing formulas...")
    formulas: PyTree = hydra.utils.call(cfg.alg.preprocess_formulas, cfg.formulas, env)

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    models, num_models = load_batched_models(cfg, env, env_params, key=model_key)
    agents = hydra.utils.instantiate(cfg.alg.agent, models)

    logger.info(
        f"Loaded {num_models} seeds. Each model has {compute_num_params(models) / num_models / 1e3}k parameters."
    )

    # set up evaluator
    eval_fn = make_eval_fn(
        cfg, num_models, num_formulas=len(cfg.formulas), return_trajs=False
    )

    # evaluate
    key, eval_key = jax.random.split(key)
    logger.info("Starting evaluation...")
    start = time.time()
    returns, disc_returns, lengths, violations, _ = eval_fn(
        agents,
        env,
        env_params,
        formulas,
        eval_key,
    )  # shape: (num_seeds, num_formulas, num_episodes)
    jax.block_until_ready(returns)
    logger.info(f"Evaluation completed in {time.time() - start:.2f} seconds.")

    # log to stdout and save to CSV
    formula_group = HydraConfig.get().runtime.choices.get("formulas", "default")
    formula_suffix = formula_group.split("/")[-1]
    log_and_save_results(cfg, returns, lengths, violations, formula_suffix)


def log_and_save_results(
    cfg: DictConfig,
    returns: jax.Array,
    lengths: jax.Array,
    violations: jax.Array,
    formula_suffix: str,
):
    """Logs aggregated results per formula and saves per-seed results to a CSV file."""
    csv_path = (
        f"runs/{cfg.env.name}/{cfg.alg.name}/{cfg.run}/results_{formula_suffix}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    num_seeds = int(returns.shape[0])
    seeds = list(range(num_seeds))

    fieldnames = [
        "seed",
        "deterministic",
        "formula",
        "return",
        "violations",
        "length",
    ]

    violations /= cfg.eval.num_episodes
    rows = []
    for i, formula in enumerate(cfg.formulas):
        # Compute per-seed stats
        returns_i = returns[:, i]  # (num_seeds, num_episodes)
        lengths_i = lengths[:, i]  # (num_seeds, num_episodes)
        violations_i = violations[:, i]  # (num_seeds,)

        means = jnp.mean(returns_i, axis=1)  # (num_seeds,)

        success_mask = returns_i > 0  # (num_seeds, num_episodes)
        success_counts = jnp.sum(success_mask, axis=1)  # (num_seeds,)
        sum_lengths = jnp.sum(lengths_i * success_mask, axis=1)
        avg_lengths = jnp.where(
            success_counts > 0, sum_lengths / success_counts, jnp.nan
        )

        # Stdout logging (aggregate across seeds)
        logger.info("========================================")
        logger.info(f"Formula: {formula}")
        logger.info(f"SR/AV: {float(jnp.mean(means)):.3f}+-{float(jnp.std(means)):.3f}")
        logger.info(
            f"Violations: {float(jnp.mean(violations_i)):.3f}+-{float(jnp.std(violations_i)):.3f}"
        )
        logger.info(
            f"Length: {float(jnp.mean(avg_lengths)):.3f}+-{float(jnp.std(avg_lengths)):.3f}"
        )

        # CSV rows (per-seed)
        for seed in seeds:
            rows.append(
                {
                    "seed": seed,
                    "deterministic": bool(cfg.eval.deterministic),
                    "formula": formula,
                    "return": float(means[seed]),
                    "violations": float(violations_i[seed]),
                    "length": float(avg_lengths[seed]),
                }
            )

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote results to {csv_path}")

    per_seed_means = jnp.mean(returns, axis=(1, 2))  # (num_seeds,)
    logger.info("========================================")
    logger.info(
        f"Overall SR/AV: {float(jnp.mean(per_seed_means)):.3f}+-{float(jnp.std(per_seed_means)):.3f}"
    )
    per_seed_violations = jnp.mean(violations, axis=1)  # (num_seeds,)
    logger.info(
        f"Overall Violations: {float(jnp.mean(per_seed_violations)):.3f}+-{float(jnp.std(per_seed_violations)):.3f}"
    )
    per_seed_lengths = jnp.mean(lengths, axis=(1, 2))  # (num_seeds,)
    logger.info(
        f"Overall Length: {float(jnp.mean(per_seed_lengths)):.3f}+-{float(jnp.std(per_seed_lengths)):.3f}"
    )


if __name__ == "__main__":
    main()
