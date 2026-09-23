"""Evaluate trained models on a specified set of LTL formulas.

Evaluates all models on all formulas. Saves results to a CSV
file and prints to stdout. Use the batch_size config options
to trade off speed and memory usage during evaluation.
"""

import csv
import logging
import time
from pathlib import Path

import hydra
import jax
import numpy as np
from hydra.core.hydra_config import HydraConfig
from jaxtyping import PyTree
from omegaconf import DictConfig

import jaxolotl
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.environments.wrappers.vectorize_wrapper import VectorizeWrapper
from jaxolotl.eqx_utils.utils import compute_num_params
from jaxolotl.eval.utils import (
    load_batched_actor_critic,
    load_batched_gcvf,
    load_latest_checkpoint_models,
    make_eval_fn,
)
from jaxolotl.utils.stats import (
    ConfidenceInterval,
    NanPolicy,
    TaskSetConfidenceIntervals,
    bootstrap_task_set,
    student_t_task_set,
)

logger = logging.getLogger(__name__)

METRIC_LABELS = {
    "return": "SR/AV",
    "violations": "Violations",
    "length": "Successful-episode length",
}


@hydra.main(version_base="1.3", config_path="../../conf", config_name="eval")
def main(cfg: DictConfig):
    # build environment
    discretize = cfg.alg.name == "gcrl_ltl"
    env_params = cfg.get("env_params", {})
    env, env_params = jaxolotl.make(
        cfg.env.name,
        reset_source=cfg.env.get("reset_source", "test"),
        discretize=discretize,
        **env_params,
    )
    env = TimeLimitWrapper(env)
    env = hydra.utils.call(cfg.alg.wrap_env, env, cfg, training=False)
    env = VectorizeWrapper(env)

    # preprocess formulas
    logger.info("Preprocessing formulas...")
    formulas: PyTree = hydra.utils.call(cfg.alg.preprocess_formulas, cfg.formulas, env)

    # load models
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    checkpoint = cfg.eval.get("checkpoint", None)
    if checkpoint is None:
        models, seeds = load_batched_actor_critic(cfg, env, env_params, key=model_key)
    else:
        models, seeds, checkpoint_step = load_latest_checkpoint_models(
            cfg, env, env_params, key=model_key, max_steps=checkpoint
        )
        logger.info("Loaded checkpoint at step %s.", checkpoint_step)
    num_models = len(seeds)

    if cfg.alg.name == "gcrl_ltl":
        gcvf, gcvf_seeds = load_batched_gcvf(cfg, env, env_params, key=model_key)
        if gcvf_seeds != seeds:
            raise ValueError(
                "Actor-critic and GCVF seed sets do not match: "
                f"{seeds} != {gcvf_seeds}."
            )
        agents_factory = hydra.utils.instantiate(cfg.alg.agent, models, _partial_=True)
        agents = agents_factory(gcvf=gcvf)
    else:
        agents = hydra.utils.instantiate(cfg.alg.agent, models)

    logger.info(
        f"Loaded seeds {seeds}. Each model has {compute_num_params(models) / num_models / 1e3}k parameters."
    )

    # set up evaluator
    eval_fn = make_eval_fn(
        cfg, num_models, num_formulas=len(cfg.formulas), return_trajs=False
    )

    # evaluate
    key, eval_key = jax.random.split(key)
    logger.info("Starting evaluation...")
    start = time.time()
    returns, _, lengths, violations, _ = eval_fn(
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
    task_set = formula_group.split("/")[-1]
    log_and_save_results(cfg, returns, lengths, violations, task_set, seeds)


def log_and_save_results(
    cfg: DictConfig,
    returns: jax.Array,
    lengths: jax.Array,
    violations: jax.Array,
    task_set: str,
    seeds: list[int],
):
    """Log confidence intervals and save raw and aggregated evaluation results."""

    aggregates, seed_metrics = compute_metric_aggregates(
        cfg, returns, lengths, violations, seeds
    )
    metadata = {
        "deterministic": bool(cfg.eval.deterministic),
        "ci_method": str(cfg.eval.confidence_interval.method),
        "confidence_level": float(cfg.eval.confidence_interval.confidence_level),
    }
    fieldnames = [
        "seed",
        "deterministic",
        "formula",
        "return",
        "violations",
        "length",
    ]
    rows = []
    agg_rows = []
    formulas = [str(formula) for formula in cfg.formulas]
    for formula_index, formula in enumerate(formulas):
        logger.info("========================================")
        logger.info("Formula: %s", formula)
        for metric, agg in aggregates.items():
            ci = agg.per_task[formula_index]
            logger.info("%s: %s", METRIC_LABELS[metric], _format_ci(ci))
            agg_rows.append(
                _summary_row(
                    task_set=task_set,
                    scope="formula",
                    formula=formula,
                    metric=metric,
                    ci=ci,
                    num_seeds=int(
                        np.isfinite(seed_metrics[metric][:, formula_index]).sum()
                    ),
                    num_formulas=1,
                    metadata=metadata,
                )
            )

        for seed_index, seed in enumerate(seeds):
            rows.append(
                {
                    "seed": seed,
                    "deterministic": bool(cfg.eval.deterministic),
                    "formula": formula,
                    "return": float(seed_metrics["return"][seed_index, formula_index]),
                    "violations": float(
                        seed_metrics["violations"][seed_index, formula_index]
                    ),
                    "length": float(seed_metrics["length"][seed_index, formula_index]),
                }
            )

    logger.info("========================================")
    logger.info("Task set: %s", task_set)
    for metric, agg in aggregates.items():
        logger.info("Overall %s: %s", METRIC_LABELS[metric], _format_ci(agg.task_set))
        agg_rows.append(
            _summary_row(
                task_set=task_set,
                scope="task_set",
                formula="",
                metric=metric,
                ci=agg.task_set,
                num_seeds=int(np.isfinite(seed_metrics[metric]).any(axis=1).sum()),
                num_formulas=int(np.isfinite(seed_metrics[metric]).any(axis=0).sum()),
                metadata=metadata,
            )
        )

    if cfg.save:
        output_dir = Path("runs", cfg.env.name, cfg.alg.name, cfg.run, "eval")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"{task_set}.csv"
        _write_csv(results_path, fieldnames, rows)

        logger.info("Wrote per-seed results to %s", results_path)

        summary_path = output_dir / f"{task_set}_agg.csv"
        _write_csv(summary_path, list(agg_rows[0]), agg_rows)
        logger.info("Wrote confidence-interval summary to %s", summary_path)


def compute_metric_aggregates(
    cfg: DictConfig,
    returns: jax.Array,
    lengths: jax.Array,
    violations: jax.Array,
    seeds: list[int],
) -> tuple[dict[str, TaskSetConfidenceIntervals], dict[str, np.ndarray]]:
    """Compute per-seed, per-formula means and confidence intervals for each metric."""

    formulas = [str(formula) for formula in cfg.formulas]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Evaluation seed labels must be unique, got {seeds}")

    metrics: dict[str, np.ndarray] = {
        "return": np.asarray(returns, dtype=np.float64),
        "violations": np.asarray(violations, dtype=np.float64),
        "length": np.asarray(lengths, dtype=np.float64),
    }
    expected_prefix = (len(seeds), len(formulas))
    for metric, values in metrics.items():
        if values.ndim != 3 or values.shape[:2] != expected_prefix:
            raise ValueError(
                f"{metric} must have shape (num_seeds, num_formulas, num_episodes) "
                f"with prefix {expected_prefix}, got {values.shape}"
            )

    return_means = metrics["return"].mean(axis=2)
    violation_means = metrics["violations"].mean(axis=2)
    success_mask = metrics["return"] > 0
    success_counts = success_mask.sum(axis=2)
    successful_length_sums = np.where(success_mask, metrics["length"], 0.0).sum(axis=2)
    length_means = np.divide(
        successful_length_sums,
        success_counts,
        out=np.full(expected_prefix, np.nan, dtype=np.float64),
        where=success_counts > 0,
    )
    # these are per-seed, per-formula means over episodes
    seed_metrics: dict[str, np.ndarray] = {
        "return": return_means,
        "violations": violation_means,
        "length": length_means,
    }

    aggregates: dict[str, TaskSetConfidenceIntervals] = {}
    for metric, values in seed_metrics.items():
        nan_policy = "omit" if metric == "length" else "raise"
        bounds = _metric_bounds(metric, finite=bool(cfg.eval.finite))
        if metric == "length" and np.isnan(values).any():
            logger.warning(
                "Successful-episode length is undefined for %d seed/task pairs (no successful episodes).",
                int(np.isnan(values).sum()),
            )
        aggregates[metric] = _confidence_intervals(
            values,
            ci_cfg=cfg.eval.confidence_interval,
            nan_policy=nan_policy,
            bounds=bounds,
        )
    return aggregates, seed_metrics


def _confidence_intervals(
    values: np.ndarray,
    *,
    ci_cfg: DictConfig,
    nan_policy: NanPolicy,
    bounds: tuple[float, float],
) -> TaskSetConfidenceIntervals:
    method = str(ci_cfg.method)
    common_options = {
        "confidence_level": float(ci_cfg.confidence_level),
        "nan_policy": nan_policy,
        "bounds": bounds,
    }
    if method == "student_t":
        return student_t_task_set(values, **common_options)
    if method == "bootstrap":
        return bootstrap_task_set(
            values,
            n_resamples=int(ci_cfg.bootstrap.n_resamples),
            random_seed=int(ci_cfg.bootstrap.seed),
            **common_options,
        )
    raise ValueError(
        "eval.confidence_interval.method must be 'student_t' or "
        f"'bootstrap', got {method!r}"
    )


def _metric_bounds(metric: str, *, finite: bool) -> tuple[float, float]:
    if metric == "return" and finite:
        return 0.0, 1.0
    return 0.0, np.inf


def _format_ci(ci: ConfidenceInterval) -> str:
    return f"{ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]"


def _summary_row(
    *,
    task_set: str,
    scope: str,
    formula: str,
    metric: str,
    ci: ConfidenceInterval,
    num_seeds: int,
    num_formulas: int,
    metadata: dict[str, str | int | float | bool],
) -> dict[str, str | int | float | bool]:
    return {
        "task_set": task_set,
        "scope": scope,
        "formula": formula,
        "metric": metric,
        "mean": ci.mean,
        "ci_lower": ci.lower,
        "ci_upper": ci.upper,
        "num_seeds": num_seeds,
        "num_formulas": num_formulas,
        **metadata,
    }


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open(mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
