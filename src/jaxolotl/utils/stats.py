"""Statistical utilities for evaluation across independent training seeds."""

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t


class ConfidenceInterval(NamedTuple):
    """A point estimate and two-sided confidence interval."""

    mean: float
    lower: float
    upper: float


class TaskSetConfidenceIntervals(NamedTuple):
    """Confidence intervals for tasks and their per-policy aggregate."""

    per_task: tuple[ConfidenceInterval, ...]
    task_set: ConfidenceInterval


NanPolicy = Literal["raise", "omit"]


def bootstrap_ci(
    values: ArrayLike,
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int | None = 42,
    nan_policy: NanPolicy = "raise",
    bounds: tuple[float, float] | None = None,
) -> ConfidenceInterval:
    """Return a percentile-bootstrap CI for a mean across independent seeds.

    Args:
        values: One value per independently trained policy.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level strictly between zero and one.
        random_seed: Seed for the bootstrap random-number generator. ``None`` uses
            nondeterministic entropy.
        nan_policy: Reject missing values by default. ``"omit"`` drops policies
            with missing values before resampling and is intended only for auxiliary
            metrics that are undefined for some policies (e.g. number of steps).
        bounds: Optional lower and upper bounds used to clip the interval.
    """
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {scores.shape}")
    _validate_bootstrap_settings(n_resamples, confidence_level, nan_policy)
    _validate_bounds(bounds)
    scores = _prepare_scores(scores, nan_policy)

    rng = np.random.default_rng(random_seed)
    counts = _bootstrap_counts(len(scores), n_resamples, rng)
    replicates = counts @ scores / len(scores)
    return _make_bootstrap_ci(
        float(scores.mean()), replicates, confidence_level, bounds=bounds
    )


def bootstrap_task_set(
    values: ArrayLike,
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int | None = 42,
    nan_policy: NanPolicy = "raise",
    bounds: tuple[float, float] | None = None,
) -> TaskSetConfidenceIntervals:
    """Bootstrap per-task and task-set means by resampling trained policies.

    ``values`` must have shape ``(num_seeds, num_tasks)``. A bootstrap replicate
    samples rows once and uses the resulting matrix for every task. This preserves
    correlations between scores obtained from the same trained policy. The task-set
    statistic gives every task equal weight, as in the evaluation protocol.

    With ``nan_policy="omit"``, missing policy scores are omitted within each task
    after drawing the shared seed indices. This is useful for conditional auxiliary
    metrics, but is not the fixed-seed protocol used for primary evaluation metrics.
    """
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(
            f"values must have shape (num_seeds, num_tasks), got shape {scores.shape}"
        )
    if 0 in scores.shape:
        raise ValueError("values must contain at least one seed and one task")
    _validate_bootstrap_settings(n_resamples, confidence_level, nan_policy)
    _validate_bounds(bounds)
    if np.isinf(scores).any():
        raise ValueError("values must not contain infinite values")
    if nan_policy == "raise" and np.isnan(scores).any():
        raise ValueError("values must not contain NaN values")
    rng = np.random.default_rng(random_seed)
    num_seeds = scores.shape[0]
    counts = _bootstrap_counts(num_seeds, n_resamples, rng)

    if nan_policy == "omit":
        task_means = _nanmean(scores, axis=0)
        valid = ~np.isnan(scores)
        replicate_totals = counts @ np.where(valid, scores, 0.0)
        replicate_counts = counts @ valid.astype(np.int32)
        replicate_task_means = np.divide(
            replicate_totals,
            replicate_counts,
            out=np.full_like(replicate_totals, np.nan),
            where=replicate_counts > 0,
        )
    else:
        task_means = scores.mean(axis=0)
        replicate_task_means = counts @ scores / num_seeds

    per_task = tuple(
        _make_bootstrap_ci(
            float(task_means[task_index]),
            replicate_task_means[:, task_index],
            confidence_level,
            omit_nan=nan_policy == "omit",
            allow_empty=nan_policy == "omit",
            bounds=bounds,
        )
        for task_index in range(scores.shape[1])
    )
    if nan_policy == "omit":
        task_set_mean = float(_nanmean(task_means, axis=0))
        replicate_task_set_means = _nanmean(replicate_task_means, axis=1)
    else:
        task_set_mean = float(task_means.mean())
        replicate_task_set_means = replicate_task_means.mean(axis=1)
    task_set = _make_bootstrap_ci(
        task_set_mean,
        replicate_task_set_means,
        confidence_level,
        omit_nan=nan_policy == "omit",
        allow_empty=nan_policy == "omit",
        bounds=bounds,
    )
    return TaskSetConfidenceIntervals(per_task=per_task, task_set=task_set)


def student_t_ci(
    values: ArrayLike,
    *,
    confidence_level: float = 0.95,
    nan_policy: NanPolicy = "raise",
    bounds: tuple[float, float] | None = None,
) -> ConfidenceInterval:
    """Return a Student-t confidence interval for a mean across policies.

    A confidence interval is undefined for a single policy, so in that case the
    point estimate is returned with NaN interval endpoints.
    """
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {scores.shape}")
    _validate_common_settings(confidence_level, nan_policy)
    _validate_bounds(bounds)
    scores = _prepare_scores(scores, nan_policy)
    return _make_student_t_ci(scores, confidence_level, bounds)


def student_t_task_set(
    values: ArrayLike,
    *,
    confidence_level: float = 0.95,
    nan_policy: NanPolicy = "raise",
    bounds: tuple[float, float] | None = None,
) -> TaskSetConfidenceIntervals:
    """Return Student-t intervals for tasks and their per-policy aggregate.

    ``values`` must have shape ``(num_seeds, num_tasks)``. Task scores are
    averaged within each policy before computing the task-set interval, preserving
    correlations between scores from the same training run.
    """
    scores = _prepare_task_scores(values, confidence_level, nan_policy, bounds)
    per_task = tuple(
        _make_student_t_ci(
            _prepare_scores(
                scores[:, task_index], nan_policy, allow_empty=nan_policy == "omit"
            ),
            confidence_level,
            bounds,
        )
        for task_index in range(scores.shape[1])
    )
    if nan_policy == "omit":
        policy_scores = _nanmean(scores, axis=1)
    else:
        policy_scores = scores.mean(axis=1)
    task_set = _make_student_t_ci(
        _prepare_scores(policy_scores, nan_policy, allow_empty=nan_policy == "omit"),
        confidence_level,
        bounds,
    )
    return TaskSetConfidenceIntervals(per_task=per_task, task_set=task_set)


def _validate_bootstrap_settings(
    n_resamples: int, confidence_level: float, nan_policy: NanPolicy
) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    _validate_common_settings(confidence_level, nan_policy)


def _validate_common_settings(confidence_level: float, nan_policy: NanPolicy) -> None:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    if nan_policy not in ("raise", "omit"):
        raise ValueError("nan_policy must be 'raise' or 'omit'")


def _validate_bounds(bounds: tuple[float, float] | None) -> None:
    if bounds is None:
        return
    lower, upper = bounds
    if np.isnan(lower) or np.isnan(upper) or lower > upper:
        raise ValueError("bounds must be an ordered pair without NaN values")


def _prepare_task_scores(
    values: ArrayLike,
    confidence_level: float,
    nan_policy: NanPolicy,
    bounds: tuple[float, float] | None,
) -> NDArray[np.float64]:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(
            f"values must have shape (num_seeds, num_tasks), got shape {scores.shape}"
        )
    if 0 in scores.shape:
        raise ValueError("values must contain at least one seed and one task")
    _validate_common_settings(confidence_level, nan_policy)
    _validate_bounds(bounds)
    if np.isinf(scores).any():
        raise ValueError("values must not contain infinite values")
    if nan_policy == "raise" and np.isnan(scores).any():
        raise ValueError("values must not contain NaN values")
    return scores


def _prepare_scores(
    scores: NDArray[np.float64],
    nan_policy: NanPolicy,
    *,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    if np.isinf(scores).any():
        raise ValueError("values must not contain infinite values")
    if nan_policy == "omit":
        scores = scores[~np.isnan(scores)]
    elif np.isnan(scores).any():
        raise ValueError("values must not contain NaN values")
    if len(scores) == 0 and not allow_empty:
        raise ValueError("values must contain at least one non-NaN value")
    return scores


def _bootstrap_counts(
    num_values: int, n_resamples: int, rng: np.random.Generator
) -> NDArray[np.int32]:
    """Draw bootstrap samples as count vectors to avoid materializing task cubes."""
    indices = rng.integers(0, num_values, size=(n_resamples, num_values))
    counts = np.zeros((n_resamples, num_values), dtype=np.int32)
    rows = np.broadcast_to(np.arange(n_resamples)[:, None], indices.shape)
    np.add.at(counts, (rows, indices), 1)
    return counts


def _nanmean(values: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Compute a NaN-omitting mean without emitting all-NaN slice warnings."""
    valid = ~np.isnan(values)
    counts = valid.sum(axis=axis)
    totals = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(
        totals,
        counts,
        out=np.full_like(totals, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _make_bootstrap_ci(
    mean: float,
    replicates: NDArray[np.float64],
    confidence_level: float,
    *,
    omit_nan: bool = False,
    allow_empty: bool = False,
    bounds: tuple[float, float] | None = None,
) -> ConfidenceInterval:
    if omit_nan:
        replicates = replicates[~np.isnan(replicates)]
    if len(replicates) == 0:
        if allow_empty:
            return ConfidenceInterval(mean=mean, lower=np.nan, upper=np.nan)
        raise ValueError("no valid bootstrap replicates were produced")
    tail = (1 - confidence_level) / 2
    lower, upper = np.percentile(replicates, [tail * 100, (1 - tail) * 100])
    lower, upper = _clip_interval(float(lower), float(upper), bounds)
    return ConfidenceInterval(mean=mean, lower=lower, upper=upper)


def _make_student_t_ci(
    scores: NDArray[np.float64],
    confidence_level: float,
    bounds: tuple[float, float] | None,
) -> ConfidenceInterval:
    if len(scores) == 0:
        return ConfidenceInterval(mean=np.nan, lower=np.nan, upper=np.nan)
    mean = float(scores.mean())
    if len(scores) < 2:
        return ConfidenceInterval(mean=mean, lower=np.nan, upper=np.nan)
    standard_error = float(scores.std(ddof=1) / np.sqrt(len(scores)))
    quantile = float(t.ppf((1 + confidence_level) / 2, df=len(scores) - 1))
    lower, upper = _clip_interval(
        mean - quantile * standard_error,
        mean + quantile * standard_error,
        bounds,
    )
    return ConfidenceInterval(mean=mean, lower=lower, upper=upper)


def _clip_interval(
    lower: float,
    upper: float,
    bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    if bounds is None:
        return lower, upper
    bound_lower, bound_upper = bounds
    return max(lower, bound_lower), min(upper, bound_upper)
