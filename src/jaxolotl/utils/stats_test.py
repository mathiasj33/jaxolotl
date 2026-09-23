import numpy as np
import pytest

from jaxolotl.utils.stats import (
    bootstrap_ci,
    bootstrap_task_set,
    student_t_ci,
    student_t_task_set,
)


def test_student_t_ci_uses_sample_variance_and_bounds() -> None:
    result = student_t_ci([0.2, 0.4, 0.6], bounds=(0.0, 1.0))

    assert result.mean == pytest.approx(0.4)
    assert result.lower == 0.0
    assert result.upper == pytest.approx(0.896827542)


def test_student_t_task_set_aggregates_within_policy() -> None:
    scores = np.array([[0.0, 1.0], [1.0, 0.0], [0.25, 0.75]])

    result = student_t_task_set(scores)

    assert result.task_set.mean == 0.5
    assert result.task_set.lower == pytest.approx(0.5)
    assert result.task_set.upper == pytest.approx(0.5)


def test_bootstrap_ci_is_degenerate_for_one_seed() -> None:
    result = bootstrap_ci([2.5], n_resamples=50)

    assert result.mean == 2.5
    assert result.lower == 2.5
    assert result.upper == 2.5


def test_task_set_bootstrap_preserves_within_seed_correlation() -> None:
    # Every policy has task-set performance 0.5. Resampling task scores
    # independently would incorrectly give the aggregate a nonzero-width interval.
    scores = np.array([[0.0, 1.0], [1.0, 0.0]])

    result = bootstrap_task_set(scores, n_resamples=1_000, random_seed=0)

    assert [ci.mean for ci in result.per_task] == [0.5, 0.5]
    assert result.task_set.mean == 0.5
    assert result.task_set.lower == pytest.approx(0.5)
    assert result.task_set.upper == pytest.approx(0.5)


def test_task_set_bootstrap_rejects_missing_primary_scores() -> None:
    with pytest.raises(ValueError, match="NaN"):
        bootstrap_task_set([[1.0, np.nan], [0.0, 1.0]])


def test_task_set_bootstrap_can_omit_missing_auxiliary_scores() -> None:
    result = bootstrap_task_set(
        [[1.0, np.nan], [3.0, 4.0]],
        n_resamples=1_000,
        random_seed=0,
        nan_policy="omit",
    )

    assert [ci.mean for ci in result.per_task] == [2.0, 4.0]
    assert result.task_set.mean == 3.0
    assert np.isfinite(result.task_set.lower)
    assert np.isfinite(result.task_set.upper)


def test_missing_auxiliary_task_is_reported_as_undefined() -> None:
    result = bootstrap_task_set(
        [[1.0, np.nan], [3.0, np.nan]],
        n_resamples=100,
        nan_policy="omit",
    )

    assert np.isnan(result.per_task[1].mean)
    assert np.isnan(result.per_task[1].lower)
    assert np.isnan(result.per_task[1].upper)
    assert result.task_set.mean == 2.0

    result = student_t_task_set([[1.0, np.nan], [3.0, np.nan]], nan_policy="omit")

    assert np.isnan(result.per_task[1].mean)
    assert np.isnan(result.per_task[1].lower)
    assert np.isnan(result.per_task[1].upper)
    assert result.task_set.mean == 2.0
