from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum.curriculum_manager import (
    CurriculumManager,
    GlobalCurriculumState,
)
from jaxolotl.alg.curriculum.wrapper import CurriculumState
from jaxolotl.environments.wrappers.wrapper import WrapperState

WINDOW = 8


class _Base(eqx.Module):
    value: jax.Array


class _OuterState(WrapperState):
    pass


def _env_state(*, current: list[int], adopted: list[int] | None = None) -> _OuterState:
    if adopted is None:
        adopted = current
    return _OuterState(
        state=CurriculumState(
            state=_Base(value=jnp.arange(len(current))),
            curriculum_stage=jnp.array(current, dtype=jnp.int32),
            adopted_stage=jnp.array(adopted, dtype=jnp.int32),
        )
    )


def _manager(
    num_envs: int,
    *,
    threshold: float = 0.95,
    adopt_prob: float = 0.0,
    min_coverage: float = 0.0,
) -> CurriculumManager:
    return CurriculumManager(
        thresholds=jnp.full((4,), threshold, dtype=jnp.float32),
        num_envs=num_envs,
        window=WINDOW,
        adopt_prob=adopt_prob,
        min_coverage=min_coverage,
    )


def _global_state(
    manager: CurriculumManager, *, frontier: int, seen: int = 0
) -> GlobalCurriculumState:
    return manager.init_state()._replace(
        curriculum_frontier=jnp.array(frontier, dtype=jnp.int32),
        current_stage_episodes=jnp.array(seen, dtype=jnp.int32),
    )


def _rollout(
    episodes: Sequence[Sequence[tuple[int, float] | None]],
) -> tuple[jax.Array, dict]:
    """Build rollout data where each entry is ``(stage, success)`` or ``None``."""
    done = jnp.array([[episode is not None for episode in row] for row in episodes])
    stage = jnp.array(
        [[0 if episode is None else episode[0] for episode in row] for row in episodes],
        dtype=jnp.int32,
    )
    success = jnp.array(
        [
            [0.0 if episode is None else episode[1] for episode in row]
            for row in episodes
        ],
        dtype=jnp.float32,
    )
    return done, {
        "curriculum_episode_stage": stage,
        "curriculum_episode_success": success,
    }


def _run(
    manager: CurriculumManager,
    state: GlobalCurriculumState,
    env_state: _OuterState,
    episodes: Sequence[Sequence[tuple[int, float] | None]],
) -> tuple[GlobalCurriculumState, _OuterState]:
    done, info = _rollout(episodes)
    return manager.update_progress(
        state,
        env_state,
        done,
        info,
        jax.random.key(0),
    )


def _single_env_episodes(outcomes: Sequence[float], stage: int = 0):
    return [[(stage, outcome)] for outcome in outcomes]


def test_cleared_full_window_advances_and_clears_evidence() -> None:
    manager = _manager(1)
    out, _ = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[0]),
        _single_env_episodes([1.0] * WINDOW),
    )

    assert int(out.curriculum_frontier) == 1
    assert float(jnp.sum(out.last_successes)) == 0.0
    assert int(out.index) == 0
    assert int(out.current_stage_episodes) == 0


def test_partly_filled_window_holds_frontier() -> None:
    manager = _manager(1)
    out, _ = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[0]),
        _single_env_episodes([1.0] * 4),
    )

    assert int(out.curriculum_frontier) == 0
    assert int(out.current_stage_episodes) == 4


def test_episodes_below_frontier_are_not_counted() -> None:
    manager = _manager(1)
    out, _ = _run(
        manager,
        _global_state(manager, frontier=1),
        _env_state(current=[0]),
        _single_env_episodes([1.0] * WINDOW, stage=0),
    )

    assert int(out.curriculum_frontier) == 1
    assert int(out.current_stage_episodes) == 0


def test_recorded_episode_stage_is_used_instead_of_final_environment_stage() -> None:
    manager = _manager(1)
    out, _ = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[1]),
        _single_env_episodes([1.0] * WINDOW, stage=0),
    )

    assert int(out.curriculum_frontier) == 1


def test_infinite_threshold_marks_final_stage() -> None:
    manager = _manager(1, threshold=float("inf"))
    out, _ = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[0]),
        _single_env_episodes([1.0] * WINDOW),
    )

    assert int(out.curriculum_frontier) == 0


def test_long_rollout_keeps_most_recent_window() -> None:
    manager = _manager(1)
    recovered, _ = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[0]),
        _single_env_episodes([0.0] * WINDOW + [1.0] * WINDOW),
    )
    lapsed, _ = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[0]),
        _single_env_episodes([1.0] * WINDOW + [0.0] * WINDOW),
    )

    assert int(recovered.curriculum_frontier) == 1
    assert int(lapsed.curriculum_frontier) == 0


def test_decision_is_independent_of_environment_batching() -> None:
    wide_manager = _manager(WINDOW)
    wide, _ = _run(
        wide_manager,
        _global_state(wide_manager, frontier=0),
        _env_state(current=[0] * WINDOW),
        [[(0, 1.0)] * WINDOW],
    )
    narrow_manager = _manager(1)
    narrow, _ = _run(
        narrow_manager,
        _global_state(narrow_manager, frontier=0),
        _env_state(current=[0]),
        _single_env_episodes([1.0] * WINDOW),
    )

    assert int(wide.curriculum_frontier) == int(narrow.curriculum_frontier) == 1


def test_adoption_moves_only_lagging_environments_by_one() -> None:
    manager = _manager(3, adopt_prob=1.0)
    out, env_state = _run(
        manager,
        _global_state(manager, frontier=2),
        _env_state(current=[0, 1, 2], adopted=[0, 1, 2]),
        [[None, None, None]],
    )

    assert int(out.curriculum_frontier) == 2
    assert env_state.adopted_stage.tolist() == [1, 2, 2]
    assert env_state.curriculum_stage.tolist() == [0, 1, 2]


def test_new_frontier_is_not_adopted_until_next_update() -> None:
    manager = _manager(1, adopt_prob=1.0)
    out, env_state = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[0]),
        _single_env_episodes([1.0] * WINDOW),
    )

    assert int(out.curriculum_frontier) == 1
    assert int(env_state.adopted_stage[0]) == 0


def test_frontier_waits_for_environment_coverage() -> None:
    manager = _manager(4, min_coverage=0.9)
    busy = [[(0, 1.0), None, None, None] for _ in range(WINDOW)]
    out, _ = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[0] * 4),
        busy,
    )

    assert int(out.curriculum_frontier) == 0
    assert out.contributed_at_frontier.tolist() == [True, False, False, False]


def test_coverage_carries_across_rollouts_and_clears_on_advance() -> None:
    manager = _manager(4, min_coverage=0.9)
    state, env_state = _run(
        manager,
        _global_state(manager, frontier=0),
        _env_state(current=[0] * 4),
        [[(0, 1.0), None, None, None] for _ in range(WINDOW)],
    )
    state, _ = _run(
        manager,
        state,
        env_state,
        [[None, (0, 1.0), (0, 1.0), (0, 1.0)]],
    )

    assert int(state.curriculum_frontier) == 1
    assert not bool(jnp.any(state.contributed_at_frontier))
