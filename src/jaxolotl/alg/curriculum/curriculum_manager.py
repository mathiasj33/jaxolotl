"""Population-wide curriculum management between training rollouts."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from jaxolotl.alg.curriculum.wrapper import CurriculumState


class GlobalCurriculumState(NamedTuple):
    """Global curriculum state."""

    curriculum_frontier: jax.Array  # int, stage currently gathering evidence
    last_successes: jax.Array  # shape (N,), successes from last `window` episodes
    index: jax.Array  # int, index to write next success into last_successes
    current_stage_episodes: (
        jax.Array
    )  # int, number of completed episodes in the current frontier stage
    contributed_at_frontier: jax.Array
    # shape (n_envs,) which environments have written a return at the current frontier stage


class CurriculumManager(eqx.Module):
    """Advance a shared frontier and migrate environments towards it."""

    thresholds: jax.Array
    num_envs: int
    window: int
    adopt_prob: float
    min_coverage: float

    def __init__(
        self,
        thresholds: jax.Array,
        num_envs: int,
        window: int,
        adopt_prob: float = 0.1,
        min_coverage: float = 0.9,
    ):
        self.thresholds = thresholds
        self.num_envs = num_envs
        self.window = window
        self.adopt_prob = adopt_prob
        self.min_coverage = min_coverage

    def init_state(self) -> GlobalCurriculumState:
        return GlobalCurriculumState(
            curriculum_frontier=jnp.zeros((), dtype=jnp.int32),
            last_successes=jnp.zeros((self.window,), dtype=jnp.float32),
            index=jnp.zeros((), dtype=jnp.int32),
            current_stage_episodes=jnp.zeros((), dtype=jnp.int32),
            contributed_at_frontier=jnp.zeros((self.num_envs,), dtype=jnp.bool),
        )

    def update_progress(
        self,
        state: GlobalCurriculumState,
        env_state: PyTree,  # batched current environment states
        done: jax.Array,  # shape (n_steps, n_envs), whether each environment is done
        info: dict,  # shape (n_steps, n_envs, ...), info dicts from each environment
        key: jax.Array,
    ) -> tuple[GlobalCurriculumState, PyTree]:
        """Update progress tracking and set adopted curriculum stages for each environment.

        Returns:
            new_state: GlobalCurriculumState, updated global curriculum state
            new_env_state: PyTree, updated environment states with adopted curriculum stages
        """

        num_envs = done.shape[1]
        if num_envs != self.num_envs:
            raise ValueError(
                f"Manager configured for {self.num_envs} environments, got {num_envs}."
            )
        done = done.reshape(-1)  # (batch,)
        stage = info["curriculum_episode_stage"].reshape(-1)  # (batch,)
        counted = jnp.logical_and(done, stage == state.curriculum_frontier)
        total = jnp.sum(counted)

        # Only keep the last `window` successes for the current frontier stage
        offset = jnp.cumsum(counted) - 1
        keep = jnp.logical_and(counted, offset >= total - self.window)
        slot = jnp.where(keep, (state.index + offset) % self.window, self.window)

        # Overwrite last_successes with the given rollouts
        successes = info["curriculum_episode_success"].reshape(-1)  # (batch,)
        last_successes = state.last_successes.at[slot].set(successes, mode="drop")
        seen = state.current_stage_episodes + total

        # Check if we should advance the global curriculum frontier
        contributed = jnp.logical_or(
            state.contributed_at_frontier,
            jnp.any(counted.reshape(-1, num_envs), axis=0),
        )
        threshold = self.thresholds[state.curriculum_frontier]
        advance = (
            (seen >= self.window)
            & (jnp.mean(contributed) >= self.min_coverage)
            & (jnp.mean(last_successes) >= threshold)
        )

        # Check which environments should step up to the next curriculum stage
        behind = env_state.adopted_stage < state.curriculum_frontier
        step_up = behind & (
            jax.random.uniform(key, env_state.adopted_stage.shape) < self.adopt_prob
        )

        # Update individual environment curriculum stages
        new_adopted = env_state.adopted_stage + step_up.astype(
            env_state.adopted_stage.dtype
        )
        new_env_state = env_state.replace_in_chain(
            CurriculumState,
            adopted_stage=new_adopted,
        )

        # Update global curriculum state
        new_state = GlobalCurriculumState(
            curriculum_frontier=state.curriculum_frontier + advance.astype(jnp.int32),
            last_successes=jnp.where(
                advance, jnp.zeros_like(last_successes), last_successes
            ),
            index=jnp.where(advance, 0, (state.index + total) % self.window),
            current_stage_episodes=jnp.where(advance, 0, seen),
            contributed_at_frontier=jnp.where(
                advance, jnp.zeros_like(contributed), contributed
            ),
        )
        return new_state, new_env_state
