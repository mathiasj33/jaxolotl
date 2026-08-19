from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.ltl2action.curriculum.curriculum import Curriculum


class CurriculumState(WrapperState):
    """State for CurriculumWrapper."""

    curriculum_stage: jax.Array  # current stage in the curriculum
    last_returns: jax.Array  # shape (N,), returns from last N episodes
    returns_index: jax.Array  # int, index to write next return into last_returns
    # int, number of completed episodes in the current stage
    current_stage_episodes: jax.Array


class CurriculumResetOptions(NamedTuple):
    """Reset options for environments that can be used with a curriculum."""

    task: PyTree  # task selected by the curriculum wrapper


class CurriculumWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TSample,
    TJaxSample: eqx.Module,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """A wrapper that samples tasks from a curriculum and advances through the curriculum
    stages based on performance."""

    curriculum: Curriculum[TSample, TJaxSample]
    episode_window: int  # number of episodes to consider for average return

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
        curriculum: Curriculum[TSample, TJaxSample],
        episode_window: int,
    ):
        super().__init__(env)
        self.curriculum = curriculum
        self.episode_window = episode_window

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: CurriculumState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[CurriculumState, EnvObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)

        if state is None:
            stage = jnp.zeros((), dtype=jnp.int32)
            task = self.curriculum.sample(stage, sample_key)
            last_returns = jnp.zeros((self.episode_window,), dtype=jnp.float32)
            returns_index = jnp.zeros((), dtype=jnp.int32)
            current_stage_episodes = jnp.zeros((), dtype=jnp.int32)
        else:
            threshold = self.curriculum.threshold(state.curriculum_stage)
            avg_return = jax.lax.cond(
                state.current_stage_episodes < self.episode_window,
                lambda: -jnp.inf,
                lambda: jnp.mean(state.last_returns),  # type: ignore
            )
            change_stage = avg_return >= threshold
            stage = jax.lax.cond(
                change_stage,
                lambda: state.curriculum_stage + 1,  # type: ignore
                lambda: state.curriculum_stage,  # type: ignore
            )
            last_returns = jax.lax.cond(
                change_stage,
                lambda: jnp.zeros((self.episode_window,), dtype=jnp.float32),
                lambda: state.last_returns,  # type: ignore
            )
            current_stage_episodes = jax.lax.cond(
                change_stage,
                lambda: jnp.zeros((), dtype=jnp.int32),
                lambda: state.current_stage_episodes,  # type: ignore
            )
            returns_index = jax.lax.cond(
                change_stage,
                lambda: jnp.zeros((), dtype=jnp.int32),
                lambda: state.returns_index,  # type: ignore
            )
            task = self.curriculum.sample(stage, sample_key)

        options = CurriculumResetOptions(task=task)  # set the task for the env reset
        re_state, obs = super().reset(reset_key, state, params, options)
        state = CurriculumState(
            state=re_state,
            curriculum_stage=stage,
            last_returns=last_returns,
            returns_index=returns_index,
            current_stage_episodes=current_stage_episodes,
        )
        return state, obs

    @eqx.filter_jit
    def cheap_reset(
        self,
        key: jax.Array,
        state: CurriculumState,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[CurriculumState, EnvObservation[TObsFeatures]]:
        raise NotImplementedError()

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: CurriculumState,
        action: jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[CurriculumState, TObsFeatures]:
        transition = super().step(key, state, action, params)
        new_returns = state.last_returns.at[state.returns_index].set(
            jax.nn.relu(transition.reward)  # binary success indicator
        )
        last_returns = jnp.where(transition.done, new_returns, state.last_returns)
        returns_index = jnp.where(
            transition.done,
            (state.returns_index + 1) % self.episode_window,
            state.returns_index,
        )
        num_episodes = state.current_stage_episodes + transition.done.astype(jnp.int32)
        new_state = CurriculumState(
            state=transition.state,
            curriculum_stage=state.curriculum_stage,
            last_returns=last_returns,
            returns_index=returns_index,
            current_stage_episodes=num_episodes,
        )
        return transition._replace(state=new_state)  # type: ignore
