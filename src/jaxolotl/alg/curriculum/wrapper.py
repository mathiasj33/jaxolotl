"""Curriculum wrapper for curriculum learning.

This is a per-environment wrapper that keeps track of the current curriculum stage and
the stage adopted for the next reset. It samples tasks according to the current stage
and passes them to wrappers further down the stack.
"""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from jaxolotl.alg.curriculum.curriculum import Curriculum
from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState


class CurriculumState(WrapperState):
    """State for CurriculumWrapper."""

    curriculum_stage: jax.Array  # int, stage of the current episode
    adopted_stage: jax.Array  # int, stage adopted for the next reset


class CurriculumResetOptions(NamedTuple):
    """Reset options for wrappers that can be used with a curriculum."""

    task: PyTree


class CurriculumWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TSample,
    TJaxSample: eqx.Module,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """Sample tasks from a curriculum governed by a shared population frontier."""

    curriculum: Curriculum[TSample, TJaxSample]

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
        curriculum: Curriculum[TSample, TJaxSample],
    ):
        super().__init__(env)
        self.curriculum = curriculum

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: CurriculumState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[CurriculumState, EnvObservation[TObsFeatures]]:
        reset_key, sample_key = jax.random.split(key)

        stage = jnp.zeros((), dtype=jnp.int32) if state is None else state.adopted_stage
        task = self.curriculum.sample(stage, sample_key)
        options = CurriculumResetOptions(task=task)
        re_state, obs = super().reset(reset_key, state, params, options)
        # We always pass the same stage as the adopted stage. This will be updated
        # externally by the curriculum manager.
        state = CurriculumState(
            state=re_state, curriculum_stage=stage, adopted_stage=stage
        )
        return state, obs

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: CurriculumState,
        action: jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[CurriculumState, TObsFeatures]:
        transition = super().step(key, state, action, params)
        new_state = CurriculumState(
            state=transition.state,
            curriculum_stage=state.curriculum_stage,
            adopted_stage=state.adopted_stage,
        )
        info = transition.info | {
            "curriculum_episode_stage": state.curriculum_stage,
            "curriculum_episode_success": transition.reward > 0.0,
        }
        return transition._replace(state=new_state, info=info)  # type: ignore
