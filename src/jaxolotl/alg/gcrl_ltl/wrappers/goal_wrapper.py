"""Simple wrapper that provides propositional goals."""

import logging
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum import CurriculumResetOptions
from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState

logger = logging.getLogger(__name__)


class GoalState(WrapperState):
    goal: jax.Array  # int, proposition index


class GoalObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    goal: jax.Array

    @classmethod
    def from_obs(
        cls,
        obs: EnvObservation[TObsFeatures],
        goal: jax.Array,
    ):
        return cls(features=obs.features, goal=goal)


class GoalWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """A wrapper for GCRL-LTL proposition-conditioned training."""

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
    ):
        super().__init__(env)

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: GoalState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[GoalState, GoalObservation]:
        assert options is not None, "CurriculumResetOptions must be provided to reset."
        re_state, obs = super().reset(key, state, params, options)
        goal = options.task
        goal_obs = GoalObservation.from_obs(obs, goal)
        wrapped_state = GoalState(state=re_state, goal=goal)
        return wrapped_state, goal_obs

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: GoalState,
        action: jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[GoalState, TObsFeatures]:
        key, subkey = jax.random.split(key)
        transition = super().step(key, state, action, params)
        reached = jnp.any(state.goal == transition.propositions)
        reward = jax.lax.cond(reached, lambda: 1.0, lambda: 0.0)
        terminated = transition.terminated | (reward > 0)
        reduced_obs = GoalObservation.from_obs(transition.observation, state.goal)
        new_state = GoalState(state=transition.state, goal=state.goal)
        return EnvTransition(
            state=new_state,
            observation=reduced_obs,
            reward=reward,
            terminated=terminated,
            truncated=transition.truncated,
            terminal_observation=reduced_obs,
            propositions=transition.propositions,
            info=transition.info,
        )
