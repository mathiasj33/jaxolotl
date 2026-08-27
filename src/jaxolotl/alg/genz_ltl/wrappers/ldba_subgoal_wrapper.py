import dataclasses
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA


class ResetOptions(NamedTuple):
    task: tuple[JaxLDBA, JaxReachAvoidSubgoal]


class LDBAWrapperState(WrapperState):
    ldba: JaxLDBA
    state_to_subgoals: JaxReachAvoidSubgoal
    ldba_state: jax.Array


class LDBASubgoalWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, ResetOptions]):
    """A wrapper that tracks task progression through an LDBA."""

    overwrite_finite: bool

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, ResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, ResetOptions]
        ),
        overwrite_finite: bool = False,
    ):
        super().__init__(env)
        self.overwrite_finite = overwrite_finite

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: LDBAWrapperState | None,
        params: TEnvParams,
        options: ResetOptions | None,
    ) -> tuple[LDBAWrapperState, EnvObservation[TObsFeatures]]:
        assert options is not None, "Reset options must be provided to LDBA wrapper."
        re_state, obs = super().reset(key, state, params, options)
        ldba, subgoals = options.task
        if self.overwrite_finite:
            ldba = dataclasses.replace(
                ldba, finite=jnp.ones_like(ldba.finite, dtype=jnp.bool)
            )
        state = LDBAWrapperState(
            state=re_state,
            ldba=ldba,
            state_to_subgoals=subgoals,
            ldba_state=ldba.initial_state,
        )
        return state, obs

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: LDBAWrapperState,
        action: jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[LDBAWrapperState, TObsFeatures]:
        transition = super().step(key, state, action, params)
        assignment = self._env.map_assignment_to_index(transition.propositions)

        # Check if we apply the epsilon transition
        # TODO: refactor this to be cleaner
        next_epsilon_state = state.ldba.get_next_epsilon_state(state.ldba_state)
        is_epsilon = next_epsilon_state != state.ldba_state
        # shape: (max_num_subgoals, num_assignments)
        next_avoid = state.state_to_subgoals.avoid[next_epsilon_state]
        is_safe = jnp.all(next_avoid != assignment)
        ldba_state = jnp.where(
            is_epsilon & is_safe, next_epsilon_state, state.ldba_state
        )

        next_ldba_state, is_accepting = state.ldba.get_next_state(
            ldba_state, assignment
        )
        reward = is_accepting.astype(jnp.int32)
        is_sink = state.ldba.is_sink_state(ldba_state)
        terminated = jnp.logical_or(
            is_sink, jnp.logical_and(state.ldba.finite, is_accepting)
        )
        info = transition.info | {"is_sink": is_sink}
        new_state = LDBAWrapperState(
            state=transition.state,
            ldba=state.ldba,
            state_to_subgoals=state.state_to_subgoals,
            ldba_state=next_ldba_state,
        )
        return EnvTransition(
            state=new_state,
            observation=transition.observation,
            reward=reward,
            terminated=jnp.logical_or(transition.terminated, terminated),
            truncated=transition.truncated,
            terminal_observation=transition.terminal_observation,
            propositions=transition.propositions,
            info=info,
        )
