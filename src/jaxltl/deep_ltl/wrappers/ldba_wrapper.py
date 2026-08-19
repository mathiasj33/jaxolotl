from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl import eqx_utils
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers import EnvWrapper
from jaxltl.environments.wrappers.wrapper import WrapperState
from jaxltl.ltl.automata.jax_ldba import JaxLDBA


class ResetOptions(NamedTuple):
    # the task as returned by preprocess_formulas: LDBA and batched reach avoid sequences
    # for each LDBA state
    task: tuple[JaxLDBA, JaxReachAvoidSequence]


class LDBAWrapperState[TObsFeatures: NamedTuple](WrapperState):
    ldba: JaxLDBA
    state_to_seqs: JaxReachAvoidSequence
    ldba_state: jax.Array

    # Epsilon-related state
    obs: EnvObservation[TObsFeatures]  # last observation
    propositions: jax.Array  # last propositions
    info: dict  # last info


class LDBAWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, ResetOptions]):
    """A wrapper that tracks task progression through an LDBA."""

    overwrite_finite: bool

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, ResetOptions]
            | Environment[Any, TEnvParams, TObsFeatures, ResetOptions]
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
        propositions = self._env.compute_propositions(re_state, params)
        ldba, seqs = options.task
        if self.overwrite_finite:
            ldba = ldba._replace(finite=jnp.ones_like(ldba.finite, dtype=jnp.bool))
        state = LDBAWrapperState(
            state=re_state,
            ldba=ldba,
            state_to_seqs=seqs,
            ldba_state=ldba.initial_state,
            obs=obs,
            propositions=propositions,
            info={"is_sink": jnp.zeros((), dtype=jnp.bool_)},
        )
        return state, obs

    @eqx.filter_jit
    def cheap_reset(
        self,
        key: jax.Array,
        state: LDBAWrapperState,
        params: TEnvParams,
        options: ResetOptions | None = None,
    ) -> tuple[LDBAWrapperState, EnvObservation[TObsFeatures]]:
        raise NotImplementedError()

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: LDBAWrapperState,
        action: tuple[jax.Array, jax.Array],  # (env_action, epsilon (int32))
        params: TEnvParams,
    ) -> EnvTransition[LDBAWrapperState, TObsFeatures]:
        env_action, epsilon_action = action
        env_transition = super().step(key, state, env_action, params)
        env_transition.info["is_sink"] = jnp.zeros(
            (), dtype=jnp.bool_
        )  # placeholder, will be updated below
        eps_transition = self._epsilon_step(state)
        transition: EnvTransition = eqx_utils.pytree_where(
            epsilon_action.astype(jnp.bool), eps_transition, env_transition
        )
        assignment = self._env.map_assignment_to_index(transition.propositions)
        next_ldba_state, is_accepting = state.ldba.get_next_state(
            state.ldba_state, assignment
        )
        next_eps_state = state.ldba.get_next_epsilon_state(state.ldba_state)
        next_ldba_state = jnp.where(
            epsilon_action.astype(jnp.bool), next_eps_state, next_ldba_state
        )
        is_accepting = jnp.where(  # epsilon transitions cannot be accepting
            epsilon_action.astype(jnp.bool), False, is_accepting
        )
        reward = is_accepting.astype(jnp.int32)
        is_sink = state.ldba.is_sink_state(state.ldba_state)
        terminated = jnp.logical_or(
            is_sink, jnp.logical_and(state.ldba.finite, is_accepting)
        )
        info = transition.info | {"is_sink": is_sink}
        new_state = LDBAWrapperState(
            state=transition.state,
            ldba=state.ldba,
            state_to_seqs=state.state_to_seqs,
            ldba_state=next_ldba_state,
            obs=transition.observation,
            info=info,
            propositions=transition.propositions,
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

    def _epsilon_step(
        self, state: LDBAWrapperState
    ) -> EnvTransition[WrapperState, TObsFeatures]:
        """Transition corresponding to an epsilon action."""
        return EnvTransition(
            state=state.state,
            observation=state.obs,
            reward=jnp.zeros(()),
            terminated=jnp.zeros((), dtype=jnp.bool),
            truncated=jnp.zeros((), dtype=jnp.bool),
            terminal_observation=state.obs,
            propositions=state.propositions,
            info=state.info,
        )
