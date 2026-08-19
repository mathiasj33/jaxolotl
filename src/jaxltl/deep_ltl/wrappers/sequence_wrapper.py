from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers import EnvWrapper
from jaxltl.environments.wrappers.wrapper import WrapperState
from jaxltl.ltl2action.wrappers.curriculum_wrapper import CurriculumResetOptions


class SequenceState[TObsFeatures: NamedTuple](WrapperState):
    """State for SequenceWrapper."""

    seq: JaxReachAvoidSequence  # current reach-avoid sequence

    # Epsilon-related state
    obs: EnvObservation[TObsFeatures]  # last observation
    propositions: jax.Array  # last propositions
    info: dict  # last info


class SequenceObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation returned by SequenceWrapper."""

    seq: JaxReachAvoidSequence
    epsilon_mask: jax.Array  # bool, indicates if epsilon transition can be taken

    @classmethod
    def from_obs(
        cls,
        obs: EnvObservation[TObsFeatures],
        seq: JaxReachAvoidSequence,
        epsilon_enabled: jax.Array,
    ):
        return cls(features=obs.features, seq=seq, epsilon_mask=epsilon_enabled)


class SequenceWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """A wrapper that adds reach-avoid sequences to observations and keeps track of their completion."""

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
    ):
        super().__init__(env)

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: SequenceState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[SequenceState, SequenceObservation[TObsFeatures]]:
        assert options is not None, "SequenceResetOptions must be provided to reset."
        re_state, obs = super().reset(key, state, params, options)
        propositions = self._env.compute_propositions(re_state, params)
        state = SequenceState(
            state=re_state,
            seq=options.task,
            obs=obs,
            propositions=propositions,
            info={},
        )
        assignment = self._env.map_assignment_to_index(propositions)
        return state, SequenceObservation.from_obs(
            obs, state.seq, self._is_epsilon_enabled(state.seq, assignment)
        )

    @eqx.filter_jit
    def cheap_reset(
        self,
        key: jax.Array,
        state: SequenceState,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[SequenceState, SequenceObservation[TObsFeatures]]:
        raise NotImplementedError()

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: SequenceState,
        action: tuple[jax.Array, jax.Array],  # (env_action, epsilon (int32))
        params: TEnvParams,
    ) -> EnvTransition[SequenceState, TObsFeatures]:
        env_action, epsilon_action = action
        env_transition = super().step(key, state, env_action, params)
        transition = jax.lax.cond(
            epsilon_action.astype(jnp.bool),
            lambda: self._epsilon_step(state),
            lambda: env_transition,
        )
        seq = jax.lax.cond(
            epsilon_action.astype(jnp.bool),
            lambda: state.seq.advance(),
            lambda: state.seq,
        )
        reach = seq.reach[0]  # (num_assignments,)
        avoid = seq.avoid[0]
        assignment = self._env.map_assignment_to_index(transition.propositions)
        avoided = jnp.logical_not(jnp.any(avoid == assignment))
        reached = jnp.logical_and(jnp.any(reach == assignment), avoided)
        seq = jax.lax.cond(reached, lambda: seq.advance(), lambda: seq)
        reached_end = jnp.all(seq.reach[0] == -1)  # Check if reached is just padding
        reward = jax.lax.cond(
            reached_end,
            lambda: 1.0,
            lambda: jax.lax.cond(avoided, lambda: 0.0, lambda: -1.0),
        )
        terminated = jnp.logical_or(reached_end, ~avoided)
        new_state = SequenceState(
            state=transition.state,
            seq=seq,
            obs=transition.observation,
            propositions=transition.propositions,
            info=transition.info,
        )
        return EnvTransition(
            state=new_state,
            observation=SequenceObservation.from_obs(
                transition.observation,
                new_state.seq,
                self._is_epsilon_enabled(new_state.seq, assignment),
            ),
            reward=reward,
            terminated=jnp.logical_or(transition.terminated, terminated),
            truncated=transition.truncated,
            terminal_observation=SequenceObservation.from_obs(
                transition.terminal_observation,
                new_state.seq,
                self._is_epsilon_enabled(new_state.seq, assignment),
            ),
            propositions=transition.propositions,
            info=transition.info,
        )

    def _epsilon_step(
        self, state: SequenceState
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

    def _is_epsilon_enabled(
        self, seq: JaxReachAvoidSequence, assignment_index: jax.Array
    ) -> jax.Array:
        """Returns a boolean indicating if an epsilon action can be taken. This is only
        true if the current step in the reach-avoid sequence is an epsilon transition,
        and the current environment assignment does not violate the next avoid set.
        """
        is_epsilon = seq.reach[0, 0] == len(self._env.assignments())
        is_valid = jnp.logical_or(
            seq.depth <= 1, jnp.all(seq.avoid[1] != assignment_index)
        )
        return jnp.logical_and(is_epsilon, is_valid)
