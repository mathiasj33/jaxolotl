"""SemLTL LDBA wrapper handling multiple epsilon transitions and semantic embeddings."""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum.wrapper import CurriculumResetOptions
from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.ltl.automata.jax_semantic_ldba import JaxSemanticLDBA


class SemanticLDBAWrapperState[TObsFeatures: NamedTuple](WrapperState):
    ldba: JaxSemanticLDBA  # current LDBA
    ldba_state: jax.Array  # current LDBA state

    # Epsilon-related state
    obs: EnvObservation[TObsFeatures]  # last observation
    propositions: jax.Array  # last propositions


class SemanticLDBAObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation extended with semantic embeddings."""

    embedding: jax.Array  # LDBA state embedding
    epsilon_embeddings: jax.Array  # (max_eps_transitions, embedding_dim)
    epsilon_mask: jax.Array  # (max_eps_transitions,), bool

    @classmethod
    def from_obs(
        cls,
        obs: EnvObservation[TObsFeatures],
        embedding: jax.Array,
        epsilon_embeddings: jax.Array,
        epsilon_enabled: jax.Array,
    ):
        return cls(
            features=obs.features,
            embedding=embedding,
            epsilon_embeddings=epsilon_embeddings,
            epsilon_mask=epsilon_enabled,
        )


class SemanticLDBAWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """A wrapper that adds semantic LDBA embeddings to observations, and keeps track of
    progress through the LDBA."""

    overwrite_finite: bool

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
        overwrite_finite: bool = False,
    ):
        super().__init__(env)
        self.overwrite_finite = overwrite_finite

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: SemanticLDBAWrapperState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[SemanticLDBAWrapperState, SemanticLDBAObservation[TObsFeatures]]:
        assert options is not None, "CurriculumResetOptions must be provided to reset."
        re_state, obs = super().reset(key, state, params, options)
        propositions = self._env.compute_propositions(re_state, params)
        state = SemanticLDBAWrapperState(
            state=re_state,
            ldba=options.task,
            ldba_state=options.task.initial_state,
            obs=obs,
            propositions=propositions,
        )
        embedding = state.ldba.get_embedding(state.ldba_state)
        eps_embeddings = state.ldba.get_epsilon_embeddings(state.ldba_state)
        assignment = self._env.map_assignment_to_index(propositions)
        epsilon_enabled = self._epsilon_enabled(
            state.ldba, state.ldba_state, assignment
        )
        return state, SemanticLDBAObservation.from_obs(
            obs, embedding, eps_embeddings, epsilon_enabled
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: SemanticLDBAWrapperState,
        action: tuple[jax.Array, jax.Array],  # (env_action, epsilon (int32))
        params: TEnvParams,
    ) -> EnvTransition[SemanticLDBAWrapperState, TObsFeatures]:
        env_action, epsilon_action = action
        max_eps_transitions = state.ldba.epsilon_transitions.shape[1]
        execute_eps = epsilon_action != max_eps_transitions
        env_transition = super().step(key, state, env_action, params)
        env_transition.info["is_sink"] = jnp.zeros(
            (), dtype=jnp.bool_
        )  # placeholder, will be updated below
        transition: EnvTransition = jax.lax.cond(
            execute_eps,
            lambda: self._epsilon_step(state, env_transition.info),
            lambda: env_transition,
        )

        assignment = self._env.map_assignment_to_index(transition.propositions)
        next_ldba_state, is_accepting = state.ldba.get_next_state(
            state.ldba_state, assignment
        )
        next_epsilon_ldba_state = state.ldba.get_next_epsilon_state(
            state.ldba_state, epsilon_action
        )
        next_ldba_state = jnp.where(
            execute_eps, next_epsilon_ldba_state, next_ldba_state
        )
        is_accepting = jnp.where(execute_eps, False, is_accepting)
        is_sink = state.ldba.is_sink_state(next_ldba_state)

        reward = jax.lax.cond(
            is_accepting,
            lambda: 1.0,
            lambda: jax.lax.cond(is_sink, lambda: -1.0, lambda: 0.0),
        )
        finite = jnp.logical_or(state.ldba.finite, self.overwrite_finite)
        terminated = jnp.logical_and(is_accepting, finite)
        terminated = jnp.logical_or(terminated, is_sink)

        info = transition.info | {"is_sink": is_sink}
        new_state = SemanticLDBAWrapperState(
            state=transition.state,
            ldba=state.ldba,
            ldba_state=next_ldba_state,
            obs=transition.observation,
            propositions=transition.propositions,
        )

        embedding = new_state.ldba.get_embedding(new_state.ldba_state)
        eps_embeddings = new_state.ldba.get_epsilon_embeddings(new_state.ldba_state)
        epsilon_enabled = self._epsilon_enabled(
            new_state.ldba, new_state.ldba_state, assignment
        )
        next_obs = SemanticLDBAObservation.from_obs(
            transition.observation, embedding, eps_embeddings, epsilon_enabled
        )
        return EnvTransition(
            state=new_state,
            observation=next_obs,
            reward=reward,
            terminated=jnp.logical_or(transition.terminated, terminated),
            truncated=transition.truncated,
            terminal_observation=SemanticLDBAObservation.from_obs(
                transition.terminal_observation,
                embedding,
                eps_embeddings,
                epsilon_enabled,
            ),
            propositions=transition.propositions,
            info=info,
        )

    def _epsilon_step(
        self, state: SemanticLDBAWrapperState, info: dict
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
            info=info,
        )

    def _epsilon_enabled(
        self, ldba: JaxSemanticLDBA, ldba_state: jax.Array, assignment_index: jax.Array
    ) -> jax.Array:
        """Returns a boolean array indicating which epsilon actions can be taken. An
        epsilon action can only be taken if doing so would not immediately lead to a
        sink state given the current assignment.

        Returns:
            epsilon_enabled: (max_eps_transitions,), bool
        """
        next_epsilon_states = ldba.get_next_epsilon_states(ldba_state)
        enabled = next_epsilon_states != -1

        # State after taking epsilon then the env transition
        next_states = jnp.where(
            enabled, ldba.transitions[next_epsilon_states, assignment_index], ldba_state
        )  # (max_eps_transitions,)
        is_valid = jnp.logical_not(ldba.is_sink_state(next_states))

        return jnp.logical_and(enabled, is_valid)
