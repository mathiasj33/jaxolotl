from typing import NamedTuple, override

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl import eqx_utils
from jaxltl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxltl.deep_ltl.wrappers.ldba_wrapper import LDBAWrapperState
from jaxltl.environments.environment import EnvObservation
from jaxltl.environments.wrappers.wrapper import EnvWrapper
from jaxltl.eval.agent import Agent
from jaxltl.rl.actor_critic import ActorCritic


class SequenceObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation returned by CurriculumWrapper."""

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


class DeepLTLAgent(Agent[LDBAWrapperState]):
    """Agent for DeepLTL that selects a new reach-avoid sequence when the LDBA state
    changes."""

    ldba_state: jax.Array
    seqs: JaxReachAvoidSequence
    eps_enabled: jax.Array
    vmap_choose_sequences: bool

    @override
    @classmethod
    def instantiate(
        cls,
        model: ActorCritic,
        vmap_choose_sequences: bool,
    ) -> "Agent":
        return cls(model, None, None, None, vmap_choose_sequences)  # type: ignore

    @override
    def get_action(self, obsv: EnvObservation) -> distrax.Distribution:
        seq_obsv = SequenceObservation.from_obs(obsv, self.seqs, self.eps_enabled)
        return self.model.get_action(seq_obsv)

    @override
    def update(
        self,
        obsv: EnvObservation,
        state: LDBAWrapperState,
        props: jax.Array,
        env: EnvWrapper,
    ) -> "DeepLTLAgent":
        needs_update = state.ldba_state != self.ldba_state
        new_seqs = self._choose_sequences(state.ldba_state, state.state_to_seqs, obsv)
        if self.seqs is None:
            seq = new_seqs
        else:
            seq = eqx_utils.pytree_where(needs_update, new_seqs, self.seqs)

        assignment_index = jax.vmap(env.map_assignment_to_index)(props)
        eps_enabled = jax.vmap(self._is_epsilon_enabled, in_axes=(None, 0, 0))(
            env, seq, assignment_index
        )
        return DeepLTLAgent(
            model=self.model,
            ldba_state=state.ldba_state,
            seqs=seq,
            eps_enabled=eps_enabled,
            vmap_choose_sequences=self.vmap_choose_sequences,
        )

    def _is_epsilon_enabled(
        self, env: EnvWrapper, seq: JaxReachAvoidSequence, assignment_index: jax.Array
    ) -> jax.Array:
        """Returns a boolean indicating if an epsilon action can be taken. This is only
        true if the current step in the reach-avoid sequence is an epsilon transition,
        and the current environment assignment does not violate the next avoid set.
        """
        is_epsilon = seq.reach[0, 0] == len(env._env.assignments())
        is_valid = jnp.logical_or(
            seq.depth <= 1, jnp.all(seq.avoid[1] != assignment_index)
        )
        return jnp.logical_and(is_epsilon, is_valid)

    @eqx.filter_jit
    def _choose_sequences(
        self,
        ldba_state: jax.Array,
        batched_seqs: JaxReachAvoidSequence,
        obsv: EnvObservation,
    ) -> JaxReachAvoidSequence:
        """Selects the best reach-avoid sequence for each environment based on the
        current observation and LDBA state."""

        def choose_sequence_for_env(
            inputs: tuple[jax.Array, JaxReachAvoidSequence, EnvObservation],
        ) -> JaxReachAvoidSequence:
            ldba_state, batched_seqs, obs = inputs
            # ldba_state: int
            # obs: EnvObservation
            state_seqs = jax.tree.map(lambda x: x[ldba_state], batched_seqs)
            num_seqs = state_seqs.reach.shape[0]
            batched_obs = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None, ...], (num_seqs,) + x.shape), obs
            )
            batched_seq_obs = SequenceObservation.from_obs(
                batched_obs,
                state_seqs,
                epsilon_enabled=jnp.ones((num_seqs,), dtype=bool),
            )  # epsilon_enabled is irrelevant for the critic
            scores = self.model.get_value(batched_seq_obs)  # (num_seqs,)
            padded = state_seqs.reach[:, 0, 0] == -1
            scores = jnp.where(padded, -jnp.inf, scores)
            best_index = jnp.argmax(scores)
            best_seq = jax.tree.map(lambda x: x[best_index], state_seqs)
            return best_seq

        batch_size = ldba_state.shape[0] if self.vmap_choose_sequences else 1
        return jax.lax.map(
            choose_sequence_for_env,
            (ldba_state, batched_seqs, obsv),
            batch_size=batch_size,
        )
