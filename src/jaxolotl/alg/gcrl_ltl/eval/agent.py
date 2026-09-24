from typing import cast, override

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl import eqx_utils
from jaxolotl.alg.common.epsilon_distribution import EpsilonDistribution
from jaxolotl.alg.common.reach_avoid.jax_sequence import (
    JaxReachAvoidSequence,
)
from jaxolotl.alg.deep_ltl.wrappers.ldba_wrapper import LDBAWrapperState
from jaxolotl.alg.gcrl_ltl.model.gcvf import GCVF
from jaxolotl.alg.gcrl_ltl.reach_avoid.jax_gcrl_sequence import JaxGCRLSequence
from jaxolotl.alg.gcrl_ltl.wrappers.goal_wrapper import GoalObservation
from jaxolotl.environments.environment import EnvObservation
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eval.agent import Agent
from jaxolotl.rl.actor_critic import ActorCritic


class GCRLAgent(Agent[LDBAWrapperState]):
    """GCRL agent that selects sequences based on the goal-conditioned value function.
    Re-plans when the LDBA state changes."""

    ldba_state: jax.Array
    seqs: JaxGCRLSequence  # selected sequences for each environment
    first_step_is_eps: jax.Array  # (num_envs,) bool
    eps_enabled: jax.Array  # (num_envs,) bool
    vmap_choose_sequences: bool
    unsafe_threshold: float  # threshold for unsafe actions
    gcvf: GCVF

    @override
    @classmethod
    def instantiate(
        cls,
        model: ActorCritic,
        vmap_choose_sequences: bool,
        unsafe_threshold: float,
        gcvf: GCVF,
    ) -> "Agent":
        return cls(
            model,
            None,  # type: ignore
            None,  # type: ignore
            None,  # type: ignore
            None,  # type: ignore
            vmap_choose_sequences,
            unsafe_threshold,
            gcvf,
        )

    @override
    def get_action(self, obsv: EnvObservation) -> distrax.Distribution:
        """Returns the action distribution for the current observation, masking out unsafe actions."""

        num_envs = self.ldba_state.shape[0]

        # First, get the action distribution without avoidance based on the current reach proposition
        reach_idx = jnp.where(self.first_step_is_eps, 1, 0)
        reach_props = self.seqs.reach_props[jnp.arange(num_envs), reach_idx]
        goal_obsv = GoalObservation.from_obs(obsv, reach_props)
        action_dist: distrax.Categorical = cast(
            distrax.Categorical, self.model.get_action(goal_obsv)
        )
        logits = cast(jax.Array, action_dist.logits)  # (num_envs, num_actions)

        # Evaluate all avoidance constraints using the value function
        max_avoid = self.seqs.avoid_props.shape[-1]
        if max_avoid > 0:
            repeated_obs = jax.tree.map(
                lambda x: jnp.repeat(x, max_avoid, axis=0), obsv
            )
            avoid_props = self.seqs.avoid_props[
                jnp.arange(num_envs), reach_idx, :
            ]  # (num_envs, max_avoid)
            valid_mask = avoid_props != -1
            avoid_props = jnp.where(
                valid_mask, avoid_props, 0
            )  # set invalid props to 0 to avoid indexing errors
            avoid_obs = GoalObservation.from_obs(repeated_obs, avoid_props.ravel())
            avoid_values = self.model.get_value(avoid_obs)  # (num_envs * max_avoid,)
            avoid_values = avoid_values.reshape(num_envs, max_avoid)

            # Compute original goal values
            goal_values = self.model.get_value(goal_obsv)  # (num_envs,)

            # Check where avoidance values exceed the unsafe threshold
            unsafe_mask = avoid_values > self.unsafe_threshold  # (num_envs, max_avoid)
            unsafe_mask = jnp.logical_and(
                unsafe_mask, avoid_values > goal_values[:, None]
            )
            unsafe_mask = jnp.logical_and(
                unsafe_mask, valid_mask
            )  # ignore invalid props
            unsafe_mask = jnp.any(unsafe_mask, axis=1)  # (num_envs,) bool

            # Get the indices of most unsafe propositions for each environment
            avoid_values = jnp.where(valid_mask, avoid_values, -jnp.inf)
            unsafe_indices = jnp.argmax(avoid_values, axis=1)  # (num_envs,) int
            unsafe_props = avoid_props[
                jnp.arange(num_envs), unsafe_indices
            ]  # (num_envs,) int
            unsafe_props = jnp.where(unsafe_mask, unsafe_props, 0)

            # Get dangerous actions for each environment based on the unsafe propositions
            avoid_obs = GoalObservation.from_obs(obsv, unsafe_props)
            avoid_actions = self.model.get_action(avoid_obs).mode()  # (num_envs,) int

            # Mask out dangerous actions from the original action distribution
            masked_logits = logits.at[jnp.arange(num_envs), avoid_actions].set(-jnp.inf)
            logits = jnp.where(unsafe_mask[:, None], masked_logits, logits)

        # Return distribution with masked logits. Take epsilon if enabled.
        action_dist = distrax.Categorical(logits=logits)
        return EpsilonDistribution(action_dist, jnp.asarray(jnp.inf), self.eps_enabled)

    @override
    def update(
        self,
        obsv: EnvObservation,
        state: LDBAWrapperState,
        props: jax.Array,
        env: EnvWrapper,
    ) -> "GCRLAgent":
        needs_update = state.ldba_state != self.ldba_state
        new_seqs = self._choose_sequences(
            state.ldba_state,
            state.state_to_seqs,  # type: ignore
            obsv,
            eps_index=len(env.propositions),
        )
        if self.seqs is None:
            seq = new_seqs
        else:
            seq = eqx_utils.pytree_where(needs_update, new_seqs, self.seqs)

        assignment_index = jax.vmap(env.map_assignment_to_index)(props)
        eps_enabled = jax.vmap(self._is_epsilon_enabled, in_axes=(None, 0, 0))(
            env, seq, assignment_index
        )
        first_step_is_eps = seq.reach[:, 0, 0] == len(env._env.assignments())
        return GCRLAgent(
            model=self.model,
            ldba_state=state.ldba_state,
            seqs=seq,
            eps_enabled=eps_enabled,
            first_step_is_eps=first_step_is_eps,
            vmap_choose_sequences=self.vmap_choose_sequences,
            unsafe_threshold=self.unsafe_threshold,
            gcvf=self.gcvf,
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
        batched_seqs: JaxGCRLSequence,
        obsv: EnvObservation,
        eps_index: int,
    ) -> JaxGCRLSequence:
        """Selects the best sequence for each environment based on the current observation and LDBA state.
        Uses the GCVF to evaluate the sequences and select the one with the highest value.
        """

        def choose_sequence_for_env(
            inputs: tuple[jax.Array, JaxGCRLSequence, EnvObservation],
        ) -> JaxGCRLSequence:
            ldba_state, batched_seqs, obs = inputs
            # ldba_state: int
            # obs: EnvObservation
            state_seqs = jax.tree.map(lambda x: x[ldba_state], batched_seqs)
            num_seqs = state_seqs.reach.shape[0]
            batched_obs = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None, ...], (num_seqs,) + x.shape), obs
            )

            props = state_seqs.reach_props  # [max_seqs, max_length] int
            # Select first non-epsilon proposition for each sequence to evaluate the initial value.
            physical = (props >= 0) & (props < eps_index)
            first_indices = jnp.argmax(physical, axis=1)
            has_goal = jnp.any(physical, axis=1)
            first_props = props[jnp.arange(num_seqs), first_indices]
            # Use a valid placeholder for sequences containing only epsilon/padding.
            first_props = jnp.where(has_goal, first_props, 0)
            seq_obs = GoalObservation.from_obs(batched_obs, first_props)  # type: ignore
            init_values = self.model.get_value(seq_obs)  # (num_seqs,)
            init_values = jnp.where(init_values < 0, 0.0, init_values)
            init_weights = jnp.where(has_goal, -jnp.log(init_values + 1e-8), 0.0)

            def compute_seq_weights(
                carry: tuple[jax.Array, jax.Array], next_props: jax.Array
            ):
                previous_props, weights = carry  # [num_seqs,]
                is_physical = (next_props >= 0) & (next_props < eps_index)
                # Never pass epsilon or padding indices to the learned embeddings.
                source = jnp.where(previous_props >= 0, previous_props, 0)
                target = jnp.where(is_physical, next_props, 0)
                seq_obs = GoalObservation.from_obs(batched_obs, source)
                values = self.gcvf(seq_obs, target)
                values = jnp.where(values < 0, 0.0, values)
                transition_weights = -jnp.log(values + 1e-8)
                has_transition = is_physical & (previous_props >= 0)
                weights += jnp.where(has_transition, transition_weights, 0.0)
                # Epsilon does not change the last physical goal, so the next
                # physical goal is scored from it even across multiple epsilons.
                previous_props = jnp.where(is_physical, next_props, previous_props)
                return (previous_props, weights), None

            (_, weights), _ = jax.lax.scan(
                compute_seq_weights,
                (-jnp.ones((num_seqs,), dtype=props.dtype), init_weights),
                props.swapaxes(0, 1),
            )

            padded = state_seqs.reach[:, 0, 0] == -1
            weights = jnp.where(padded, jnp.inf, weights)
            best_index = jnp.argmin(weights)
            best_seq = jax.tree.map(lambda x: x[best_index], state_seqs)
            return best_seq

        batch_size = ldba_state.shape[0] if self.vmap_choose_sequences else 1
        return eqx_utils.batch_map(
            choose_sequence_for_env,
            (ldba_state, batched_seqs, obsv),
            batch_size=batch_size,
        )
