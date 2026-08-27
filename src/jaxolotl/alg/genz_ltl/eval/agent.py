from typing import override

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl import eqx_utils
from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.alg.genz_ltl.wrappers.ldba_subgoal_wrapper import LDBAWrapperState
from jaxolotl.alg.genz_ltl.wrappers.subgoal_wrapper import SubgoalObservation
from jaxolotl.environments.environment import EnvObservation
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eval.agent import Agent
from jaxolotl.rl.actor_critic import ActorCritic


class GenZLTLAgent(Agent[LDBAWrapperState]):
    """Agent for GenZ-LTL that selects a new reach-avoid subgoal when the LDBA state
    changes. Also implements a timeout-switching mechanism."""

    ldba_state: jax.Array
    subgoals: JaxReachAvoidSubgoal
    subgoal_indices: jax.Array  # indices of the currently selected subgoals (num_envs,)
    vmap_choose_subgoals: bool
    timeout_steps: int
    subgoal_mask: jax.Array  # (num_envs, max_num_subgoals)
    subgoal_steps: jax.Array  # (num_envs,)

    @override
    @classmethod
    def instantiate(
        cls,
        model: ActorCritic,
        vmap_choose_subgoals: bool,
        timeout_steps: int | None = None,
    ) -> "Agent":
        return cls(
            model,
            None,  # type: ignore
            None,  # type: ignore
            None,  # type: ignore
            vmap_choose_subgoals,
            timeout_steps,  # type: ignore
            None,  # type: ignore
            None,  # type: ignore
        )

    @override
    def get_action(self, obsv: EnvObservation) -> distrax.Distribution:
        subgoal_obsv = SubgoalObservation.from_obs(obsv, self.subgoals)
        return self.model.get_action(subgoal_obsv)

    @override
    def update(
        self,
        obsv: EnvObservation,
        state: LDBAWrapperState,
        props: jax.Array,
        env: EnvWrapper,
    ) -> "GenZLTLAgent":
        num_envs = state.ldba_state.shape[0]
        max_num_subgoals = state.state_to_subgoals.reach.shape[-1]
        if self.subgoals is None:
            subgoal_mask = jnp.zeros((num_envs, max_num_subgoals), dtype=bool)
            subgoals, indices = self._choose_subgoals(
                state.ldba_state, state.state_to_subgoals, obsv, subgoal_mask
            )
            return GenZLTLAgent(
                model=self.model,
                ldba_state=state.ldba_state,
                subgoals=subgoals,
                subgoal_indices=indices,
                vmap_choose_subgoals=self.vmap_choose_subgoals,
                timeout_steps=self.timeout_steps,
                subgoal_mask=subgoal_mask,
                subgoal_steps=jnp.zeros((num_envs,), dtype=jnp.int32),
            )

        ldba_state_changed = state.ldba_state != self.ldba_state
        timed_out = jnp.zeros((num_envs,), dtype=bool)
        subgoal_mask = self.subgoal_mask
        if self.timeout_steps is not None:
            timed_out = self.subgoal_steps >= self.timeout_steps
            current_mask_vals = subgoal_mask[jnp.arange(num_envs), self.subgoal_indices]
            subgoal_mask = subgoal_mask.at[
                jnp.arange(num_envs), self.subgoal_indices
            ].set(current_mask_vals | timed_out)
            all_masked = jnp.all(
                (
                    subgoal_mask
                    | (
                        state.state_to_subgoals.reach[
                            jnp.arange(num_envs), self.ldba_state
                        ]
                        == -1
                    )
                ),
                axis=-1,
            )
            # Reset subgoals if none are left
            subgoal_mask = subgoal_mask & ~all_masked[:, None]

        subgoal_mask = jnp.where(ldba_state_changed[:, None], False, subgoal_mask)
        new_subgoals, indices = self._choose_subgoals(
            state.ldba_state, state.state_to_subgoals, obsv, subgoal_mask
        )
        subgoals = eqx_utils.pytree_where(
            ldba_state_changed | timed_out, new_subgoals, self.subgoals
        )
        indices = jnp.where(
            ldba_state_changed | timed_out, indices, self.subgoal_indices
        )
        subgoal_steps = jnp.where(
            ldba_state_changed | timed_out, 0, self.subgoal_steps + 1
        )
        return GenZLTLAgent(
            model=self.model,
            ldba_state=state.ldba_state,
            subgoals=subgoals,
            subgoal_indices=indices,
            vmap_choose_subgoals=self.vmap_choose_subgoals,
            timeout_steps=self.timeout_steps,
            subgoal_mask=subgoal_mask,
            subgoal_steps=subgoal_steps,
        )

    @eqx.filter_jit
    def _choose_subgoals(
        self,
        ldba_state: jax.Array,
        batched_subgoals: JaxReachAvoidSubgoal,
        obsv: EnvObservation,
        subgoal_mask: jax.Array,
    ) -> tuple[JaxReachAvoidSubgoal, jax.Array]:
        """Selects the best reach-avoid subgoal for each environment based on the
        current observation and LDBA state.

        Returns:
            - new_subgoals: JaxReachAvoidSubgoal with shape (num_envs, ...)
            - subgoal_indices: jax.Array of shape (num_envs,) with the indices of the selected subgoals
        """

        def choose_subgoal_for_env(
            inputs: tuple[jax.Array, JaxReachAvoidSubgoal, jax.Array, EnvObservation],
        ) -> tuple[JaxReachAvoidSubgoal, jax.Array]:
            ldba_state, subgoals, subgoal_mask, obs = inputs
            # ldba_state: int
            # obs: EnvObservation
            state_subgoals = jax.tree.map(
                lambda x: x[ldba_state], subgoals
            )  # (num_subgoals,)
            num_subgoals = state_subgoals.reach.shape[0]
            batched_obs = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None, ...], (num_subgoals,) + x.shape), obs
            )
            subgoal_obsv = SubgoalObservation.from_obs(batched_obs, state_subgoals)
            values = self.model.get_value(subgoal_obsv)  # (num_subgoals,)
            costs = self.model.get_cost_value(  # type: ignore
                subgoal_obsv
            )  # (num_subgoals,)
            lags = self.model.get_lagrangian(  # type: ignore
                subgoal_obsv
            )  # (num_subgoals,)
            scores = values - lags * costs
            mask = state_subgoals.reach == -1
            if subgoal_mask is not None:
                mask = mask | subgoal_mask
            scores = jnp.where(mask, -jnp.inf, scores)
            best_index = jnp.argmax(scores)  # type: ignore
            best_subgoal = jax.tree.map(lambda x: x[best_index], state_subgoals)
            return best_subgoal, best_index

        batch_size = ldba_state.shape[0] if self.vmap_choose_subgoals else 1
        return eqx_utils.batch_map(
            choose_subgoal_for_env,
            (ldba_state, batched_subgoals, subgoal_mask, obsv),
            batch_size=batch_size,
        )
