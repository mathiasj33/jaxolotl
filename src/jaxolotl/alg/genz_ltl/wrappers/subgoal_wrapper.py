import logging
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum import CurriculumResetOptions
from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState
from jaxolotl.environments.zone_env import zone_env
from jaxolotl.ltl.logic.assignment import Assignment

logger = logging.getLogger(__name__)


class SubgoalState(WrapperState):
    """State for GenZ-LTL one-subgoal execution."""

    goal: JaxReachAvoidSubgoal


class SubgoalObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation returned by SubgoalWrapper."""

    subgoal: JaxReachAvoidSubgoal

    @classmethod
    def from_obs(
        cls,
        obs: EnvObservation[TObsFeatures],
        subgoal: JaxReachAvoidSubgoal,
    ):
        return cls(features=obs.features, subgoal=subgoal)


class SubgoalWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """A wrapper for one-subgoal-at-a-time training with ZoneEnv observation reduction.

    If terminate_on_success is set, the episode terminates once the subgoal is reached;
    otherwise a new subgoal is sampled in-episode and execution continues.
    """

    terminate_on_success: bool

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
        ),
        terminate_on_success: bool = False,
    ):
        super().__init__(env)
        self.terminate_on_success = terminate_on_success

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: SubgoalState | None,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[SubgoalState, SubgoalObservation]:
        assert options is not None, "CurriculumResetOptions must be provided to reset."
        re_state, obs = super().reset(key, state, params, options)
        subgoal = options.task
        reduced_obs = SubgoalObservation.from_obs(obs, subgoal)
        wrapped_state = SubgoalState(state=re_state, goal=subgoal)
        return wrapped_state, reduced_obs

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: SubgoalState,
        action: jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[SubgoalState, TObsFeatures]:
        key, subkey = jax.random.split(key)
        transition = super().step(key, state, action, params)
        assignment = self._env.map_assignment_to_index(transition.propositions)
        avoided = jnp.logical_not(jnp.any(state.goal.avoid == assignment))
        reached = jnp.logical_and(state.goal.reach == assignment, avoided)
        reward = jax.lax.cond(reached, lambda: 1.0, lambda: 0.0)
        # We assume here that termination is a cost (e.g. hitting a wall in ZoneEnv).
        safe = avoided & ~transition.terminated
        cost = jax.lax.cond(safe, lambda: -1.0, lambda: 1.0)
        terminated = transition.terminated | (cost > 0)
        if self.terminate_on_success:
            terminated = terminated | reached
            goal = state.goal
        else:
            new_goal = self._sample_new_goal(assignment, state, subkey)
            goal = jax.lax.cond(reached, lambda: new_goal, lambda: state.goal)
        reduced_obs = SubgoalObservation.from_obs(transition.observation, goal)
        new_state = SubgoalState(state=transition.state, goal=goal)
        return EnvTransition(
            state=new_state,
            observation=reduced_obs,
            reward=reward,
            terminated=terminated,
            truncated=transition.truncated,
            terminal_observation=reduced_obs,
            propositions=transition.propositions,
            info=transition.info
            | {
                "cost": cost,
                "subgoal_success": reached.astype(jnp.bool_),
            },
        )

    def _sample_new_goal(
        self, assignment: jax.Array, state: SubgoalState, key: jax.Array
    ) -> JaxReachAvoidSubgoal:
        # Sample a new reach that is not the current assignment.
        reach_key, avoid_key = jax.random.split(key)
        num_assignments = len(self._env.assignments()) - 1  # exclude empty assignment

        valid_reach_mask = jnp.ones(num_assignments, dtype=bool)
        valid_reach_mask = valid_reach_mask.at[assignment].set(False)
        unwrapped_state = state.unwrapped()
        if isinstance(unwrapped_state, zone_env.EnvState):
            assignment_to_index = {a: i for i, a in enumerate(self._env.assignments())}
            green_assignment_idx = assignment_to_index[Assignment("green")]
            exclude_green = unwrapped_state.masked_colors[green_assignment_idx]
            # Conditionally exclude the green index
            valid_reach_mask = jnp.where(
                exclude_green,
                valid_reach_mask.at[green_assignment_idx].set(False),
                valid_reach_mask,
            )

        # Sample a valid index using the mask (convert boolean to normalized float probabilities)
        probs = valid_reach_mask.astype(jnp.float32)
        probs = probs / jnp.sum(probs)
        reach_idx = jax.random.choice(reach_key, num_assignments, p=probs)

        # Include each eligible avoid assignment independently with probability 0.5.
        indices = jnp.arange(num_assignments)
        eligible_avoid = (indices != assignment) & (indices != reach_idx)
        included = jax.random.bernoulli(avoid_key, 0.5, shape=(num_assignments,))
        avoid = jnp.where(eligible_avoid & included, indices, -1)
        avoid = jnp.sort(avoid, descending=True)
        # Pad to (num_assignments,) with -1.
        avoid = jnp.pad(
            avoid,
            (0, len(self._env.assignments()) - len(avoid)),
            constant_values=-1,
        )

        # Create one-hot encodings for reach and avoid
        reach_props = self._env.assignments_array[reach_idx]
        reach_one_hot = (
            (jnp.arange(len(self._env.propositions)) == reach_props[:, None])
            .any(axis=0)
            .astype(jnp.int32)
        )
        avoid_one_hot = (
            (jnp.arange(len(self._env.assignments())) == avoid[:, None])
            .any(axis=0)
            .astype(jnp.int32)
        )
        return JaxReachAvoidSubgoal(
            reach=reach_idx,
            avoid=avoid,
            reach_one_hot=reach_one_hot,
            avoid_one_hot=avoid_one_hot,
        )
