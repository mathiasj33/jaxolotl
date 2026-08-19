import logging
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers import EnvWrapper
from jaxltl.environments.wrappers.wrapper import WrapperState
from jaxltl.environments.zone_env_nm import zone_env_nm
from jaxltl.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import JaxReachAvoidSubgoal
from jaxltl.ltl2action.wrappers.curriculum_wrapper import CurriculumResetOptions

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
    """A wrapper for one-subgoal-at-a-time training with ZoneEnv observation reduction."""

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
    def cheap_reset(
        self,
        key: jax.Array,
        state: SubgoalState,
        params: TEnvParams,
        options: CurriculumResetOptions | None = None,
    ) -> tuple[SubgoalState, SubgoalObservation]:
        raise NotImplementedError()

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
        cost = jax.lax.cond(avoided, lambda: 0.0, lambda: 1.0)
        terminated = transition.terminated | (cost > 0)
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
            info=transition.info | {"cost": cost},
        )

    def _sample_new_goal(
        self, assignment: jax.Array, state: SubgoalState, key: jax.Array
    ) -> JaxReachAvoidSubgoal:
        # Sample a new reach that is not the current assignment.
        key, reach_key, avoid_key = jax.random.split(key, 3)
        num_assignments = len(self._env.assignments()) - 1  # exclude empty assignment

        valid_reach_mask = jnp.ones(num_assignments, dtype=bool)
        valid_reach_mask = valid_reach_mask.at[assignment].set(False)
        unwrapped_state = state.unwrapped()
        if isinstance(unwrapped_state, zone_env_nm.EnvState):
            logger.info(
                "Applying zone_env_nm-specific logic to exclude green assignment!"
            )
            GREEN_ASSIGNMENT_IDX = 1  # green is the second assignment in zone_env_nm
            exclude_green = unwrapped_state.masked_colors[GREEN_ASSIGNMENT_IDX]
            # Conditionally exclude the green index based on the tracer
            valid_reach_mask = jnp.where(
                exclude_green,
                valid_reach_mask.at[GREEN_ASSIGNMENT_IDX].set(False),
                valid_reach_mask,
            )

        # Sample a valid index using the mask (convert boolean to normalized float probabilities)
        probs = valid_reach_mask.astype(jnp.float32)
        probs = probs / jnp.sum(probs)
        reach_idx = jax.random.choice(reach_key, num_assignments, p=probs)

        # Sample a new avoid that does not contain the current assignment nor the reach.
        key_size, key_perm = jax.random.split(avoid_key)
        low = jnp.minimum(assignment, reach_idx)  # indices to exclude
        high = jnp.maximum(assignment, reach_idx)

        # Sample a random subset size m in [0, n-2].
        m = jax.random.randint(key_size, shape=(), minval=0, maxval=num_assignments - 1)
        indices = jnp.arange(num_assignments - 2)
        shuffled = jax.random.permutation(key_perm, indices)

        # Shift the indices to account for the excluded reach and assignment.
        shifted = jnp.where(shuffled >= low, shuffled + 1, shuffled)
        shifted = jnp.where(shifted >= high, shifted + 1, shifted)

        # Mask out the other indices.
        mask = jnp.arange(num_assignments - 2) < m
        shifted = jnp.where(mask, shifted, -1)
        shifted = jnp.sort(shifted, descending=True)
        # Pad to (num_assignments,) with -1.
        shifted = jnp.pad(
            shifted,
            (0, len(self._env.assignments()) - len(shifted)),
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
            (jnp.arange(len(self._env.assignments())) == shifted[:, None])
            .any(axis=0)
            .astype(jnp.int32)
        )
        return JaxReachAvoidSubgoal(
            reach=reach_idx,
            avoid=shifted,
            reach_one_hot=reach_one_hot,
            avoid_one_hot=avoid_one_hot,
        )
