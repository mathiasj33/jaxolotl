from typing import Any
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.environment import Environment
from jaxltl.genz_ltl.wrappers.subgoal_wrapper import SubgoalWrapper
from jaxltl.ltl.logic.assignment import Assignment


class _DummyParams(eqx.Module):
    max_steps_in_episode: int = 10


class FakeEnv(Environment):
    def __init__(self):
        props = ("p", "q", "r")
        super().__init__(default_params=_DummyParams(), propositions=props)

    def _reset(self, key, state, params, options=None):
        return eqx.Module()

    def _cheap_reset(self, key, state, params, options=None):
        return state

    def _step(self, key, state, action, params):
        return state, jnp.array(0.0), jnp.array(False), {}

    def _compute_obs(self, state, params):
        return eqx.Module()

    def compute_propositions(self, state, params):
        return -jnp.ones((len(self.propositions),), dtype=jnp.int32)

    def _observation_space(self, params):
        raise NotImplementedError()

    def _action_space(self, params):
        raise NotImplementedError()

    @staticmethod
    def assignments():
        # zero-or-one propositions: [{'p'},{'q'},{'r'},{}]
        return Assignment.zero_or_one_propositions(set(("p", "q", "r")))

    def get_renderer(self, params, **kwargs):
        raise NotImplementedError()


# --- Mock States for Testing ---


class DummySubgoalState(eqx.Module):
    """Mocks the SubgoalState to return an underlying state via unwrapped()."""

    env_state: Any

    def unwrapped(self):
        return self.env_state


class MockComplexEnvState(eqx.Module):
    """Mocks zone_env_nm.EnvState to contain the masked_colors array."""

    masked_colors: jax.Array


def test_sample_new_goal_excludes_current_assignment_and_reach():
    env = FakeEnv()
    wrapper = SubgoalWrapper(env)

    num_assignments = len(env.assignments())
    empty_assignment_idx = env.assignments().index(Assignment(frozenset()))
    key = jax.random.key(0)

    # -------------------------------------------------------------------------
    # TEST 1: Standard behavior (No green exclusion)
    # -------------------------------------------------------------------------
    basic_state = DummySubgoalState(env_state=eqx.Module())

    for a in range(num_assignments):
        key, subkey = jax.random.split(key)
        for _ in range(50):
            # Pass the basic_state to match the new signature
            subgoal = wrapper._sample_new_goal(
                jnp.array(a, dtype=jnp.int32),
                basic_state,  # type: ignore
                subkey,
            )
            reach = int(subgoal.reach)
            avoid = list(subgoal.avoid.tolist())

            assert reach != a
            assert reach != empty_assignment_idx
            assert a not in avoid
            assert reach not in avoid
            assert empty_assignment_idx not in avoid

            reach_assignment = env.assignments()[reach]
            for p in reach_assignment:
                idx = env.propositions.index(p)
                assert subgoal.reach_one_hot[idx] == 1
            assert jnp.sum(subgoal.reach_one_hot) == 1
            for idx in avoid:
                if idx == -1:
                    continue
                assert subgoal.avoid_one_hot[idx] == 1
            assert jnp.sum(subgoal.avoid_one_hot) == len([x for x in avoid if x != -1])

    # -------------------------------------------------------------------------
    # TEST 2: Complex behavior (Green is excluded)
    # -------------------------------------------------------------------------
    GREEN_ASSIGNMENT_IDX = 1

    # Create an array where green (index 1) is marked as masked/excluded
    masked_colors = jnp.zeros(num_assignments, dtype=bool)
    masked_colors = masked_colors.at[GREEN_ASSIGNMENT_IDX].set(True)

    # Wrap it in our mock complex state
    complex_state = MockComplexEnvState(masked_colors=masked_colors)
    subgoal_complex_state = DummySubgoalState(env_state=complex_state)

    # Patch the class reference in the wrapper file so isinstance() evaluates to True
    # Update the module path below to exactly match where _sample_new_goal checks it.
    patch_path = "jaxltl.genz_ltl.wrappers.subgoal_wrapper.zone_env_nm.EnvState"

    with patch(patch_path, MockComplexEnvState):
        for a in range(num_assignments):
            key, subkey = jax.random.split(key)
            for _ in range(50):
                subgoal = wrapper._sample_new_goal(
                    jnp.array(a, dtype=jnp.int32),
                    subgoal_complex_state,  # type: ignore
                    subkey,
                )
                reach = int(subgoal.reach)
                avoid = list(subgoal.avoid.tolist())

                # The primary new assertion: reach must never be green
                assert reach != GREEN_ASSIGNMENT_IDX

                # Standard assertions must still hold true
                assert reach != a
                assert reach != empty_assignment_idx
                assert a not in avoid
                assert reach not in avoid
                assert empty_assignment_idx not in avoid
