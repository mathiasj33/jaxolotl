from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxltl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxltl.environments.wrappers import EnvWrapper
from jaxltl.environments.wrappers.wrapper import WrapperState
from jaxltl.ltl2action.utils.jax_formula_closure import (
    JaxFormulaClosureGraph,
    JaxFormulaGraph,
)
from jaxltl.ltl2action.wrappers.curriculum_wrapper import CurriculumResetOptions


class ResetOptions(NamedTuple):
    task: JaxFormulaClosureGraph


class FormulaClosureState(WrapperState):
    closure: JaxFormulaClosureGraph
    closure_state: jax.Array


class FormulaGraphObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation extended with a formula graph."""

    graph: JaxFormulaGraph

    @classmethod
    def from_obs(
        cls,
        obs: EnvObservation[TObsFeatures],
        graph: JaxFormulaGraph,
    ):
        return cls(features=obs.features, graph=graph)


class FormulaClosureWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, CurriculumResetOptions]):
    """A wrapper that add formula graph information to the observations. Tracks formula
    progression using a JaxFormulaClosureGraph."""

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
        state: FormulaClosureState | None,
        params: TEnvParams,
        options: ResetOptions | None,
    ) -> tuple[FormulaClosureState, FormulaGraphObservation[TObsFeatures]]:
        assert options is not None, "FormulaResetOptions must be provided to reset."
        re_state, obs = super().reset(key, state, params, options)  # type: ignore
        state = FormulaClosureState(
            state=re_state,
            closure=options.task,
            closure_state=options.task.initial_state,
        )
        formula_graph = state.closure.get_graph(state.closure_state)
        formula_obs = FormulaGraphObservation.from_obs(obs, formula_graph)
        return state, formula_obs

    @eqx.filter_jit
    def cheap_reset(self, key, state, params, options):
        raise NotImplementedError()

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: FormulaClosureState,
        action: jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[FormulaClosureState, TObsFeatures]:
        transition = super().step(key, state, action, params)
        assignment = self._env.map_assignment_to_index(transition.propositions)

        # update formula closure state
        next_closure_state = state.closure.get_next_state(
            state.closure_state, assignment
        )

        # compute reward and termination
        is_true = next_closure_state == state.closure.true_state
        is_false = next_closure_state == state.closure.false_state
        reward = jax.lax.cond(
            is_true,
            lambda: 1.0,
            lambda: jax.lax.cond(is_false, lambda: -1.0, lambda: 0.0),
        )
        terminated = jnp.logical_or(is_true, is_false)

        # update observation and state
        next_graph = state.closure.get_graph(next_closure_state)
        next_obs = FormulaGraphObservation.from_obs(transition.observation, next_graph)
        new_state = FormulaClosureState(
            state=transition.state,
            closure=state.closure,
            closure_state=next_closure_state,
        )
        return EnvTransition(
            state=new_state,
            observation=next_obs,
            reward=reward,
            terminated=jnp.logical_or(transition.terminated, terminated),
            truncated=transition.truncated,
            terminal_observation=FormulaGraphObservation.from_obs(
                transition.terminal_observation,
                next_graph,
            ),
            propositions=transition.propositions,
            info=transition.info | {"is_sink": is_false},
        )
