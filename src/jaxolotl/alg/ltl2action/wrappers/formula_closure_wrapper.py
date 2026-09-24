from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.alg.curriculum import CurriculumResetOptions
from jaxolotl.alg.ltl2action.utils.jax_formula_closure import (
    JaxFormulaClosureGraph,
    JaxFormulaGraph,
    StaticGraphTable,
)
from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers import EnvWrapper
from jaxolotl.environments.wrappers.wrapper import WrapperState


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


class FormulaIndexObservation[TObsFeatures: NamedTuple](EnvObservation[TObsFeatures]):
    """Observation extended with an index into a static table of unique graphs."""

    graph_index: jax.Array  # () int32
    graph_table: StaticGraphTable = eqx.field(static=True)

    @classmethod
    def from_obs(
        cls,
        obs: EnvObservation[TObsFeatures],
        graph_index: jax.Array,
        graph_table: StaticGraphTable,
    ):
        return cls(
            features=obs.features, graph_index=graph_index, graph_table=graph_table
        )


def _formula_obs(
    obs: EnvObservation,
    closure: JaxFormulaClosureGraph,
    closure_state: jax.Array,
) -> EnvObservation:
    """Attach the formula graph for the given closure state to the observation.

    Emits an index observation when the closure stores deduplicated graphs
    (training curricula), and a materialized-graph observation otherwise.
    """
    if closure.graph_indices is not None:
        assert closure.graph_table is not None
        return FormulaIndexObservation.from_obs(
            obs, closure.graph_indices[closure_state], closure.graph_table
        )
    return FormulaGraphObservation.from_obs(obs, closure.get_graph(closure_state))


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
            | Environment[Any, Any, TEnvParams, TObsFeatures, CurriculumResetOptions]
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
        formula_obs = _formula_obs(obs, state.closure, state.closure_state)
        return state, formula_obs  # type: ignore

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
        next_obs = _formula_obs(
            transition.observation, state.closure, next_closure_state
        )
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
            terminal_observation=_formula_obs(
                transition.terminal_observation, state.closure, next_closure_state
            ),
            propositions=transition.propositions,
            info=transition.info | {"is_sink": is_false},
        )
