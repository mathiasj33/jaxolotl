"""Utilities for preprocessing LTL formulas into JaxLDBAs and JaxReachAvoidSubgoals."""

import jax
import jax.numpy as jnp

from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata import ltl2ldba
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA
from jaxolotl.ltl.reach_avoid import path_search


def preprocess_formulas(
    formulas: list[str], env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxReachAvoidSubgoal]:
    """Converts a list of formulas into a batched JaxLDBA and batched JaxReachAvoidSubgoal,
    with a set of subgoals for every LDBA state."""

    ldbas, subgoals = [], []
    for formula in formulas:
        ldba, batched_subgoals = _preprocess_formula(formula, env)
        ldbas.append(ldba)
        subgoals.append(batched_subgoals)
    ldba = _batch_ldbas(ldbas)
    batched_subgoals = _batch_subgoals(subgoals)
    return ldba, batched_subgoals


def _preprocess_formula(
    formula: str, env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxReachAvoidSubgoal]:
    """Preprocesses the formula into a JaxLDBA and batched JaxReachAvoidSubgoal."""

    ldba = _build_ldba(formula, env)
    jldba = JaxLDBA.from_ldba(ldba, env)
    state_to_seqs = path_search.compute_sequences(ldba, num_loops=2)
    batched_subgoals = JaxReachAvoidSubgoal.from_state_to_seqs(state_to_seqs, env)
    return jldba, batched_subgoals


def _build_ldba(formula: str, env: Environment | EnvWrapper):
    ldba = ltl2ldba(formula, env.propositions)
    ldba.prune(env.assignments())
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


def _batch_ldbas(ldbas: list[JaxLDBA]) -> JaxLDBA:
    """Batch multiple JaxLDBAs into a single JaxLDBA with an added batch dimension."""

    num_states = jnp.array([ldba.num_states for ldba in ldbas], dtype=jnp.int32)
    max_num_states = jnp.max(num_states)
    batch_size = len(ldbas)
    num_assignments = ldbas[0].transitions.shape[1] - 1

    transitions = -jnp.ones(
        (batch_size, max_num_states, num_assignments + 1), dtype=jnp.int32
    )
    accepting = jnp.zeros((batch_size, max_num_states, num_assignments), dtype=bool)
    sink_states = jnp.zeros((batch_size, max_num_states), dtype=bool)
    initial_states = jnp.zeros((batch_size,), dtype=jnp.int32)

    for i, ldba in enumerate(ldbas):
        transitions = transitions.at[i, : ldba.num_states, :].set(ldba.transitions)
        accepting = accepting.at[i, : ldba.num_states, :].set(ldba.accepting)
        sink_states = sink_states.at[i, : ldba.num_states].set(ldba.sink_states)
        initial_states = initial_states.at[i].set(ldba.initial_state)

    return JaxLDBA(
        num_states=num_states,
        initial_state=initial_states,
        transitions=transitions,
        accepting=accepting,
        sink_states=sink_states,
        finite=jnp.array([ldba.finite for ldba in ldbas]),
    )


def _batch_subgoals(
    subgoals: list[JaxReachAvoidSubgoal],
) -> JaxReachAvoidSubgoal:
    """Batch multiple JaxReachAvoidSubgoals into a single JaxReachAvoidSubgoal with an added batch dimension.

    Args:
        subgoals: List of JaxReachAvoidSubgoal to batch. Shape: (num_states, num_subgoals, num_assignments)

    Returns:
        JaxReachAvoidSubgoal: Batched subgoals. Shape: (batch_size, max_num_states, max_num_subgoals, num_assignments) for avoid
    """
    max_num_states = max(goal.reach.shape[0] for goal in subgoals)
    max_num_subgoals = max(goal.reach.shape[1] for goal in subgoals)

    def pad_leaf(x):
        # Calculate padding
        pad_axis0 = max(0, max_num_states - x.shape[0])
        pad_axis1 = max(0, max_num_subgoals - x.shape[1])

        pad_config = (
            (0, pad_axis0),  # Axis 0 (states)
            (0, pad_axis1),  # Axis 1 (subgoals)
        )
        pad_config += ((0, 0),) * (x.ndim - len(pad_config))
        return jnp.pad(x, pad_width=pad_config, mode="constant", constant_values=-1)

    subgoals = jax.tree.map(pad_leaf, subgoals)
    return jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *subgoals,
    )
