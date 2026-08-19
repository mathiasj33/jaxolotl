"""Utilities for preprocessing LTL formulas into JaxLDBAs and JaxReachAvoidSequences."""

import jax
import jax.numpy as jnp

from jaxolotl.alg.deep_ltl.reach_avoid import path_search
from jaxolotl.alg.deep_ltl.reach_avoid.jax_reach_avoid_sequence import (
    JaxReachAvoidSequence,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata import ltl2ldba
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA


def preprocess_formulas(
    formulas: list[str], env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxReachAvoidSequence]:
    """Converts a list of formulas into a batched JaxLDBA and batched JaxReachAvoidSequence,
    with a set of sequences for every LDBA state."""

    ldbas, seqs = [], []
    for formula in formulas:
        ldba, batched_seqs = _preprocess_formula(formula, env)
        ldbas.append(ldba)
        seqs.append(batched_seqs)
    ldba = _batch_ldbas(ldbas)
    batched_seqs = _batch_sequences(seqs)
    return ldba, batched_seqs


def _preprocess_formula(
    formula: str, env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxReachAvoidSequence]:
    """Preprocesses the formula into a JaxLDBA and batched JaxReachAvoidSequence."""

    ldba = _build_ldba(formula, env)
    jldba = JaxLDBA.from_ldba(ldba, env)
    state_to_seqs = path_search.compute_sequences(ldba, num_loops=2)
    batched_seqs = JaxReachAvoidSequence.from_state_to_seqs(state_to_seqs, env)
    return jldba, batched_seqs


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


def _batch_sequences(
    seqs: list[JaxReachAvoidSequence],
) -> JaxReachAvoidSequence:
    """Batch multiple JaxReachAvoidSequences into a single JaxReachAvoidSequence with an added batch dimension.

    Args:
        seqs: List of JaxReachAvoidSequence to batch. Shape: (num_states, num_seqs, max_length, num_assignments)

    Returns:
        JaxReachAvoidSequence: Batched sequence. Shape: (batch_size, max_num_states, max_num_seqs, max_length, num_assignments)
    """
    max_num_states = max(seq.reach.shape[0] for seq in seqs)
    max_num_seqs = max(seq.reach.shape[1] for seq in seqs)
    max_length = max(seq.reach.shape[2] for seq in seqs)

    def pad_leaf(x):
        # Calculate padding
        pad_axis0 = max(0, max_num_states - x.shape[0])
        pad_axis1 = max(0, max_num_seqs - x.shape[1])

        pad_config = (
            (0, pad_axis0),  # Axis 0 (states)
            (0, pad_axis1),  # Axis 1 (seqs)
        )

        if x.ndim > 2:
            pad_axis2 = max(0, max_length - x.shape[2])
            pad_config += ((0, pad_axis2),)  # Axis 2 (length)

        pad_config += ((0, 0),) * (x.ndim - len(pad_config))

        return jnp.pad(x, pad_width=pad_config, mode="constant", constant_values=-1)

    seqs = jax.tree.map(pad_leaf, seqs)
    return jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *seqs,
    )
