"""Utilities for preprocessing LTL formulas into JaxLDBAs and JaxClauseReachAvoidSequences."""

from typing import cast

import jax
import jax.numpy as jnp

from jaxolotl.deep_ltl.eval.preprocessing import _batch_ldbas, _build_ldba
from jaxolotl.deep_ltl.reach_avoid import path_search
from jaxolotl.deep_ltl.reach_avoid.jax_reach_avoid_sequence import JaxReachAvoidSequence
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA
from jaxolotl.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.struct_ltl.reach_avoid.jax_clause_graph_reach_avoid_sequence import (
    EdgeData,
    JaxGraphReachAvoidSequence,
    NodeData,
)
from jaxolotl.struct_ltl.reach_avoid.jax_clause_reach_avoid_sequence import (
    JaxClauseReachAvoidSequence,
)
from jaxolotl.struct_ltl.reach_avoid.jax_tokenized_reach_avoid_sequence import (
    JaxTokenizedReachAvoidSequence,
)

_LENGTH_AXIS = 2
_CLAUSE_AXIS = 3


def preprocess_formulas(
    formulas: list[str],
    env: Environment | EnvWrapper,
    tokenize: bool = False,
    graph: bool = False,
) -> tuple[JaxLDBA, JaxReachAvoidSequence]:
    """Converts a list of formulas into a batched JaxLDBA and batched JaxClauseReachAvoidSequence,
    with a set of sequences for every LDBA state."""
    if tokenize and graph:
        raise ValueError("Only one of `tokenize` or `graph` can be enabled.")

    ldbas, seqs = [], []
    for formula in formulas:
        ldba, batched_seqs = _preprocess_formula(
            formula,
            env,
            tokenize=tokenize,
            graph=graph,
        )
        ldbas.append(ldba)
        seqs.append(batched_seqs)
    ldba = _batch_ldbas(ldbas)
    if graph:
        batched_seqs = _batch_graph_sequences(seqs)  # type: ignore[arg-type]
    elif tokenize:
        batched_seqs = _batch_tokenized_sequences(seqs)  # type: ignore[arg-type]
    else:
        batched_seqs = _batch_sequences(seqs)  # type: ignore[arg-type]
    return ldba, batched_seqs


def _preprocess_formula(
    formula: str,
    env: Environment | EnvWrapper,
    tokenize: bool = False,
    graph: bool = False,
) -> tuple[JaxLDBA, JaxReachAvoidSequence]:
    """Preprocesses the formula into a JaxLDBA and batched JaxReachAvoidSequence."""

    ldba = _build_ldba(formula, env)
    jldba = JaxLDBA.from_ldba(ldba, env)
    state_to_seqs = path_search.compute_sequences(ldba, num_loops=2)
    state_to_boolean_seqs = {
        state: [
            expanded_seq
            for seq in seq_list
            for expanded_seq in BooleanReachAvoidSequence.from_reach_avoid_sequence(
                seq, env
            ).expand_clauses()
        ]
        for state, seq_list in state_to_seqs.items()
    }
    if graph:
        batched_seqs = JaxGraphReachAvoidSequence.from_state_to_seqs(
            state_to_boolean_seqs,
            env.propositions,
            env.assignments(),
            env.max_nodes,
            env.max_edges,
        )
    else:
        clz = (
            JaxTokenizedReachAvoidSequence if tokenize else JaxClauseReachAvoidSequence
        )
        batched_seqs = clz.from_state_to_seqs(state_to_boolean_seqs, env)
    return jldba, batched_seqs


def _batch_sequences(
    seqs: list[JaxClauseReachAvoidSequence],
) -> JaxClauseReachAvoidSequence:
    """Batch multiple JaxClauseReachAvoidSequence into a single JaxClauseReachAvoidSequence with an added batch dimension.

    Args:
        seqs: List of JaxClauseReachAvoidSequence to batch. Shape: (num_states, num_seqs, max_length, num_assignments)

    Returns:
        JaxClauseReachAvoidSequence: Batched sequence. Shape: (batch_size, max_num_states, max_num_seqs, max_length, num_assignments)
    """
    max_num_states = max(seq.reach.shape[0] for seq in seqs)
    max_num_seqs = max(seq.reach.shape[1] for seq in seqs)
    max_length = max(seq.reach.shape[2] for seq in seqs)
    max_clauses = max(seq.avoid_clauses.shape[-2] for seq in seqs)

    def pad_leaf(x):
        # Calculate padding
        pad_axis0 = max(0, max_num_states - x.shape[0])
        pad_axis1 = max(0, max_num_seqs - x.shape[1])

        pad_config = (
            (0, pad_axis0),  # Axis 0 (states)
            (0, pad_axis1),  # Axis 1 (seqs)
        )

        if x.ndim > _LENGTH_AXIS:
            pad_axis2 = max(0, max_length - x.shape[_LENGTH_AXIS])
            pad_config += ((0, pad_axis2),)  # Axis 2 (length)

        if x.ndim > _CLAUSE_AXIS + 1:
            pad_axis3 = max(0, max_clauses - x.shape[_CLAUSE_AXIS])
            pad_config += ((0, pad_axis3),)  # Axis 3 (clauses)

        pad_config += ((0, 0),) * (x.ndim - len(pad_config))

        return jnp.pad(
            x,
            pad_width=pad_config,
            mode="constant",
            constant_values=-1 if x.dtype == jnp.int32 else 0,
        )

    seqs = jax.tree.map(pad_leaf, seqs)
    return jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *seqs,
    )


def _batch_tokenized_sequences(
    seqs: list[JaxTokenizedReachAvoidSequence],
) -> JaxTokenizedReachAvoidSequence:
    """Batch multiple JaxTokenizedReachAvoidSequence into a single JaxTokenizedReachAvoidSequence with an added batch dimension.

    Args:
        seqs: List of JaxTokenizedReachAvoidSequence to batch. Shape: (num_states, num_seqs, max_length, max_tokens)

    Returns:
        JaxTokenizedReachAvoidSequence: Batched sequence. Shape: (batch_size, max_num_states, max_num_seqs, max_length, max_tokens)
    """
    max_num_states = max(seq.reach.shape[0] for seq in seqs)
    max_num_seqs = max(seq.reach.shape[1] for seq in seqs)
    max_length = max(seq.reach.shape[2] for seq in seqs)
    max_tokens = max(
        max(seq.reach_tokens.shape[-1], seq.avoid_tokens.shape[-1]) for seq in seqs
    )

    def pad_leaf(x):
        # Calculate padding
        pad_axis0 = max(0, max_num_states - x.shape[0])
        pad_axis1 = max(0, max_num_seqs - x.shape[1])

        pad_config = (
            (0, pad_axis0),  # Axis 0 (states)
            (0, pad_axis1),  # Axis 1 (seqs)
        )

        if x.ndim > _LENGTH_AXIS:
            pad_axis2 = max(0, max_length - x.shape[_LENGTH_AXIS])
            pad_config += ((0, pad_axis2),)  # Axis 2 (length)

        pad_config += ((0, 0),) * (x.ndim - len(pad_config))

        return jnp.pad(
            x,
            pad_width=pad_config,
            mode="constant",
            constant_values=-1 if x.dtype == jnp.int32 else 0,
        )

    seqs = jax.tree.map(pad_leaf, seqs)

    # pad tokens
    padded = []
    for seq in seqs:
        reach_tokens = seq.reach_tokens
        avoid_tokens = seq.avoid_tokens

        pad_axis3 = max(0, max_tokens - reach_tokens.shape[-1])
        pad_config = (
            (0, 0),  # Axis 0 (states)
            (0, 0),  # Axis 1 (seqs)
            (0, 0),  # Axis 2 (length)
            (0, pad_axis3),  # Axis 3 (tokens)
        )

        reach_tokens = jnp.pad(
            reach_tokens,
            pad_width=pad_config,
            mode="constant",
            constant_values=-1,
        )

        pad_axis3 = max(0, max_tokens - avoid_tokens.shape[-1])
        pad_config = (
            (0, 0),  # Axis 0 (states)
            (0, 0),  # Axis 1 (seqs)
            (0, 0),  # Axis 2 (length)
            (0, pad_axis3),  # Axis 3 (tokens)
        )
        avoid_tokens = jnp.pad(
            avoid_tokens,
            pad_width=pad_config,
            mode="constant",
            constant_values=-1,
        )

        padded.append(
            JaxTokenizedReachAvoidSequence(
                reach=seq.reach,
                avoid=seq.avoid,
                reach_tokens=reach_tokens,
                avoid_tokens=avoid_tokens,
                repeat_last=seq.repeat_last,
                last_index=seq.last_index,
            )
        )

    return jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *padded,
    )


def _batch_graph_sequences(
    seqs: list[JaxGraphReachAvoidSequence],
) -> JaxGraphReachAvoidSequence:
    """Batch graph sequences while preserving graph-padding invariants.

    A padded graph must contain one masked node and no edges: ``n_node`` must
    therefore be padded with one, not with the generic integer sentinel ``-1``.
    """
    if not seqs:
        raise ValueError("Cannot batch an empty list of graph sequences.")

    def pad_and_stack_pytrees(pytrees: list):
        max_shapes = jax.tree.map(
            lambda *xs: jnp.max(jnp.array([x.shape for x in xs]), axis=0),
            *pytrees,
        )

        def pad_leaf(leaf: jax.Array, max_shape: jax.Array) -> jax.Array:
            pad_config = tuple(
                (0, int(max_d - d))
                for d, max_d in zip(leaf.shape, max_shape, strict=True)
            )
            if leaf.dtype == jnp.bool_:
                constant_values = False
            elif jnp.issubdtype(leaf.dtype, jnp.integer):
                constant_values = -1
            else:
                constant_values = 0
            return jnp.pad(
                leaf,
                pad_width=pad_config,
                mode="constant",
                constant_values=constant_values,
            )

        padded_pytrees = [
            jax.tree.map(pad_leaf, pytree, max_shapes) for pytree in pytrees
        ]
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *padded_pytrees)

    def pad_graphs(graphs: list):
        max_states = max(graph.n_node.shape[0] for graph in graphs)
        max_seqs = max(graph.n_node.shape[1] for graph in graphs)
        max_length = max(graph.n_node.shape[2] for graph in graphs)
        max_nodes = max(
            cast(NodeData, graph.nodes)["mask"].shape[-1] for graph in graphs
        )
        max_edges = max(
            cast(EdgeData, graph.edges)["mask"].shape[-1] for graph in graphs
        )

        def pad_leaf(
            leaf: jax.Array, target_shape: tuple[int, ...], value: int | bool
        ) -> jax.Array:
            padding = tuple(
                (0, target - current)
                for current, target in zip(leaf.shape, target_shape, strict=True)
            )
            return jnp.pad(leaf, padding, mode="constant", constant_values=value)

        padded_graphs = []
        for graph in graphs:
            nodes = cast(NodeData, graph.nodes)
            edges = cast(EdgeData, graph.edges)
            padded_graphs.append(
                graph._replace(
                    nodes={
                        "type_idx": pad_leaf(
                            nodes["type_idx"], (max_states, max_seqs, max_nodes), -1
                        ),
                        "prop_idx": pad_leaf(
                            nodes["prop_idx"], (max_states, max_seqs, max_nodes), -1
                        ),
                        "mask": pad_leaf(
                            nodes["mask"], (max_states, max_seqs, max_nodes), False
                        ),
                    },
                    edges={
                        "mask": pad_leaf(
                            edges["mask"], (max_states, max_seqs, max_edges), False
                        )
                    },
                    senders=pad_leaf(
                        graph.senders, (max_states, max_seqs, max_edges), 0
                    ),
                    receivers=pad_leaf(
                        graph.receivers, (max_states, max_seqs, max_edges), 0
                    ),
                    n_node=pad_leaf(
                        graph.n_node, (max_states, max_seqs, max_length), 1
                    ),
                    n_edge=pad_leaf(
                        graph.n_edge, (max_states, max_seqs, max_length), 0
                    ),
                )
            )
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *padded_graphs)

    batched_reach_graphs = pad_graphs([seq.reach_graphs for seq in seqs])
    batched_avoid_graphs = pad_graphs([seq.avoid_graphs for seq in seqs])
    batched_reach = pad_and_stack_pytrees([seq.reach for seq in seqs])
    batched_avoid = pad_and_stack_pytrees([seq.avoid for seq in seqs])
    batched_repeat_last = pad_and_stack_pytrees([seq.repeat_last for seq in seqs])
    batched_last_index = pad_and_stack_pytrees([seq.last_index for seq in seqs])

    return JaxGraphReachAvoidSequence(
        reach=batched_reach,
        avoid=batched_avoid,
        reach_graphs=batched_reach_graphs,
        avoid_graphs=batched_avoid_graphs,
        repeat_last=batched_repeat_last,
        last_index=batched_last_index,
    )
