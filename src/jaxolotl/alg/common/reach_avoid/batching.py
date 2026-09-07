"""Utility functions for batching reach-avoid sequences into padded arrays.

Since both the training and evaluation pipelines are compiled end-to-end, we preprocess
reach-avoid sequences into padded arrays beforehand."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.reach_avoid.sequence import EpsilonType, ReachAvoidSequence


@dataclass(frozen=True)
class AssignmentArrays:
    """Padded NumPy representation shared by reach-avoid encoders."""

    reach: np.ndarray  # shape: (num_sequences, max_length, max_assignments)
    avoid: np.ndarray  # shape: (num_sequences, max_length, max_assignments)
    repeat_last: np.ndarray  # shape: (num_sequences,)


def batch_assignments(
    sequences: Sequence[ReachAvoidSequence],
    assignments: Sequence[Assignment],
    *,
    max_length: int | None = None,
) -> AssignmentArrays:
    """Batch assignment sets for a flat batch of reach-avoid sequences."""
    if not sequences:
        raise ValueError("Cannot batch an empty sequence batch.")

    required_length = max(len(sequence.reach_avoid) for sequence in sequences)
    if max_length is None:
        max_length = required_length
    elif max_length < required_length:
        raise ValueError(
            "max_length is smaller than the longest supplied sequence: "
            f"max_length={max_length}, required_length={required_length}"
        )

    assignment_to_index = {
        assignment: index for index, assignment in enumerate(assignments)
    }
    max_assignments = _resolve_max_assignments(sequences)
    if max_assignments == 0:
        raise ValueError(
            "Sequences must contain at least one assignment in reach or avoid sets."
        )
    epsilon_index = len(assignments)
    reach = -np.ones((len(sequences), max_length, max_assignments), dtype=np.int32)
    avoid = -np.ones_like(reach)
    repeat_last = np.ones((len(sequences),), dtype=np.int32)

    for sequence_index, sequence in enumerate(sequences):
        repeat_last[sequence_index] = sequence.repeat_last
        for step_index, (step_reach, step_avoid) in enumerate(sequence.reach_avoid):
            if isinstance(step_reach, EpsilonType):
                reach[sequence_index, step_index, 0] = epsilon_index
            else:
                indices = [assignment_to_index[item] for item in step_reach]
                reach[sequence_index, step_index, : len(indices)] = indices
            indices = [assignment_to_index[item] for item in step_avoid]
            avoid[sequence_index, step_index, : len(indices)] = indices

    return AssignmentArrays(reach=reach, avoid=avoid, repeat_last=repeat_last)


def _resolve_max_assignments(sequences: Sequence[ReachAvoidSequence]) -> int:
    """Resolves the maximum number of assignments in any reach or avoid set in the given sequences."""
    max_assignments = 0
    for sequence in sequences:
        for step_reach, step_avoid in sequence.reach_avoid:
            if not isinstance(step_reach, EpsilonType):
                max_assignments = max(max_assignments, len(step_reach))
            max_assignments = max(max_assignments, len(step_avoid))
    return max_assignments


def batch_state_sequences[TSequence, TEncoded](
    state_to_sequences: Mapping[int, Sequence[TSequence]],
    encode: Callable[[list[TSequence]], TEncoded],
    padding_values: TEncoded,
    *,
    num_states: int | None = None,
) -> TEncoded:
    """Batch sequences once, then scatter them into state and sequence axes.

    Args:
        state_to_sequences: Mapping from LDBA states to sequences to encode.
        encode: Function to encode a flat list of sequences into a PyTree of arrays.
        padding_values: PyTree of scalar values to use for padding missing states/sequences.
        num_states: Optional number of states to encode. If not provided, will be inferred from
            the maximum state index in `state_to_sequences`.

    Returns:
        TEncoded: PyTree of arrays with shape (num_states, max_num_sequences, ...).
    """
    if any(state < 0 for state in state_to_sequences):
        raise ValueError("State indices must be non-negative.")

    required_states = max(state_to_sequences, default=-1) + 1
    if num_states is None:
        num_states = required_states
    elif num_states < required_states:
        raise ValueError(
            f"num_states={num_states} cannot contain state index {required_states - 1}."
        )

    max_sequences = max(
        (len(sequences) for sequences in state_to_sequences.values()), default=0
    )
    flat_sequences: list[TSequence] = []
    state_indices: list[int] = []
    sequence_indices: list[int] = []
    for state, sequences in state_to_sequences.items():
        for sequence_index, sequence in enumerate(sequences):
            flat_sequences.append(sequence)
            state_indices.append(state)
            sequence_indices.append(sequence_index)

    if not flat_sequences:
        raise ValueError("Cannot encode state mapping without any sequences.")

    encoded = encode(flat_sequences)
    state_index_array = jnp.asarray(state_indices, dtype=jnp.int32)
    sequence_index_array = jnp.asarray(sequence_indices, dtype=jnp.int32)

    def scatter_leaf(leaf: Any, padding_value: Any):
        leaf = jnp.asarray(leaf)
        if leaf.shape[0] != len(flat_sequences):
            raise ValueError(
                "Every encoded leaf must have the flat sequence batch as its "
                f"leading axis; got shape {leaf.shape}."
            )
        packed = jnp.full(
            (num_states, max_sequences, *leaf.shape[1:]),
            padding_value,
            dtype=leaf.dtype,
        )
        return packed.at[state_index_array, sequence_index_array].set(leaf)

    return jax.tree.map(scatter_leaf, encoded, padding_values)
