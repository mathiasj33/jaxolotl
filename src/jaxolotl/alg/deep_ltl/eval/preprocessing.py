"""Utilities for preprocessing LTL formulas into JaxLDBAs and JaxReachAvoidSequences."""

import jax
import jax.numpy as jnp

from jaxolotl.alg.reach_avoid import preprocessing
from jaxolotl.alg.reach_avoid.jax_sequence import (
    JaxReachAvoidSequence,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA


def preprocess_formulas(
    formulas: list[str], env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxReachAvoidSequence]:
    """Converts a list of formulas into a batched JaxLDBA and batched JaxReachAvoidSequence,
    with a set of sequences for every LDBA state."""

    return preprocessing.preprocess_formulas(
        formulas,
        env,
        JaxReachAvoidSequence.from_state_to_seqs,
        _batch_sequences,
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
