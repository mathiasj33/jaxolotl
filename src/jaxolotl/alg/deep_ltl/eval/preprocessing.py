"""Utilities for preprocessing LTL formulas into JaxLDBAs and JaxReachAvoidSequences."""

from jaxolotl.alg.reach_avoid import preprocessing
from jaxolotl.alg.reach_avoid.jax_sequence import (
    JaxReachAvoidSequence,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eqx_utils.batching import pad_and_stack
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
    """Pad and stack reach-avoid sequences along a formula batch axis."""
    return pad_and_stack(seqs, JaxReachAvoidSequence.padding_values())
