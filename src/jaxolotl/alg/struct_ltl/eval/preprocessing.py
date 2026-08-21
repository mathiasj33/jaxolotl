"""Utilities for preprocessing LTL formulas into JaxLDBAs and JaxClauseReachAvoidSequences."""

from jaxolotl.alg.common.reach_avoid.jax_sequence import (
    JaxReachAvoidSequence,
)
from jaxolotl.alg.common.reach_avoid.preprocessing import (
    preprocess_formulas as preprocess_reach_avoid_formulas,
)
from jaxolotl.alg.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.alg.struct_ltl.reach_avoid.jax_clause_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
)
from jaxolotl.alg.struct_ltl.reach_avoid.jax_clause_reach_avoid_sequence import (
    JaxClauseReachAvoidSequence,
)
from jaxolotl.alg.struct_ltl.reach_avoid.jax_tokenized_reach_avoid_sequence import (
    JaxTokenizedReachAvoidSequence,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eqx_utils.batching import pad_and_stack
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA
from jaxolotl.ltl.reach_avoid.sequence import ReachAvoidSequence


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

    if graph:
        encoder = JaxGraphReachAvoidSequence.from_state_to_seqs
        batcher = _batch_graph_sequences
    elif tokenize:
        encoder = JaxTokenizedReachAvoidSequence.from_state_to_seqs
        batcher = _batch_tokenized_sequences
    else:
        encoder = JaxClauseReachAvoidSequence.from_state_to_seqs
        batcher = _batch_sequences
    return preprocess_reach_avoid_formulas(
        formulas,
        env,
        encoder,  # type: ignore[arg-type]
        batcher,  # type: ignore[arg-type]
        transform_sequences=_to_boolean_sequences,
    )


def _to_boolean_sequences(
    state_to_sequences: dict[int, list[ReachAvoidSequence]],
    env: Environment | EnvWrapper,
) -> dict[int, list[BooleanReachAvoidSequence]]:
    """Synthesize and expand Boolean formulas for reach-avoid sequences."""
    return {
        state: [
            expanded_seq
            for seq in seq_list
            for expanded_seq in BooleanReachAvoidSequence.from_reach_avoid_sequence(
                seq, env
            ).expand_clauses()
        ]
        for state, seq_list in state_to_sequences.items()
    }


def _batch_sequences(
    sequences: list[JaxClauseReachAvoidSequence],
) -> JaxClauseReachAvoidSequence:
    """Pad and stack clause sequences along a formula batch axis."""
    return pad_and_stack(sequences, JaxClauseReachAvoidSequence.padding_values())


def _batch_tokenized_sequences(
    sequences: list[JaxTokenizedReachAvoidSequence],
) -> JaxTokenizedReachAvoidSequence:
    """Pad and stack tokenized sequences along a formula batch axis."""
    return pad_and_stack(sequences, JaxTokenizedReachAvoidSequence.padding_values())


def _batch_graph_sequences(
    sequences: list[JaxGraphReachAvoidSequence],
) -> JaxGraphReachAvoidSequence:
    """Pad and stack graph sequences while preserving graph invariants."""
    return pad_and_stack(sequences, JaxGraphReachAvoidSequence.padding_values())
