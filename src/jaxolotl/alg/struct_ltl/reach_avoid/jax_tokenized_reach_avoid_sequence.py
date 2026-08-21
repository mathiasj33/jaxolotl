"""JAX-compatible tokenized reach-avoid sequence representation.

This module provides a JAX representation of reach-avoid sequences where
Boolean formulas are represented as token sequences rather than structured
clause representations. Reach and avoid formulas are stored separately.
"""

from dataclasses import replace
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl.alg.common.reach_avoid.batching import (
    batch_assignments,
    batch_state_sequences,
)
from jaxolotl.alg.common.reach_avoid.jax_sequence import (
    JaxReachAvoidSequence,
)
from jaxolotl.alg.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.alg.struct_ltl.reach_avoid.formula_tokenizer import (
    Vocabulary,
    encode_tokens,
    tokenize_reach_avoid_step,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


class JaxTokenizedReachAvoidSequence(JaxReachAvoidSequence):
    """JAX representation of a reach-avoid sequence with tokenized formulas.

    Instead of structured clause representations, formulas are represented as
    sequences of tokens suitable for processing by sequence models like GRUs.
    """

    # Reach token sequences for each step, with -1 for padding
    # shape: (max_length, max_reach_tokens)
    reach_tokens: jax.Array

    # Avoid token sequences for each step, with -1 for padding
    # shape: (max_length, max_avoid_tokens)
    avoid_tokens: jax.Array

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    @override
    def advance(self) -> "JaxTokenizedReachAvoidSequence":
        """Advance the reach-avoid sequence by one step. Returns a new sequence."""

        is_last_step = self.depth == 1
        should_repeat = jnp.logical_and(
            is_last_step, self.last_index + 1 < self.repeat_last
        )

        def _repeat_step():
            return replace(self, last_index=self.last_index + 1)

        def _advance_step():
            # Advance assignment arrays one step
            new_reach = jnp.roll(self.reach, -1, axis=0)
            new_avoid = jnp.roll(self.avoid, -1, axis=0)

            # Pad the last row with -1s
            new_reach = new_reach.at[-1, :].set(-1)
            new_avoid = new_avoid.at[-1, :].set(-1)

            # Advance token arrays one step
            new_reach_tokens = jnp.roll(self.reach_tokens, -1, axis=0)
            new_avoid_tokens = jnp.roll(self.avoid_tokens, -1, axis=0)

            # Pad the last row with -1s
            new_reach_tokens = new_reach_tokens.at[-1, :].set(-1)
            new_avoid_tokens = new_avoid_tokens.at[-1, :].set(-1)

            return JaxTokenizedReachAvoidSequence(
                reach=new_reach,
                avoid=new_avoid,
                reach_tokens=new_reach_tokens,
                avoid_tokens=new_avoid_tokens,
                repeat_last=self.repeat_last,
                last_index=jnp.zeros_like(self.last_index),
            )

        return jax.lax.cond(
            jnp.all(should_repeat),
            _repeat_step,
            _advance_step,
        )

    @classmethod
    def from_reach_avoid_seqs(
        cls,
        seqs: list[BooleanReachAvoidSequence],
        env: Environment | EnvWrapper,
        max_reach_tokens: int | None = None,
        max_avoid_tokens: int | None = None,
        max_length: int | None = None,
    ) -> "JaxTokenizedReachAvoidSequence":
        """Convert a list of BooleanReachAvoidSequences into a batched JAX representation.

        Args:
            seqs: List of BooleanReachAvoidSequences to convert.
            env: Environment for proposition and assignment information.
            max_reach_tokens: Maximum tokens per reach formula. If None, uses the
                maximum observed across sequences.
            max_avoid_tokens: Maximum tokens per avoid formula. If None, uses the
                maximum observed across sequences.
            max_length: Maximum sequence length. If None, uses the maximum observed.

        Returns:
            Batched JaxTokenizedReachAvoidSequence.
        """
        vocab = Vocabulary.from_propositions(env.propositions)
        max_length = max_length or max(len(seq.reach_avoid) for seq in seqs)

        # First pass: compute token sequences and find max lengths
        # [seq_idx][step_idx] -> (reach_tokens, avoid_tokens)
        all_tokens: list[list[tuple[list[int], list[int]]]] = []
        observed_max_reach = 1
        observed_max_avoid = 1

        for seq in seqs:
            seq_tokens: list[tuple[list[int], list[int]]] = []
            for reach_graph, avoid_graph in seq.reach_avoid_formulas:
                reach_toks, avoid_toks = tokenize_reach_avoid_step(
                    reach_graph, avoid_graph
                )
                reach_encoded = encode_tokens(reach_toks, vocab)
                avoid_encoded = encode_tokens(avoid_toks, vocab)
                seq_tokens.append((reach_encoded, avoid_encoded))
                observed_max_reach = max(observed_max_reach, len(reach_encoded))
                observed_max_avoid = max(observed_max_avoid, len(avoid_encoded))
            all_tokens.append(seq_tokens)

        max_reach_tokens = max_reach_tokens or observed_max_reach
        max_avoid_tokens = max_avoid_tokens or observed_max_avoid

        # --- Assignments (same as JaxClauseReachAvoidSequence) ---
        assignment_arrays = batch_assignments(
            seqs, env.assignments(), max_length=max_length
        )

        # --- Token arrays ---
        reach_tokens = -np.ones(
            (len(seqs), max_length, max_reach_tokens), dtype=np.int32
        )
        avoid_tokens = -np.ones(
            (len(seqs), max_length, max_avoid_tokens), dtype=np.int32
        )

        # --- Fill arrays ---
        for seq_idx, _seq in enumerate(seqs):
            # Fill token arrays
            for step_idx, (reach_toks, avoid_toks) in enumerate(all_tokens[seq_idx]):
                if step_idx >= max_length:
                    break

                # Reach tokens
                reach_len = min(len(reach_toks), max_reach_tokens)
                for i, tok_idx in enumerate(reach_toks[:reach_len]):
                    reach_tokens[seq_idx, step_idx, i] = tok_idx

                # Avoid tokens
                avoid_len = min(len(avoid_toks), max_avoid_tokens)
                for i, tok_idx in enumerate(avoid_toks[:avoid_len]):
                    avoid_tokens[seq_idx, step_idx, i] = tok_idx

        return cls(
            reach=jnp.array(assignment_arrays.reach),
            avoid=jnp.array(assignment_arrays.avoid),
            reach_tokens=jnp.array(reach_tokens),
            avoid_tokens=jnp.array(avoid_tokens),
            repeat_last=jnp.array(assignment_arrays.repeat_last),
            last_index=jnp.zeros_like(assignment_arrays.repeat_last),
        )

    @classmethod
    def padding_values(cls) -> "JaxTokenizedReachAvoidSequence":
        """Return semantic scalar padding values for every array field."""
        return cls(
            reach=jnp.asarray(-1, dtype=jnp.int32),
            avoid=jnp.asarray(-1, dtype=jnp.int32),
            reach_tokens=jnp.asarray(-1, dtype=jnp.int32),
            avoid_tokens=jnp.asarray(-1, dtype=jnp.int32),
            repeat_last=jnp.asarray(1, dtype=jnp.int32),
            last_index=jnp.asarray(0, dtype=jnp.int32),
        )

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[BooleanReachAvoidSequence]],
        env: Environment | EnvWrapper,
    ) -> "JaxTokenizedReachAvoidSequence":
        """Encode and pack tokenized sequences by LDBA state."""
        return batch_state_sequences(
            state_to_seqs,
            lambda sequences: cls.from_reach_avoid_seqs(sequences, env),
            cls.padding_values(),
        )
