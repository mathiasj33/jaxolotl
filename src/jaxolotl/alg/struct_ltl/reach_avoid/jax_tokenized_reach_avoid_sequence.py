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

from jaxolotl.alg.deep_ltl.reach_avoid.jax_reach_avoid_sequence import (
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
from jaxolotl.ltl.reach_avoid.sequence import EpsilonType


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
        assignments = env.assignments()
        assignment_map = {name: i for i, name in enumerate(assignments)}
        epsilon_idx = len(assignments)
        reach_assign = -np.ones(
            (len(seqs), max_length, len(assignments)), dtype=np.int32
        )
        avoid_assign = -np.ones_like(reach_assign)

        # --- Token arrays ---
        reach_tokens = -np.ones(
            (len(seqs), max_length, max_reach_tokens), dtype=np.int32
        )
        avoid_tokens = -np.ones(
            (len(seqs), max_length, max_avoid_tokens), dtype=np.int32
        )

        # --- Other ---
        repeat_last = np.ones((len(seqs),), dtype=np.int32)

        # --- Fill arrays ---
        for seq_idx, seq in enumerate(seqs):
            repeat_last[seq_idx] = seq.repeat_last

            # Fill assignment arrays
            for ra_idx, (r, a) in enumerate(seq.reach_avoid):
                if isinstance(r, EpsilonType):
                    reach_assign[seq_idx, ra_idx, 0] = epsilon_idx
                else:
                    for j, assign in enumerate(r):
                        reach_assign[seq_idx, ra_idx, j] = assignment_map[assign]
                for j, assign in enumerate(a):
                    avoid_assign[seq_idx, ra_idx, j] = assignment_map[assign]

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
            reach=jnp.array(reach_assign),
            avoid=jnp.array(avoid_assign),
            reach_tokens=jnp.array(reach_tokens),
            avoid_tokens=jnp.array(avoid_tokens),
            repeat_last=jnp.array(repeat_last),
            last_index=jnp.zeros_like(repeat_last),
        )

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[BooleanReachAvoidSequence]],
        env: Environment | EnvWrapper,
    ) -> "JaxTokenizedReachAvoidSequence":
        """Convert a mapping from LDBA states to lists of BooleanReachAvoidSequences.

        Returns:
            JaxTokenizedReachAvoidSequence with shape:
                reach_tokens: (num_states, max_num_seqs, max_length, max_reach_tokens)
                etc.
        """
        vocab = Vocabulary.from_propositions(env.propositions)

        max_seqs = max(len(seqs) for seqs in state_to_seqs.values())
        max_length = max(
            len(seq.reach_avoid) for seqs in state_to_seqs.values() for seq in seqs
        )
        num_states = len(state_to_seqs)

        # First pass: compute all tokens and find max
        all_tokens, max_reach_tokens, max_avoid_tokens = cls._compute_all_tokens(
            state_to_seqs, vocab
        )

        # Initialize arrays
        arrays = cls._init_state_arrays(
            num_states, max_seqs, max_length, max_reach_tokens, max_avoid_tokens, env
        )

        # Fill arrays
        cls._fill_state_arrays(
            state_to_seqs,
            all_tokens,
            env,
            max_length,
            max_reach_tokens,
            max_avoid_tokens,
            arrays,
        )

        return cls(
            reach=jnp.array(arrays["reach"]),
            avoid=jnp.array(arrays["avoid"]),
            reach_tokens=jnp.array(arrays["reach_tokens"]),
            avoid_tokens=jnp.array(arrays["avoid_tokens"]),
            repeat_last=jnp.array(arrays["repeat_last"]),
            last_index=jnp.zeros_like(arrays["repeat_last"]),
        )

    @staticmethod
    def _compute_all_tokens(
        state_to_seqs: dict[int, list[BooleanReachAvoidSequence]],
        vocab: Vocabulary,
    ) -> tuple[dict[int, list[list[tuple[list[int], list[int]]]]], int, int]:
        """Compute token sequences for all states and sequences."""
        all_tokens: dict[int, list[list[tuple[list[int], list[int]]]]] = {}
        max_reach_tokens = 1
        max_avoid_tokens = 1

        for state, seqs in state_to_seqs.items():
            all_tokens[state] = []
            for seq in seqs:
                seq_tokens: list[tuple[list[int], list[int]]] = []
                for reach_graph, avoid_graph in seq.reach_avoid_formulas:
                    reach_toks, avoid_toks = tokenize_reach_avoid_step(
                        reach_graph, avoid_graph
                    )
                    reach_encoded = encode_tokens(reach_toks, vocab)
                    avoid_encoded = encode_tokens(avoid_toks, vocab)
                    seq_tokens.append((reach_encoded, avoid_encoded))
                    max_reach_tokens = max(max_reach_tokens, len(reach_encoded))
                    max_avoid_tokens = max(max_avoid_tokens, len(avoid_encoded))
                all_tokens[state].append(seq_tokens)

        return all_tokens, max_reach_tokens, max_avoid_tokens

    @staticmethod
    def _init_state_arrays(
        num_states: int,
        max_seqs: int,
        max_length: int,
        max_reach_tokens: int,
        max_avoid_tokens: int,
        env: Environment | EnvWrapper,
    ) -> dict:
        """Initialize numpy arrays for state-to-seqs conversion."""
        num_assignments = len(env.assignments())
        return {
            "repeat_last": np.ones((num_states, max_seqs), dtype=np.int32),
            "reach": -np.ones(
                (num_states, max_seqs, max_length, num_assignments), dtype=np.int32
            ),
            "avoid": -np.ones(
                (num_states, max_seqs, max_length, num_assignments), dtype=np.int32
            ),
            "reach_tokens": -np.ones(
                (num_states, max_seqs, max_length, max_reach_tokens), dtype=np.int32
            ),
            "avoid_tokens": -np.ones(
                (num_states, max_seqs, max_length, max_avoid_tokens), dtype=np.int32
            ),
        }

    @staticmethod
    def _fill_state_arrays(
        state_to_seqs: dict[int, list[BooleanReachAvoidSequence]],
        all_tokens: dict[int, list[list[tuple[list[int], list[int]]]]],
        env: Environment | EnvWrapper,
        max_length: int,
        max_reach_tokens: int,
        max_avoid_tokens: int,
        arrays: dict,
    ) -> None:
        """Fill the arrays with data from sequences."""
        for state, seqs in state_to_seqs.items():
            for seq_idx, seq in enumerate(seqs):
                arrays["repeat_last"][state, seq_idx] = seq.repeat_last
                _fill_assignments(seq, state, seq_idx, env, arrays)
                _fill_tokens(
                    all_tokens[state][seq_idx],
                    state,
                    seq_idx,
                    max_length,
                    max_reach_tokens,
                    max_avoid_tokens,
                    arrays,
                )


def _fill_assignments(
    seq: BooleanReachAvoidSequence,
    state: int,
    seq_idx: int,
    env: Environment | EnvWrapper,
    arrays: dict,
) -> None:
    """Fill assignment arrays for a single sequence."""
    for i, (r, a) in enumerate(seq.reach_avoid):
        if isinstance(r, EpsilonType):
            arrays["reach"][state, seq_idx, i, 0] = len(env.assignments())
        else:
            for j, assignment in enumerate(r):
                arrays["reach"][state, seq_idx, i, j] = env.assignments().index(
                    assignment
                )
        for j, assignment in enumerate(a):
            arrays["avoid"][state, seq_idx, i, j] = env.assignments().index(assignment)


def _fill_tokens(
    seq_step_tokens: list[tuple[list[int], list[int]]],
    state: int,
    seq_idx: int,
    max_length: int,
    max_reach_tokens: int,
    max_avoid_tokens: int,
    arrays: dict,
) -> None:
    """Fill token arrays for a single sequence."""
    for step_idx, (reach_toks, avoid_toks) in enumerate(seq_step_tokens):
        if step_idx >= max_length:
            break

        # Reach tokens
        reach_len = min(len(reach_toks), max_reach_tokens)
        for i, tok_idx in enumerate(reach_toks[:reach_len]):
            arrays["reach_tokens"][state, seq_idx, step_idx, i] = tok_idx

        # Avoid tokens
        avoid_len = min(len(avoid_toks), max_avoid_tokens)
        for i, tok_idx in enumerate(avoid_toks[:avoid_len]):
            arrays["avoid_tokens"][state, seq_idx, step_idx, i] = tok_idx
