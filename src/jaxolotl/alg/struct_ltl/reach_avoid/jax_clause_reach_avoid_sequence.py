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
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.reach_avoid.sequence import EpsilonType


class JaxClauseReachAvoidSequence(JaxReachAvoidSequence):
    """Jax representation of a reach-avoid sequence with assignments and clauses."""

    # reach: only single clause, whereas avoid can be multiple clauses
    reach_clauses: jax.Array  # shape: (max_length, num_propositions)
    reach_negatives: jax.Array  # shape: (max_length, num_propositions), bool
    avoid_clauses: jax.Array  # shape: (max_length, max_clauses, num_propositions)
    avoid_negatives: jax.Array  # shape: (max_length, max_clauses, num_propositions)
    num_avoid_clauses: jax.Array  # shape: (max_length,)
    # number of avoid clauses at each step, needed for proper padding

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    @override
    def advance(self) -> "JaxClauseReachAvoidSequence":
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

            # Advance clause arrays one step
            new_reach_clauses = jnp.roll(self.reach_clauses, -1, axis=0)
            new_reach_negatives = jnp.roll(self.reach_negatives, -1, axis=0)
            new_avoid_clauses = jnp.roll(self.avoid_clauses, -1, axis=0)
            new_avoid_negatives = jnp.roll(self.avoid_negatives, -1, axis=0)

            # Pad the last row with -1s and Falses
            new_reach_clauses = new_reach_clauses.at[-1, :].set(-1)
            new_reach_negatives = new_reach_negatives.at[-1, :].set(False)
            new_avoid_clauses = new_avoid_clauses.at[-1, :, :].set(-1)
            new_avoid_negatives = new_avoid_negatives.at[-1, :, :].set(False)

            # Advance num_avoid_clauses
            new_num_avoid_clauses = jnp.roll(self.num_avoid_clauses, -1, axis=0)
            new_num_avoid_clauses = new_num_avoid_clauses.at[-1].set(0)

            return JaxClauseReachAvoidSequence(
                reach=new_reach,
                avoid=new_avoid,
                reach_clauses=new_reach_clauses,
                reach_negatives=new_reach_negatives,
                avoid_clauses=new_avoid_clauses,
                avoid_negatives=new_avoid_negatives,
                num_avoid_clauses=new_num_avoid_clauses,
                repeat_last=self.repeat_last,
                last_index=jnp.zeros_like(self.last_index),
            )

        return jax.lax.cond(
            jnp.all(should_repeat),
            _repeat_step,
            _advance_step,
        )

    @classmethod
    def from_reach_avoid_seqs(  # noqa: PLR0912
        cls,
        seqs: list[BooleanReachAvoidSequence],
        env: Environment | EnvWrapper,
        max_clauses: int | None = None,
        max_length: int | None = None,
    ) -> "JaxClauseReachAvoidSequence":
        """
        Converts a list of BooleanReachAvoidSequences into a batched Jax representation.

        Args:
            seqs: list of BooleanReachAvoidSequences to convert.
            propositions: list of proposition names in the environment.
            assignments: list of assignments in the environment.
            max_clauses: maximum number of avoid clauses to pad to. If None, uses the
                maximum number of avoid clauses in the sequences.
            max_length: maximum length of sequences to pad to. If None, uses the
                maximum length of the sequences.
        """
        max_length = max_length or max(len(seq.reach_avoid) for seq in seqs)
        max_clauses = max_clauses or max(
            len(avoid) for seq in seqs for _, avoid in seq.clauses
        )

        # --- Assignments ---
        assignment_arrays = batch_assignments(
            seqs, env.assignments(), max_length=max_length
        )

        # --- Clauses ---
        propositions = env.propositions
        prop_map = {name: i for i, name in enumerate(propositions)}
        reach_clauses = -np.ones(
            (len(seqs), max_length, len(propositions)), dtype=np.int32
        )
        reach_negatives = np.zeros_like(reach_clauses, dtype=bool)
        avoid_clauses = -np.ones(
            (len(seqs), max_length, max_clauses, len(propositions)),
            dtype=np.int32,
        )
        avoid_negatives = np.zeros_like(avoid_clauses, dtype=bool)
        num_avoid_clauses = np.zeros((len(seqs), max_length), dtype=np.int32)

        # --- Fill arrays ---
        for seq_idx, seq in enumerate(seqs):
            for i, (reach, avoid) in enumerate(seq.clauses):
                # Reach clauses
                if isinstance(reach, EpsilonType):
                    reach_clauses[seq_idx, i, 0] = len(env.assignments())
                else:
                    if len(reach) != 1:
                        raise ValueError(
                            "Reach clauses must contain exactly one clause. "
                            f"Got {len(reach)} clauses."
                        )
                    clause = reach[0]
                    for j, atom in enumerate(list(clause.neg) + list(clause.pos)):
                        reach_clauses[seq_idx, i, j] = prop_map[atom]
                    reach_negatives[seq_idx, i, : len(clause.neg)] = True

                # Avoid clauses
                for c_idx, clause in enumerate(avoid):
                    for j, atom in enumerate(list(clause.neg) + list(clause.pos)):
                        avoid_clauses[seq_idx, i, c_idx, j] = prop_map[atom]
                    avoid_negatives[seq_idx, i, c_idx, : len(clause.neg)] = True
                num_avoid_clauses[seq_idx, i] = len(avoid)

        return cls(
            reach=jnp.array(assignment_arrays.reach),
            avoid=jnp.array(assignment_arrays.avoid),
            reach_clauses=jnp.array(reach_clauses),
            reach_negatives=jnp.array(reach_negatives),
            avoid_clauses=jnp.array(avoid_clauses),
            avoid_negatives=jnp.array(avoid_negatives),
            num_avoid_clauses=jnp.array(num_avoid_clauses),
            repeat_last=jnp.array(assignment_arrays.repeat_last),
            last_index=jnp.zeros_like(assignment_arrays.repeat_last),
        )

    @classmethod
    def padding_values(cls) -> "JaxClauseReachAvoidSequence":
        """Return semantic scalar padding values for every array field."""
        return cls(
            reach=jnp.asarray(-1, dtype=jnp.int32),
            avoid=jnp.asarray(-1, dtype=jnp.int32),
            reach_clauses=jnp.asarray(-1, dtype=jnp.int32),
            reach_negatives=jnp.asarray(False),
            avoid_clauses=jnp.asarray(-1, dtype=jnp.int32),
            avoid_negatives=jnp.asarray(False),
            num_avoid_clauses=jnp.asarray(0, dtype=jnp.int32),
            repeat_last=jnp.asarray(1, dtype=jnp.int32),
            last_index=jnp.asarray(0, dtype=jnp.int32),
        )

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[BooleanReachAvoidSequence]],
        env: Environment | EnvWrapper,
    ) -> "JaxClauseReachAvoidSequence":
        """Encode and pack Boolean sequences by LDBA state."""
        return batch_state_sequences(
            state_to_seqs,
            lambda sequences: cls.from_reach_avoid_seqs(sequences, env),
            cls.padding_values(),
        )
