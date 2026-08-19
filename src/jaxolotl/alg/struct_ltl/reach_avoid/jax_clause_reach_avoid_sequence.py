from dataclasses import replace
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl.alg.reach_avoid.jax_sequence import (
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
        Converts a list of GraphReachAvoidSequences into a batched Jax representation.

        Args:
            seqs: list of ReachAvoidSequences to convert.
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
        assignments = env.assignments()
        assignment_map = {name: i for i, name in enumerate(assignments)}
        epsilon_idx = len(assignments)
        reach_assign = -np.ones(
            (len(seqs), max_length, len(assignments)), dtype=np.int32
        )
        avoid_assign = -np.ones_like(reach_assign)

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

        # --- Other ---
        repeat_last = np.ones((len(seqs),), dtype=np.int32)

        # --- Fill arrays ---
        for seq_idx, seq in enumerate(seqs):
            repeat_last[seq_idx] = seq.repeat_last
            for ra_idx, (r, a) in enumerate(seq.reach_avoid):
                if isinstance(r, EpsilonType):
                    reach_assign[seq_idx, ra_idx, 0] = epsilon_idx
                else:
                    for j, assign in enumerate(r):
                        reach_assign[seq_idx, ra_idx, j] = assignment_map[assign]
                for j, assign in enumerate(a):
                    avoid_assign[seq_idx, ra_idx, j] = assignment_map[assign]

            for i, (reach, avoid) in enumerate(seq.clauses):
                # Reach clauses
                if isinstance(reach, EpsilonType):
                    reach_clauses[seq_idx, i, 0] = epsilon_idx
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
            reach=jnp.array(reach_assign),
            avoid=jnp.array(avoid_assign),
            reach_clauses=jnp.array(reach_clauses),
            reach_negatives=jnp.array(reach_negatives),
            avoid_clauses=jnp.array(avoid_clauses),
            avoid_negatives=jnp.array(avoid_negatives),
            num_avoid_clauses=jnp.array(num_avoid_clauses),
            repeat_last=jnp.array(repeat_last),
            last_index=jnp.zeros_like(repeat_last),
        )

    @classmethod
    def from_state_to_seqs(  # noqa: PLR0912
        cls,
        state_to_seqs: dict[int, list[BooleanReachAvoidSequence]],
        env: Environment | EnvWrapper,
    ) -> "JaxClauseReachAvoidSequence":
        """Converts a mapping from LDBA states to lists of ReachAvoidSequences into a
        batched Jax reach-avoid sequence.

        Returns:
            JaxReachAvoidSequence: with shape
                reach: (num_states, max_num_seqs, max_length, num_assignments)
                avoid: (num_states, max_num_seqs, max_length, num_assignments)
        """

        max_seqs = max(len(seqs) for seqs in state_to_seqs.values())
        max_length = max(
            len(seq.reach_avoid) for seqs in state_to_seqs.values() for seq in seqs
        )
        num_states = len(state_to_seqs)
        # Use numpy arrays and then convert to jax arrays for efficiency
        repeat_last = np.ones((num_states, max_seqs), dtype=np.int32)
        reach_np = -np.ones(
            (num_states, max_seqs, max_length, len(env.assignments())),
            dtype=np.int32,
        )
        avoid_np = -np.ones_like(reach_np)
        for state, seqs in state_to_seqs.items():
            for seq_idx, seq in enumerate(seqs):
                repeat_last[state, seq_idx] = seq.repeat_last
                for i, (r, a) in enumerate(seq.reach_avoid):
                    if isinstance(r, EpsilonType):
                        reach_np[state, seq_idx, i, 0] = len(env.assignments())
                    else:
                        for j, assignment in enumerate(r):
                            reach_np[state, seq_idx, i, j] = env.assignments().index(
                                assignment
                            )
                    for j, assignment in enumerate(a):
                        avoid_np[state, seq_idx, i, j] = env.assignments().index(
                            assignment
                        )

        prop_map = {name: i for i, name in enumerate(env.propositions)}
        reach_clauses = -np.ones(
            (num_states, max_seqs, max_length, len(env.propositions)), dtype=np.int32
        )
        reach_negatives = np.zeros_like(reach_clauses, dtype=bool)
        max_clauses = max(
            len(avoid)
            for seqs in state_to_seqs.values()
            for seq in seqs
            for _, avoid in seq.clauses
            if avoid is not None
        )
        max_clauses = max(1, max_clauses)  # at least 1
        avoid_clauses = -np.ones(
            (num_states, max_seqs, max_length, max_clauses, len(env.propositions)),
            dtype=np.int32,
        )
        avoid_negatives = np.zeros_like(avoid_clauses, dtype=bool)
        num_avoid_clauses = np.zeros(
            (
                num_states,
                max_seqs,
                max_length,
            ),
            dtype=np.int32,
        )
        for state, seqs in state_to_seqs.items():
            for seq_idx, seq in enumerate(seqs):
                for i, (reach, avoid) in enumerate(seq.clauses):
                    # Reach clauses
                    if reach is not None:
                        if isinstance(reach, EpsilonType):
                            reach_clauses[state, seq_idx, i, 0] = len(env.assignments())
                            continue
                        if len(reach) != 1:
                            raise ValueError(
                                "Reach clauses must contain exactly one clause. "
                                f"Got {len(reach)} clauses."
                            )
                        clause = reach[0]
                        for j, atom in enumerate(list(clause.neg) + list(clause.pos)):
                            reach_clauses[state, seq_idx, i, j] = prop_map[atom]
                        reach_negatives[state, seq_idx, i, : len(clause.neg)] = True

                    # Avoid clauses
                    if avoid is not None:
                        for c_idx, clause in enumerate(avoid):
                            for j, atom in enumerate(
                                list(clause.neg) + list(clause.pos)
                            ):
                                avoid_clauses[state, seq_idx, i, c_idx, j] = prop_map[
                                    atom
                                ]
                            avoid_negatives[
                                state, seq_idx, i, c_idx, : len(clause.neg)
                            ] = True
                        num_avoid_clauses[state, seq_idx, i] = len(avoid)

        return cls(
            reach=jnp.array(reach_np),
            avoid=jnp.array(avoid_np),
            reach_clauses=jnp.array(reach_clauses),
            reach_negatives=jnp.array(reach_negatives),
            avoid_clauses=jnp.array(avoid_clauses),
            avoid_negatives=jnp.array(avoid_negatives),
            num_avoid_clauses=jnp.array(num_avoid_clauses),
            repeat_last=jnp.array(repeat_last),
            last_index=jnp.zeros_like(repeat_last),
        )
