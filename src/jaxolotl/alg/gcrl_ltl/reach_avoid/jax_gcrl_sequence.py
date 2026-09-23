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
from jaxolotl.alg.common.reach_avoid.jax_sequence import JaxReachAvoidSequence
from jaxolotl.alg.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.logic.utils import Clause
from jaxolotl.ltl.reach_avoid.sequence import EpsilonType


def _step_avoid_atoms(
    reach: list[Clause] | EpsilonType, avoid: list[Clause]
) -> list[str]:
    """Ordered avoid atoms for one step, folding in the zones forbidden by a
    negative-only reach clause so the policy is steered away from them."""
    atoms = [next(iter(clause.pos)) for clause in avoid]
    if not isinstance(reach, EpsilonType) and len(reach) == 1:
        clause = reach[0]
        if len(clause.pos) == 0:
            atoms += [atom for atom in sorted(clause.neg) if atom not in atoms]
    return atoms


class JaxGCRLSequence(JaxReachAvoidSequence):
    """Jax representation of a reach-avoid sequence with single reach and multiple avoid propositions."""

    reach_props: jax.Array  # [max_length] int
    avoid_props: jax.Array  # [max_length, max_avoid] int

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    @override
    def advance(self) -> "JaxGCRLSequence":
        is_last_step = self.depth == 1
        should_repeat = jnp.logical_and(
            is_last_step, self.last_index + 1 < self.repeat_last
        )

        def _repeat_step():
            return replace(self, last_index=self.last_index + 1)

        def _advance_step():
            # advance arrays one step
            new_reach = jnp.roll(self.reach, -1, axis=0)
            new_avoid = jnp.roll(self.avoid, -1, axis=0)
            reach_props = jnp.roll(self.reach_props, -1, axis=0)
            avoid_props = jnp.roll(self.avoid_props, -1, axis=0)

            # pad the last row with -1s
            new_reach = new_reach.at[-1, :].set(-1)
            new_avoid = new_avoid.at[-1, :].set(-1)
            new_reach_props = reach_props.at[-1].set(-1)
            new_avoid_props = avoid_props.at[-1, :].set(-1)

            return JaxGCRLSequence(
                reach=new_reach,
                avoid=new_avoid,
                reach_props=new_reach_props,
                avoid_props=new_avoid_props,
                repeat_last=self.repeat_last,
                last_index=jnp.zeros((), dtype=jnp.int32),
            )

        seq = jax.lax.cond(
            should_repeat,
            _repeat_step,
            _advance_step,
        )
        return seq

    @classmethod
    def from_reach_avoid_seqs(  # noqa: PLR0912
        cls,
        seqs: list[BooleanReachAvoidSequence],
        env: Environment | EnvWrapper,
    ) -> "JaxGCRLSequence":
        """
        Converts a list of BooleanReachAvoidSequences into a batched JaxGCRLSequence.

        Args:
            seqs: list of BooleanReachAvoidSequences to convert.
            propositions: list of proposition names in the environment.
            assignments: list of assignments in the environment.
        """
        max_length = max(len(seq.reach_avoid) for seq in seqs)
        max_avoid = 0
        for seq in seqs:
            for reach, avoid in seq.clauses:
                for clause in avoid:
                    if len(clause.neg) > 0:
                        raise ValueError(
                            "Avoid clauses for GCRL must not contain negated literals."
                        )
                    if len(clause.pos) > 1:
                        raise ValueError(
                            "Avoid clauses for GCRL must contain at most one literal."
                        )
                max_avoid = max(max_avoid, len(_step_avoid_atoms(reach, avoid)))
        # --- Assignments ---
        assignment_arrays = batch_assignments(
            seqs, env.assignments(), max_length=max_length
        )

        # --- Props ---
        prop_map = {name: i for i, name in enumerate(env.propositions)}
        reach_props = -np.ones((len(seqs), max_length), dtype=np.int32)
        avoid_props = -np.ones((len(seqs), max_length, max_avoid), dtype=np.int32)

        # --- Fill arrays ---
        for seq_idx, seq in enumerate(seqs):
            for i, (reach, avoid) in enumerate(seq.clauses):
                # Reach props
                if isinstance(reach, EpsilonType):
                    reach_props[seq_idx, i] = len(env.propositions)
                else:
                    if len(reach) != 1:
                        raise ValueError(
                            "Reach clauses must contain exactly one clause. "
                            f"Got {len(reach)} clauses."
                        )
                    clause = reach[0]
                    if len(clause.pos) == 1 and len(clause.neg) == 0:
                        atom = list(clause.pos)[0]
                    elif len(clause.pos) == 0 and len(clause.neg) > 0:
                        # A negative-only reach ("one step outside the listed
                        # zones", the accepting transition of a safety
                        # component): the wrapper progresses on any allowed
                        # assignment, so the goal only steers the policy —
                        # condition on an arbitrary allowed zone and treat the
                        # forbidden zones as avoid goals (see
                        # _step_avoid_atoms).
                        atom = next(p for p in env.propositions if p not in clause.neg)
                    else:
                        raise ValueError(
                            "Reach clauses for GCRL must be a single positive "
                            f"literal or purely negative. Got {clause!r}."
                        )
                    reach_props[seq_idx, i] = prop_map[atom]

                # Avoid props
                for c_idx, atom in enumerate(_step_avoid_atoms(reach, avoid)):
                    avoid_props[seq_idx, i, c_idx] = prop_map[atom]

        return cls(
            reach=jnp.array(assignment_arrays.reach),
            avoid=jnp.array(assignment_arrays.avoid),
            reach_props=jnp.array(reach_props),
            avoid_props=jnp.array(avoid_props),
            repeat_last=jnp.array(assignment_arrays.repeat_last),
            last_index=jnp.zeros_like(assignment_arrays.repeat_last),
        )

    @classmethod
    def padding_values(cls) -> "JaxGCRLSequence":
        """Return semantic scalar padding values for every array field."""
        return cls(
            reach=jnp.asarray(-1, dtype=jnp.int32),
            avoid=jnp.asarray(-1, dtype=jnp.int32),
            reach_props=jnp.asarray(-1, dtype=jnp.int32),
            avoid_props=jnp.asarray(-1, dtype=jnp.int32),
            repeat_last=jnp.asarray(1, dtype=jnp.int32),
            last_index=jnp.asarray(0, dtype=jnp.int32),
        )

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[BooleanReachAvoidSequence]],
        env: Environment | EnvWrapper,
    ) -> "JaxGCRLSequence":
        """Encode and pack Boolean sequences by LDBA state."""
        return batch_state_sequences(
            state_to_seqs,
            lambda sequences: cls.from_reach_avoid_seqs(sequences, env),
            cls.padding_values(),
        )
