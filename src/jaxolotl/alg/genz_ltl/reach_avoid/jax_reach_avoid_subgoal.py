from collections import defaultdict
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl.alg.deep_ltl.reach_avoid.reach_avoid_sequence import (
    EpsilonType,
    ReachAvoidSequence,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.logic.assignment import Assignment


@dataclass
class ReachAvoidSubgoal:
    reach: Assignment
    avoid: list[Assignment]


class JaxReachAvoidSubgoal(eqx.Module):
    reach: jax.Array  # shape: () assignment to reach
    avoid: jax.Array  # shape: (num_assignments,) padded with -1

    # Representation for one-hot encoding of reach-avoid subgoals
    reach_one_hot: jax.Array  # shape: (num_props,) bitvector of propositions in reach
    avoid_one_hot: (
        jax.Array
    )  # shape: (num_assignments,) bitvector of assignments to avoid

    @classmethod
    def from_state_to_seqs(
        cls,
        state_to_seqs: dict[int, list[ReachAvoidSequence]],
        env: Environment | EnvWrapper,
    ) -> "JaxReachAvoidSubgoal":
        """Converts a mapping from LDBA states to lists of ReachAvoidSequences into a
        batched Jax reach-avoid subgoal.

        Returns:
            JaxReachAvoidSubgoal: with shape
                reach: (num_states, max_num_subgoals)
                avoid: (num_states, max_num_subgoals, num_assignments)
        """

        state_to_subgoals: defaultdict[int, list[ReachAvoidSubgoal]] = defaultdict(list)
        for state, seqs in state_to_seqs.items():
            for seq in seqs:
                subgoals = cls._seq_to_subgoals(seq)
                state_to_subgoals[state].extend(subgoals)
        max_subgoals = max(len(subgoals) for subgoals in state_to_subgoals.values())
        num_states = max(state_to_subgoals.keys()) + 1
        # Use numpy arrays and then convert to jax arrays for efficiency
        reach = -np.ones((num_states, max_subgoals), dtype=np.int32)
        avoid = -np.ones(
            (num_states, max_subgoals, len(env.assignments())), dtype=np.int32
        )
        num_props = len(env.propositions)
        reach_one_hot = np.zeros((num_states, max_subgoals, num_props), dtype=np.int32)
        avoid_one_hot = np.zeros(
            (num_states, max_subgoals, len(env.assignments())), dtype=np.int32
        )
        for state, subgoals in state_to_subgoals.items():
            for subgoal_idx, subgoal in enumerate(subgoals):
                reach[state, subgoal_idx] = env.assignments().index(subgoal.reach)
                for i, assignment in enumerate(subgoal.avoid):
                    avoid[state, subgoal_idx, i] = env.assignments().index(assignment)
                # Set bits for propositions in reach and avoid
                for i, prop in enumerate(env.propositions):
                    reach_one_hot[state, subgoal_idx, i] = int(prop in subgoal.reach)
                for i, assignment in enumerate(env.assignments()):
                    avoid_one_hot[state, subgoal_idx, i] = int(
                        assignment in subgoal.avoid
                    )

        return cls(
            reach=jnp.array(reach),
            avoid=jnp.array(avoid),
            reach_one_hot=jnp.array(reach_one_hot),
            avoid_one_hot=jnp.array(avoid_one_hot),
        )

    @staticmethod
    def _seq_to_subgoals(seq: ReachAvoidSequence) -> list[ReachAvoidSubgoal]:
        """Converts a ReachAvoidSequence to a list of ReachAvoidSubgoals."""
        if len(seq.reach_avoid) == 0:  # e.g. sink state
            return []
        if isinstance(seq.reach_avoid[0][0], EpsilonType):
            next_reaches = seq.reach_avoid[1][0]
            assert not isinstance(next_reaches, EpsilonType)
            return [ReachAvoidSubgoal(reach=r, avoid=[]) for r in next_reaches]
        reach, avoid = seq.reach_avoid[0]
        assert not isinstance(reach, EpsilonType)
        return [ReachAvoidSubgoal(reach=r, avoid=list(avoid)) for r in reach]
