"""Utilities for preprocessing LTL formulas into JaxLDBAs and JaxReachAvoidSubgoals."""

import jax
import jax.numpy as jnp

from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.alg.reach_avoid.preprocessing import (
    preprocess_formulas as preprocess_reach_avoid_formulas,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA


def preprocess_formulas(
    formulas: list[str], env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxReachAvoidSubgoal]:
    """Converts a list of formulas into a batched JaxLDBA and batched JaxReachAvoidSubgoal,
    with a set of subgoals for every LDBA state."""

    return preprocess_reach_avoid_formulas(
        formulas,
        env,
        JaxReachAvoidSubgoal.from_state_to_seqs,
        _batch_subgoals,
    )


def _batch_subgoals(
    subgoals: list[JaxReachAvoidSubgoal],
) -> JaxReachAvoidSubgoal:
    """Batch multiple JaxReachAvoidSubgoals into a single JaxReachAvoidSubgoal with an added batch dimension.

    Args:
        subgoals: List of JaxReachAvoidSubgoal to batch. Shape: (num_states, num_subgoals, num_assignments)

    Returns:
        JaxReachAvoidSubgoal: Batched subgoals. Shape: (batch_size, max_num_states, max_num_subgoals, num_assignments) for avoid
    """
    max_num_states = max(goal.reach.shape[0] for goal in subgoals)
    max_num_subgoals = max(goal.reach.shape[1] for goal in subgoals)

    def pad_leaf(x):
        # Calculate padding
        pad_axis0 = max(0, max_num_states - x.shape[0])
        pad_axis1 = max(0, max_num_subgoals - x.shape[1])

        pad_config = (
            (0, pad_axis0),  # Axis 0 (states)
            (0, pad_axis1),  # Axis 1 (subgoals)
        )
        pad_config += ((0, 0),) * (x.ndim - len(pad_config))
        return jnp.pad(x, pad_width=pad_config, mode="constant", constant_values=-1)

    subgoals = jax.tree.map(pad_leaf, subgoals)
    return jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0),
        *subgoals,
    )
