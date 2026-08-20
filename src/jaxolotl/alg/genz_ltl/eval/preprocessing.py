"""Utilities for preprocessing LTL formulas into JaxLDBAs and JaxReachAvoidSubgoals."""

from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.alg.reach_avoid.preprocessing import (
    preprocess_formulas as preprocess_reach_avoid_formulas,
)
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eqx_utils.batching import pad_and_stack
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
    """Pad and stack subgoals along a formula batch axis."""
    return pad_and_stack(subgoals, JaxReachAvoidSubgoal.padding_values())
