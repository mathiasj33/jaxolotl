from dataclasses import dataclass

import jax
import numpy.testing as npt
import pytest

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
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import VarNode
from jaxolotl.ltl.reach_avoid.sequence import EPSILON


@dataclass(frozen=True)
class EncodingEnv:
    propositions = ("red", "green")

    def assignments(self) -> list[Assignment]:
        return [
            Assignment(frozenset({"red"})),
            Assignment(frozenset({"green"})),
            Assignment(frozenset()),
        ]


@pytest.mark.parametrize(
    "encoder",
    [
        JaxClauseReachAvoidSequence,
        JaxTokenizedReachAvoidSequence,
        JaxGraphReachAvoidSequence,
    ],
)
def test_structured_state_encoding_matches_flat_encoding(encoder):
    env = EncodingEnv()
    sequence = BooleanReachAvoidSequence(
        [(VarNode("red"), VarNode("green")), (EPSILON, None)],
        env.assignments(),
        repeat_last=4,
    )

    flat = encoder.from_reach_avoid_seqs([sequence], env)
    by_state = encoder.from_state_to_seqs({0: [], 2: [sequence]}, env)

    flat_leaves = jax.tree.leaves(flat)
    state_leaves = jax.tree.leaves(by_state)
    assert len(flat_leaves) == len(state_leaves)
    for flat_leaf, state_leaf in zip(flat_leaves, state_leaves, strict=True):
        npt.assert_array_equal(state_leaf[2, 0], flat_leaf[0])

    assert by_state.reach.shape[:2] == (3, 1)
    npt.assert_array_equal(by_state.reach[1], -1)
    assert by_state.repeat_last[2, 0] == 4
    assert by_state.repeat_last[1, 0] == 1
