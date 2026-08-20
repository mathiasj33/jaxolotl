from dataclasses import dataclass

import numpy.testing as npt

from jaxolotl.alg.struct_ltl.eval.preprocessing import (
    _batch_graph_sequences,
    _batch_sequences,
    _batch_tokenized_sequences,
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
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import VarNode


@dataclass(frozen=True)
class EncodingEnv:
    propositions = ("red", "green")

    def assignments(self) -> list[Assignment]:
        return [
            Assignment("red"),
            Assignment("green"),
            Assignment(),
        ]


def test_batch_graph_sequences_preserves_padded_graph_invariants():
    env = EncodingEnv()
    short = BooleanReachAvoidSequence([(VarNode("red"), None)], env.assignments())
    long = BooleanReachAvoidSequence(
        [(VarNode("red"), None), (VarNode("green"), None)], env.assignments()
    )
    first = JaxGraphReachAvoidSequence.from_state_to_seqs({0: [short]}, env)  # type: ignore
    second = JaxGraphReachAvoidSequence.from_state_to_seqs({0: [long], 1: [short]}, env)  # type: ignore

    batched = _batch_graph_sequences([first, second])

    assert batched.reach.shape[:3] == (2, 2, 1)
    npt.assert_array_equal(batched.reach[0, 1], -1)
    npt.assert_array_equal(batched.repeat_last[0, 1], [1])
    npt.assert_array_equal(batched.last_index[0, 1], [0])
    npt.assert_array_equal(batched.reach_graphs.n_node[0, 1], [[1, 1]])
    npt.assert_array_equal(batched.reach_graphs.n_edge[0, 1], [[0, 0]])
    assert not batched.reach_graphs.nodes["mask"][0, 1].any()  # type: ignore


def test_batch_clause_sequences_uses_field_specific_padding():
    env = EncodingEnv()
    short = BooleanReachAvoidSequence(
        [(VarNode("red"), VarNode("green"))], env.assignments()
    )
    first = JaxClauseReachAvoidSequence.from_state_to_seqs({0: [short]}, env)  # type: ignore
    second = JaxClauseReachAvoidSequence.from_state_to_seqs(
        {0: [short], 1: [short]},
        env,  # type: ignore
    )

    batched = _batch_sequences([first, second])

    npt.assert_array_equal(batched.reach_clauses[0, 1], -1)
    assert not batched.reach_negatives[0, 1].any()
    assert not batched.avoid_negatives[0, 1].any()
    npt.assert_array_equal(batched.num_avoid_clauses[0, 1], [[0]])
    npt.assert_array_equal(batched.repeat_last[0, 1], [1])
    npt.assert_array_equal(batched.last_index[0, 1], [0])


def test_batch_tokenized_sequences_uses_field_specific_padding():
    env = EncodingEnv()
    short = BooleanReachAvoidSequence([(VarNode("red"), None)], env.assignments())
    first = JaxTokenizedReachAvoidSequence.from_state_to_seqs({0: [short]}, env)  # type: ignore
    second = JaxTokenizedReachAvoidSequence.from_state_to_seqs(
        {0: [short], 1: [short]},
        env,  # type: ignore
    )

    batched = _batch_tokenized_sequences([first, second])

    npt.assert_array_equal(batched.reach_tokens[0, 1], -1)
    npt.assert_array_equal(batched.avoid_tokens[0, 1], -1)
    npt.assert_array_equal(batched.repeat_last[0, 1], [1])
    npt.assert_array_equal(batched.last_index[0, 1], [0])
