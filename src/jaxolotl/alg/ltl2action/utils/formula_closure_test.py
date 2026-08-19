import pytest

from jaxolotl.alg.ltl2action.utils.formula_closure import (
    ClosureGraphNode,
    FormulaClosureGraph,
)
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.progression.ltl_parser import TrueNode


def test_initial_node_requires_graph_to_be_built():
    graph = FormulaClosureGraph("F a")

    with pytest.raises(ValueError, match="has not been built"):
        _ = graph.initial_node


def test_build_creates_the_complete_progression_closure():
    empty = Assignment()
    a = Assignment("a")
    assignments = (empty, a)
    graph = FormulaClosureGraph("F a")

    graph.build(assignments)

    initial = graph.initial_node
    terminal = graph.nodes[TrueNode()]

    assert graph.num_nodes == 2
    assert set(graph.formula_graphs) == set(graph.nodes)
    assert initial.formula == graph.formula_node
    assert initial.edges[empty] is initial
    assert initial.edges[a] is terminal
    assert terminal.edges[empty] is terminal
    assert terminal.edges[a] is terminal


def test_every_node_has_one_canonical_target_per_assignment():
    assignments = (Assignment(), Assignment("a"), Assignment("b"))
    graph = FormulaClosureGraph("a U b")

    graph.build(assignments)

    for node in graph.nodes.values():
        assert set(node.edges) == set(assignments)
        for target in node.edges.values():
            assert target is graph.nodes[target.formula]


def test_closure_graph_nodes_do_not_share_edge_dictionaries():
    first_formula = TrueNode()
    first = ClosureGraphNode(first_formula)
    second = ClosureGraphNode(TrueNode())

    first.edges[Assignment()] = first

    assert first.formula is first_formula
    assert second.edges == {}
