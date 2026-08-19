import pytest

from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import (
    AndNode,
    FalseNode,
    MultiAndNode,
    MultiOrNode,
    NotNode,
    VarNode,
)
from jaxolotl.ltl.logic.utils import (
    Clause,
    compute_sat,
    formula_to_clauses,
    synthesize_formula,
)


def test_compute_sat():
    # Formula: a AND (NOT b)
    formula = AndNode(VarNode("a"), NotNode(VarNode("b")))

    # All possible assignments for props
    all_assignments: tuple[Assignment, ...] = (
        Assignment(frozenset()),
        Assignment(frozenset({"a"})),
        Assignment(frozenset({"b"})),
        Assignment(frozenset({"a", "b"})),
    )

    # Expected satisfying assignments
    expected_sat = frozenset([Assignment(frozenset({"a"}))])

    # Compute the satisfying assignments
    actual_sat = compute_sat(formula, all_assignments)

    assert expected_sat == actual_sat


def test_synthesize_formula_simple():
    props = ("a", "b")
    possible_assignments = frozenset(
        [
            Assignment(frozenset()),
            Assignment(frozenset({"a"})),
            Assignment(frozenset({"b"})),
            Assignment(frozenset({"a", "b"})),
        ]
    )
    # Target formula is 'a'
    target_assignments = frozenset(
        [Assignment(frozenset({"a"})), Assignment(frozenset({"a", "b"}))]
    )

    # Synthesize the formula
    formula = synthesize_formula(target_assignments, possible_assignments, props)

    # The synthesized formula should be equivalent to VarNode("a")
    # We can check this by evaluating its satisfying set
    sat_assignments = compute_sat(formula, tuple(possible_assignments))
    assert target_assignments == sat_assignments


def test_synthesize_formula_with_dont_cares():
    props = ("a", "b", "c")
    # Let's say 'c' is a "don't care" proposition in the context of the target
    possible_assignments = frozenset(
        [
            Assignment(frozenset()),
            Assignment(frozenset({"a"})),
            Assignment(frozenset({"b"})),
            Assignment(frozenset({"c"})),
            Assignment(frozenset({"a", "b"})),
            Assignment(frozenset({"a", "c"})),
            Assignment(frozenset({"b", "c"})),
            Assignment(frozenset({"a", "b", "c"})),
        ]
    )
    # Target formula is 'a AND b'
    target_assignments = frozenset(
        [
            Assignment(frozenset({"a", "b"})),
            Assignment(frozenset({"a", "b", "c"})),
        ]
    )

    # Synthesize the formula
    formula = synthesize_formula(target_assignments, possible_assignments, props)

    # The synthesized formula should be equivalent to 'a AND b'
    # We can check this by evaluating its satisfying set
    sat_assignments = compute_sat(formula, tuple(possible_assignments))
    assert target_assignments == sat_assignments


def test_synthesize_formula_realistic_with_dont_cares():
    # Inspired by the example in local/minimize.py
    # Props represent mutually exclusive regions and an item.
    props = ("region_a", "region_b", "item_c")

    # Universe of possible assignments is constrained: a and b are mutually exclusive.
    # This creates "don't care" states for combinations like {a, b}
    possible_assignments = frozenset(
        [
            Assignment(frozenset()),
            Assignment(frozenset({"item_c"})),
            Assignment(frozenset({"region_a"})),
            Assignment(frozenset({"region_a", "item_c"})),
            Assignment(frozenset({"region_b"})),
            Assignment(frozenset({"region_b", "item_c"})),
        ]
    )

    # Target formula is 'region_a'
    target_assignments = frozenset(
        [
            Assignment(frozenset({"region_a"})),
            Assignment(frozenset({"region_a", "item_c"})),
        ]
    )

    # Synthesize the formula
    formula = synthesize_formula(target_assignments, possible_assignments, props)

    # The synthesized formula should be equivalent to 'region_a'.
    # We can check this by evaluating its satisfying set against the universe.
    sat_assignments = compute_sat(formula, tuple(possible_assignments))
    assert target_assignments == sat_assignments


def test_synthesize_formula_complex_logic():
    # More complex scenario with regions and items.
    props = ("region_a", "region_b", "vase", "crate")

    # Universe: regions are mutually exclusive, but items can coexist.
    regions = [frozenset(), frozenset({"region_a"}), frozenset({"region_b"})]
    items = [
        frozenset(),
        frozenset({"vase"}),
        frozenset({"crate"}),
        frozenset({"vase", "crate"}),
    ]
    possible_assignments = frozenset(
        [Assignment(r | i) for r in regions for i in items]
    )

    # Target formula: "(region_a AND vase) OR (region_b AND crate)"
    target_assignments = frozenset(
        [
            Assignment(frozenset({"region_a", "vase"})),
            Assignment(frozenset({"region_a", "vase", "crate"})),
            Assignment(frozenset({"region_b", "crate"})),
            Assignment(frozenset({"region_b", "vase", "crate"})),
        ]
    )

    # Synthesize the formula
    formula = synthesize_formula(target_assignments, possible_assignments, props)

    # The synthesized formula should be equivalent to the target.
    # We verify by checking the satisfying set.
    sat_assignments = compute_sat(formula, tuple(possible_assignments))
    assert target_assignments == sat_assignments


def test_formula_to_clauses_false():
    formula = FalseNode()
    assert formula_to_clauses(formula) == []


def test_formula_to_clauses_single_positive_literal():
    formula = VarNode("a")
    expected = [Clause(pos=frozenset({"a"}), neg=frozenset())]
    assert formula_to_clauses(formula) == expected


def test_formula_to_clauses_single_negative_literal():
    formula = NotNode(VarNode("a"))
    expected = [Clause(pos=frozenset(), neg=frozenset({"a"}))]
    assert formula_to_clauses(formula) == expected


def test_formula_to_clauses_single_clause():
    # a AND b AND (NOT c)
    formula = MultiAndNode([VarNode("a"), VarNode("b"), NotNode(VarNode("c"))])
    expected = [Clause(pos=frozenset({"a", "b"}), neg=frozenset({"c"}))]
    assert formula_to_clauses(formula) == expected


def test_formula_to_clauses_dnf():
    # (a AND (NOT b)) OR (c AND d)
    clause1 = MultiAndNode([VarNode("a"), NotNode(VarNode("b"))])
    clause2 = MultiAndNode([VarNode("c"), VarNode("d")])
    formula = MultiOrNode([clause1, clause2])

    expected = [
        Clause(pos=frozenset({"a"}), neg=frozenset({"b"})),
        Clause(pos=frozenset({"c", "d"}), neg=frozenset()),
    ]
    # The order of clauses from MultiOrNode is not guaranteed
    result_clauses = formula_to_clauses(formula)
    assert len(result_clauses) == len(expected)
    assert set(result_clauses) == set(expected)


def test_formula_to_clauses_mixed_literals_and_clauses():
    # a OR (b AND c)
    clause1 = VarNode("a")
    clause2 = MultiAndNode([VarNode("b"), VarNode("c")])
    formula = MultiOrNode([clause1, clause2])

    expected = [
        Clause(pos=frozenset({"a"}), neg=frozenset()),
        Clause(pos=frozenset({"b", "c"}), neg=frozenset()),
    ]
    result_clauses = formula_to_clauses(formula)
    assert len(result_clauses) == len(expected)
    assert set(result_clauses) == set(expected)


def test_formula_to_clauses_not_dnf_raises_error():
    # a AND (b OR c) - not in DNF that the function supports
    formula = MultiAndNode([VarNode("a"), MultiOrNode([VarNode("b"), VarNode("c")])])
    with pytest.raises(
        ValueError, match="Invalid operand in AND node for DNF conversion."
    ):
        formula_to_clauses(formula)


def test_formula_to_clauses_nested_or_raises_error():
    # (a OR b) OR c
    # The function expects each operand of an OR to be a single clause
    formula = MultiOrNode([MultiOrNode([VarNode("a"), VarNode("b")]), VarNode("c")])
    with pytest.raises(
        ValueError, match="Each operand in OR node must correspond to a single clause."
    ):
        formula_to_clauses(formula)
