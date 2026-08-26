from jaxolotl.alg.struct_ltl.reach_avoid.boolean_reach_avoid_sequence import (
    BooleanReachAvoidSequence,
)
from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import MultiAndNode, MultiOrNode, VarNode
from jaxolotl.ltl.reach_avoid.sequence import EPSILON

ASSIGNMENTS = (
    Assignment(),
    Assignment("a"),
    Assignment("b"),
    Assignment("c"),
    Assignment("d"),
)


def test_expand_clauses_empty_sequence_preserves_metadata():
    sequence = BooleanReachAvoidSequence([], ASSIGNMENTS, repeat_last=3)

    expanded = sequence.expand_clauses()

    assert len(expanded) == 1
    assert expanded[0] is not sequence
    assert expanded[0].reach_avoid_formulas == ()
    assert expanded[0].assignments == ASSIGNMENTS
    assert expanded[0].repeat_last == 3


def test_expand_clauses_leaves_non_disjunctive_reaches_unchanged():
    avoid = VarNode("d")
    reach = MultiAndNode([VarNode("a"), VarNode("b")])
    formulas = ((EPSILON, avoid), (None, None), (reach, avoid))
    sequence = BooleanReachAvoidSequence(formulas, ASSIGNMENTS, repeat_last=2)

    expanded = sequence.expand_clauses()

    assert len(expanded) == 1
    assert expanded[0].reach_avoid_formulas == formulas
    assert expanded[0].assignments == ASSIGNMENTS
    assert expanded[0].repeat_last == 2


def test_expand_clauses_returns_cartesian_product_in_operand_order():
    a = VarNode("a")
    b = VarNode("b")
    c = VarNode("c")
    d = VarNode("d")
    first_avoid = VarNode("c")
    second_avoid = VarNode("a")
    trailing_reach = MultiAndNode([a, d])
    sequence = BooleanReachAvoidSequence(
        (
            (MultiOrNode([a, b]), first_avoid),
            (MultiOrNode([c, d]), second_avoid),
            (trailing_reach, None),
        ),
        ASSIGNMENTS,
        repeat_last=4,
    )

    expanded = sequence.expand_clauses()

    assert [item.reach_avoid_formulas for item in expanded] == [
        ((a, first_avoid), (c, second_avoid), (trailing_reach, None)),
        ((a, first_avoid), (d, second_avoid), (trailing_reach, None)),
        ((b, first_avoid), (c, second_avoid), (trailing_reach, None)),
        ((b, first_avoid), (d, second_avoid), (trailing_reach, None)),
    ]
    assert all(item.assignments == ASSIGNMENTS for item in expanded)
    assert all(item.repeat_last == 4 for item in expanded)
