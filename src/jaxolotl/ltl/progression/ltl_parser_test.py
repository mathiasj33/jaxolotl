import pytest

from jaxolotl.ltl.progression.ltl_parser import (
    AlwaysNode,
    AndNode,
    EventuallyNode,
    FalseNode,
    ImplicationNode,
    LTLParser,
    NotNode,
    OrNode,
    TrueNode,
    UntilNode,
    VarNode,
    parse,
)


def test_parses_atomic_propositions_and_constants():
    assert parse("proposition_12") == VarNode("proposition_12")
    assert parse("true") == TrueNode()
    assert parse("1") == TrueNode()
    assert parse("false") == FalseNode()
    assert parse("0") == FalseNode()


def test_operator_precedence():
    formula = parse("!a U b & c | d => F G e")

    expected = ImplicationNode(
        OrNode(
            AndNode(UntilNode(NotNode(VarNode("a")), VarNode("b")), VarNode("c")),
            VarNode("d"),
        ),
        EventuallyNode(AlwaysNode(VarNode("e"))),
    )
    assert formula == expected


def test_parentheses_override_precedence():
    formula = parse("!(a | b) U (c & (d => e))")

    expected = UntilNode(
        NotNode(OrNode(VarNode("a"), VarNode("b"))),
        AndNode(VarNode("c"), ImplicationNode(VarNode("d"), VarNode("e"))),
    )
    assert formula == expected


def test_repeated_binary_operators_are_left_associative():
    assert parse("a & b & c") == AndNode(
        AndNode(VarNode("a"), VarNode("b")), VarNode("c")
    )
    assert parse("a U b U c") == UntilNode(
        UntilNode(VarNode("a"), VarNode("b")), VarNode("c")
    )


def test_ast_children_and_size_properties():
    expected_num_nodes = 5
    expected_num_edges = 4
    left = VarNode("a")
    right = EventuallyNode(NotNode(VarNode("b")))
    formula = AndNode(left, right)

    assert formula.children == [left, right]
    assert formula.num_nodes == expected_num_nodes
    assert formula.num_edges == expected_num_edges
    assert VarNode("a").children == []


def test_equivalent_nodes_have_equal_hashes():
    first = parse("G (a => F b)")
    second = AlwaysNode(ImplicationNode(VarNode("a"), EventuallyNode(VarNode("b"))))

    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("", "Unexpected token"),
        ("a b", "Unexpected token at the end"),
        ("(a | b", r"Expected '\)'"),
        ("a &", "Unexpected token"),
    ],
)
def test_rejects_incomplete_or_malformed_formulas(formula: str, message: str):
    with pytest.raises(SyntaxError, match=message):
        LTLParser(formula).parse()
