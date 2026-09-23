import pytest

from jaxolotl.alg.ltl2action.utils.formula_processing import (
    holds_on_empty_suffix,
    replace_implication,
)
from jaxolotl.ltl.progression.ltl_parser import (
    AlwaysNode,
    AndNode,
    EventuallyNode,
    ImplicationNode,
    LTLNode,
    NotNode,
    OrNode,
    UntilNode,
    VarNode,
    parse,
)


def contains_implication(formula: LTLNode) -> bool:
    return isinstance(formula, ImplicationNode) or any(
        contains_implication(child) for child in formula.children
    )


def test_replaces_top_level_implication():
    formula = ImplicationNode(VarNode("a"), VarNode("b"))

    transformed = replace_implication(formula)

    assert transformed == OrNode(NotNode(VarNode("a")), VarNode("b"))
    assert not contains_implication(transformed)


def test_recursively_replaces_implications_in_both_operands():
    formula = ImplicationNode(
        ImplicationNode(VarNode("a"), VarNode("b")),
        ImplicationNode(VarNode("c"), VarNode("d")),
    )

    transformed = replace_implication(formula)

    expected = OrNode(
        NotNode(OrNode(NotNode(VarNode("a")), VarNode("b"))),
        OrNode(NotNode(VarNode("c")), VarNode("d")),
    )
    assert transformed == expected
    assert not contains_implication(transformed)


def test_recurses_through_boolean_and_temporal_nodes():
    formula = AlwaysNode(
        UntilNode(
            NotNode(ImplicationNode(VarNode("a"), VarNode("b"))),
            AndNode(
                EventuallyNode(ImplicationNode(VarNode("c"), VarNode("d"))),
                OrNode(
                    VarNode("e"),
                    ImplicationNode(VarNode("f"), VarNode("g")),
                ),
            ),
        )
    )

    transformed = replace_implication(formula)

    expected = AlwaysNode(
        UntilNode(
            NotNode(OrNode(NotNode(VarNode("a")), VarNode("b"))),
            AndNode(
                EventuallyNode(OrNode(NotNode(VarNode("c")), VarNode("d"))),
                OrNode(
                    VarNode("e"),
                    OrNode(NotNode(VarNode("f")), VarNode("g")),
                ),
            ),
        )
    )
    assert transformed == expected
    assert not contains_implication(transformed)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("true", True),
        ("false", False),
        ("a", False),
        ("!a", True),
        ("F a", False),
        ("a U b", False),
        ("G !a", True),
        ("!(F a)", True),
        ("!(G a)", False),
        ("G !a & F b", False),
        ("G !a | F b", True),
        ("G !a & G !b", True),
        ("a => F b", True),
    ],
)
def test_holds_on_empty_suffix(formula: str, expected: bool):
    assert holds_on_empty_suffix(parse(formula)) is expected


def test_formula_without_implications_is_unchanged():
    formula = UntilNode(
        NotNode(VarNode("a")),
        EventuallyNode(AndNode(VarNode("b"), VarNode("c"))),
    )

    transformed = replace_implication(formula)

    assert transformed is formula
    assert transformed == UntilNode(
        NotNode(VarNode("a")),
        EventuallyNode(AndNode(VarNode("b"), VarNode("c"))),
    )
