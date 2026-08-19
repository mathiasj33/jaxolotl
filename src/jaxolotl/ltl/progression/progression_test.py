import pytest

from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.progression.ltl_parser import (
    AlwaysNode,
    AndNode,
    EventuallyNode,
    FalseNode,
    ImplicationNode,
    LTLNode,
    NotNode,
    OrNode,
    TrueNode,
    UntilNode,
    VarNode,
)
from jaxolotl.ltl.progression.progression import progress, simplify


def test_progress_preserves_boolean_constants():
    true = TrueNode()
    false = FalseNode()

    assert progress(true, Assignment()) is true
    assert progress(false, Assignment("a")) is false


def test_progress_evaluates_atomic_propositions():
    formula = VarNode("a")

    assert progress(formula, Assignment("a")) == TrueNode()
    assert progress(formula, Assignment()) == FalseNode()


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        (NotNode(VarNode("a")), NotNode(TrueNode())),
        (
            AndNode(VarNode("a"), VarNode("b")),
            AndNode(TrueNode(), FalseNode()),
        ),
        (
            OrNode(VarNode("a"), VarNode("b")),
            OrNode(TrueNode(), FalseNode()),
        ),
        (
            ImplicationNode(VarNode("a"), VarNode("b")),
            ImplicationNode(TrueNode(), FalseNode()),
        ),
    ],
)
def test_progress_recurses_through_boolean_operators(
    formula: LTLNode, expected: LTLNode
):
    assert progress(formula, Assignment("a")) == expected


def test_progress_eventually_retains_the_original_obligation():
    formula = EventuallyNode(VarNode("a"))

    progressed = progress(formula, Assignment("a"))

    assert progressed == OrNode(TrueNode(), formula)
    assert progressed.children[1] is formula


def test_progress_always_retains_the_original_obligation():
    formula = AlwaysNode(VarNode("a"))

    progressed = progress(formula, Assignment())

    assert progressed == AndNode(FalseNode(), formula)
    assert progressed.children[1] is formula


@pytest.mark.parametrize(
    ("current_assignment", "expected_left", "expected_right"),
    [
        (Assignment("a"), FalseNode(), TrueNode()),
        (Assignment("b"), TrueNode(), FalseNode()),
        (Assignment("a", "b"), TrueNode(), TrueNode()),
    ],
)
def test_progress_until_combines_completion_and_continuation(
    current_assignment: Assignment,
    expected_left: LTLNode,
    expected_right: LTLNode,
):
    formula = UntilNode(VarNode("a"), VarNode("b"))

    progressed = progress(formula, current_assignment)

    assert progressed == OrNode(
        expected_left,
        AndNode(expected_right, formula),
    )


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        (AndNode(TrueNode(), VarNode("a")), VarNode("a")),
        (OrNode(FalseNode(), VarNode("a")), VarNode("a")),
        (NotNode(TrueNode()), FalseNode()),
    ],
)
def test_simplify_applies_boolean_identities(formula: LTLNode, expected: LTLNode):
    assert simplify(formula) == expected


class UnsupportedNode(LTLNode):
    def __eq__(self, other) -> bool:
        return isinstance(other, UnsupportedNode)

    def __hash__(self) -> int:
        return hash(UnsupportedNode)

    @property
    def children(self) -> list[LTLNode]:
        return []


def test_progress_rejects_unsupported_node_types():
    with pytest.raises(NotImplementedError, match="UnsupportedNode"):
        progress(UnsupportedNode(), Assignment())
