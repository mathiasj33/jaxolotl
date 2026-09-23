"""Utility functions for pre-processing formulas into a normal form suitable for LTL2Action."""

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


def replace_implication(formula: LTLNode) -> LTLNode:
    """Replaces all implications in the formula with their equivalent disjunction form.

    Args:
        formula: The input LTL formula as an LTLNode.

    Returns:
        The transformed LTL formula with implications replaced.
    """

    if isinstance(formula, ImplicationNode):
        left = replace_implication(formula.left)
        right = replace_implication(formula.right)
        return OrNode(NotNode(left), right)
    else:
        # Recursively process child nodes
        for attr in dir(formula):
            child = getattr(formula, attr)
            if isinstance(child, LTLNode):
                replaced_child = replace_implication(child)
                setattr(formula, attr, replaced_child)
        return formula


def holds_on_empty_suffix(formula: LTLNode) -> bool:
    """Evaluate the formula over an empty (zero-length) remaining trace.

    Atomic propositions and strong eventualities (F, U) are false on the empty
    word, while G holds vacuously. A progression residual that holds on the
    empty suffix carries no pending liveness obligation, so under truncated
    (finite) semantics the trace seen so far already satisfies the original
    formula. Only valid for finite formula sets: an infinite-semantics formula
    like ``GF a`` also holds on the empty suffix.
    """
    match formula:
        case TrueNode() | AlwaysNode():
            return True
        case FalseNode() | VarNode() | EventuallyNode() | UntilNode():
            return False
        case NotNode():
            return not holds_on_empty_suffix(formula.operand)
        case AndNode():
            return holds_on_empty_suffix(formula.left) and holds_on_empty_suffix(
                formula.right
            )
        case OrNode():
            return holds_on_empty_suffix(formula.left) or holds_on_empty_suffix(
                formula.right
            )
        case ImplicationNode():
            return not holds_on_empty_suffix(formula.left) or holds_on_empty_suffix(
                formula.right
            )
        case _:
            raise TypeError(f"Unsupported LTL node type: {type(formula).__name__}")


if __name__ == "__main__":
    from jaxolotl.ltl.progression.ltl_parser import parse

    formula_str = "G(a => F (b & c)) | (d => !e)"
    formula = parse(formula_str)
    print("Original formula:", formula)
    transformed_formula = replace_implication(formula)
    print("Transformed formula:", transformed_formula)
