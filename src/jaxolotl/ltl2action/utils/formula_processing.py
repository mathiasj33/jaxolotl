"""Utility functions for pre-processing formulas into a normal form suitable for LTL2Action."""

from jaxolotl.ltl.progression.ltl_parser import (
    ImplicationNode,
    LTLNode,
    NotNode,
    OrNode,
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


if __name__ == "__main__":
    from jaxolotl.ltl.progression.ltl_parser import parse

    formula_str = "G(a => F (b & c)) | (d => !e)"
    formula = parse(formula_str)
    print("Original formula:", formula)
    transformed_formula = replace_implication(formula)
    print("Transformed formula:", transformed_formula)
