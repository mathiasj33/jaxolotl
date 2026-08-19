import functools

import spot

from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.progression.ltl_parser import (
    AlwaysNode,
    AndNode,
    EventuallyNode,
    FalseNode,
    ImplicationNode,
    LTLNode,
    LTLParser,
    NotNode,
    OrNode,
    TrueNode,
    UntilNode,
    VarNode,
    parse,
)


@functools.cache
def progress(formula: LTLNode, assignment: Assignment) -> LTLNode:  # noqa: PLR0911
    """Applies LTL progression to a formula given a current assignment.

    Args:
        formula (LTLNode): The LTL formula to be progressed.
        assignment (Assignment): The current assignment of atomic propositions.

    Returns:
        LTLNode: The progressed LTL formula.
    """
    if isinstance(formula, TrueNode | FalseNode):
        return formula
    elif isinstance(formula, VarNode):
        if formula.name in assignment.true_propositions:
            return TrueNode()
        return FalseNode()
    elif isinstance(formula, NotNode):
        prog = progress(formula.operand, assignment)
        return NotNode(prog)
    elif isinstance(formula, AndNode):
        left_prog = progress(formula.left, assignment)
        right_prog = progress(formula.right, assignment)
        return AndNode(left_prog, right_prog)
    elif isinstance(formula, OrNode):
        left_prog = progress(formula.left, assignment)
        right_prog = progress(formula.right, assignment)
        return OrNode(left_prog, right_prog)
    elif isinstance(formula, ImplicationNode):
        left_prog = progress(formula.left, assignment)
        right_prog = progress(formula.right, assignment)
        return ImplicationNode(left_prog, right_prog)
    elif isinstance(formula, EventuallyNode):
        prog = progress(formula.operand, assignment)
        return OrNode(prog, formula)
    elif isinstance(formula, AlwaysNode):
        prog = progress(formula.operand, assignment)
        return AndNode(prog, formula)
    elif isinstance(formula, UntilNode):
        left_prog = progress(formula.left, assignment)
        right_prog = progress(formula.right, assignment)
        return OrNode(right_prog, AndNode(left_prog, formula))
    else:
        raise NotImplementedError(f"Progression not implemented for {type(formula)}")


@functools.cache
def simplify(formula: LTLNode) -> LTLNode:
    """Simplifies an LTL formula using Spot.

    Args:
        formula (LTLNode): The LTL formula to be simplified.

    Returns:
        LTLNode: The simplified LTL formula.
    """
    formula_str = str(formula)
    spot_formula = spot.formula(formula_str)
    simplified = spot.simplify(spot_formula).unabbreviate("RWMG")
    spot_str = str(simplified)
    if len(spot_str) > len(formula_str) + 10:
        # If Spot's simplification is much longer, return the original formula
        return formula
    if "M" in spot_str or "R" in spot_str or "W" in spot_str or "G" in spot_str:
        raise ValueError("Simplification did not fully unabbreviate the formula.")
    return parse(str(simplified))


# TODO: write tests


if __name__ == "__main__":
    formula = "!c U d"
    parser = LTLParser(formula)
    ast = parser.parse()
    progressed = progress(ast, Assignment(frozenset({"d"})))
    print("Progressed:", progressed)
    simplified = simplify(progressed)
    print("Simplified:", simplified)
