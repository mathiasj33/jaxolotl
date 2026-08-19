import functools
import itertools
from dataclasses import dataclass

from sympy import SOPform, Symbol
from sympy.logic.boolalg import And, BooleanFalse, BooleanTrue, Not, Or

from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.logic.boolean_parser import (
    BooleanNode,
    FalseNode,
    MultiAndNode,
    MultiOrNode,
    NotNode,
    VarNode,
)


@functools.cache
def compute_sat(
    graph: BooleanNode, all_assignments: tuple[Assignment, ...]
) -> frozenset[Assignment]:
    """Computes the set of assignments that satisfy a given Boolean formula. Cached."""
    return frozenset(a for a in all_assignments if graph.eval(a))


@functools.cache
def synthesize_formula(
    target_assignments: frozenset[Assignment],
    possible_assignments: frozenset[Assignment],
    props: tuple[str, ...],
) -> BooleanNode:
    """
    Generates a minimal Boolean formula that is true for 'target_assignments'
    and false for the rest of 'possible_assignments'.

    Uses the Quine-McCluskey algorithm via SymPy's SOPform function.

    Any assignment NOT in 'possible_assignments' is treated as a 'don't care',
    allowing the solver to simplify the logic further.
    """

    targets = set(target_assignments)
    possible = set(possible_assignments)

    # 1. Identify all variables (propositions) involved in the universe
    all_props: set[str] = set(props)

    assert all_props, "No propositions provided for formula synthesis."

    # Sort variables to ensure deterministic ordering for SymPy
    sorted_vars = sorted(all_props)
    sympy_vars = [Symbol(v) for v in sorted_vars]

    # 2. Helper to convert an Assignment to a binary tuple based on sorted_vars
    def to_bit_pattern(assignment: "Assignment") -> tuple[int, ...]:
        return tuple(1 if v in assignment else 0 for v in sorted_vars)

    # 3. Categorize truth table inputs
    # We map every possible combination of variables to ON (minterms) or DC (don't cares).
    target_patterns = {to_bit_pattern(a) for a in targets}
    universe_patterns = {to_bit_pattern(a) for a in possible}

    minterms = []
    dontcares = []

    # Iterate through all theoretically possible binary combinations (2^N)
    # Note: This is feasible for N < ~15. For larger N, a different approach (Espresso) is needed.
    for pattern in itertools.product([0, 1], repeat=len(sorted_vars)):
        if pattern in target_patterns:
            minterms.append(pattern)
        elif pattern not in universe_patterns:
            # If it's not in the universe of valid assignments, we don't care.
            dontcares.append(pattern)

    # 4. Use SymPy to generate the minimized Sum of Products (SOP)
    # SOPform uses Quine-McCluskey algorithm
    expr = SOPform(sympy_vars, minterms, dontcares=dontcares)

    # 5. Convert SymPy expression to graph
    return sympy_to_graph(expr, sorted_vars)


def sympy_to_graph(expr, sorted_vars_names: list[str]) -> BooleanNode:
    """Recursively converts a SymPy boolean expression to a custom graph object."""
    if expr == True or isinstance(expr, BooleanTrue):  # noqa: E712
        first_var = VarNode(sorted_vars_names[0])
        return MultiOrNode([first_var, NotNode(first_var)])  # Tautology
    if expr == False or isinstance(expr, BooleanFalse):  # noqa: E712
        return FalseNode()

    if isinstance(expr, Symbol):
        return VarNode(str(expr))

    if isinstance(expr, Not):
        return NotNode(sympy_to_graph(expr.args[0], sorted_vars_names))

    if isinstance(expr, And):
        operands = [sympy_to_graph(arg, sorted_vars_names) for arg in expr.args]
        return MultiAndNode(operands)

    if isinstance(expr, Or):
        operands = [sympy_to_graph(arg, sorted_vars_names) for arg in expr.args]
        return MultiOrNode(operands)

    raise ValueError(f"Unknown SymPy expression type: {type(expr)}")


@dataclass(frozen=True)
class Clause:
    pos: frozenset[str]
    neg: frozenset[str]

    def __repr__(self) -> str:
        pos_str = " ∧ ".join(sorted(self.pos)) if self.pos else "True"
        neg_str = " ∧ ".join(f"¬{v}" for v in sorted(self.neg)) if self.neg else "True"
        if self.pos and self.neg:
            return f"({pos_str} ∧ {neg_str})"
        elif self.pos:
            return f"({pos_str})"
        else:
            return f"({neg_str})"


@functools.cache
def formula_to_clauses(formula: BooleanNode | None) -> list[Clause]:
    """Converts a given Boolean formula in DNF into a list of clauses.

    Args:
        formula: The Boolean formula (in DNF) as a Node.

    Returns:
        A list of Clause objects representing the DNF clauses.
    """
    if formula is None:
        return []

    if isinstance(formula, FalseNode):
        return []

    if isinstance(formula, VarNode):
        return [Clause(pos=frozenset({formula.name}), neg=frozenset())]

    if isinstance(formula, NotNode) and isinstance(formula.operand, VarNode):
        return [Clause(pos=frozenset(), neg=frozenset({formula.operand.name}))]

    if isinstance(formula, MultiOrNode):
        clauses = []
        for operand in formula.operands:
            operand_clauses = formula_to_clauses(operand)
            if len(operand_clauses) != 1:
                raise ValueError(
                    "Each operand in OR node must correspond to a single clause."
                )
            clauses.extend(operand_clauses)
        return clauses

    if isinstance(formula, MultiAndNode):
        pos = set()
        neg = set()
        for operand in formula.operands:
            if isinstance(operand, VarNode):
                pos.add(operand.name)
            elif isinstance(operand, NotNode) and isinstance(operand.operand, VarNode):
                neg.add(operand.operand.name)
            else:
                raise ValueError("Invalid operand in AND node for DNF conversion.")
        return [Clause(pos=frozenset(pos), neg=frozenset(neg))]

    raise ValueError("Formula must be in Disjunctive Normal Form (DNF).")


def push_down_nots(node: BooleanNode) -> BooleanNode:
    """Pushes NOT operators down to the variable level using De Morgan's laws."""
    if isinstance(node, NotNode):
        operand = node.operand
        if isinstance(operand, NotNode):
            return push_down_nots(operand.operand)
        elif isinstance(operand, MultiAndNode):
            return MultiOrNode([push_down_nots(NotNode(op)) for op in operand.operands])
        elif isinstance(operand, MultiOrNode):
            return MultiAndNode(
                [push_down_nots(NotNode(op)) for op in operand.operands]
            )
        elif isinstance(operand, VarNode):
            return node  # Already at variable level
        else:
            raise ValueError("Unsupported node type for NOT push down.")
    elif isinstance(node, MultiAndNode):
        return MultiAndNode([push_down_nots(op) for op in node.operands])
    elif isinstance(node, MultiOrNode):
        return MultiOrNode([push_down_nots(op) for op in node.operands])
    else:
        return node  # VarNode or other nodes remain unchanged


if __name__ == "__main__":
    # test push down nots
    a = VarNode("a")
    b = VarNode("b")
    c = VarNode("c")
    formula = NotNode(MultiAndNode([a, MultiOrNode([b, NotNode(c)])]))
    print("Original formula:", formula)
    pushed = push_down_nots(formula)
    print("After pushing down NOTs:", pushed)
    # Expected: ¬a ∨ (¬b ∧ c)
