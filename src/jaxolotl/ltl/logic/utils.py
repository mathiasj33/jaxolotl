import functools
import itertools
import os
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

_SYNTHESIS_BACKEND = (
    os.environ.get("JAXOLOTL_SYNTHESIS_BACKEND", "prime_implicant").strip().lower()
)
if _SYNTHESIS_BACKEND not in ("prime_implicant", "sopform"):
    raise ValueError(
        "JAXOLOTL_SYNTHESIS_BACKEND must be 'prime_implicant' or 'sopform', "
        f"got {_SYNTHESIS_BACKEND!r}"
    )

BitPattern = tuple[int, ...]
Cube = tuple[int | None, ...]


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
    Generates a minimum DNF that is true for 'target_assignments'
    and false for the rest of 'possible_assignments'.

    Any assignment NOT in 'possible_assignments' is treated as a 'don't care',
    allowing the solver to simplify the logic further.

    ``JAXOLOTL_SYNTHESIS_BACKEND`` selects the backend at import time:

    - ``prime_implicant`` (default) enumerates prime implicants from the ON/OFF
      sets and solves an exact cover without materializing the full truth table.
    - ``sopform`` uses SymPy's Quine-McCluskey implementation over the full truth
      table.

    The prime-implicant backend minimizes clause count, then literal count, with a
    deterministic tie-break. Its worst case is still exponential in the number of
    prime implicants, but sparse structured assignment universes avoid the up-front
    ``2^len(props)`` cost of ``sopform``.
    """
    if _SYNTHESIS_BACKEND == "sopform":
        return _synthesize_sopform(target_assignments, possible_assignments, props)
    return _synthesize_prime_implicant(target_assignments, possible_assignments, props)


def _validate_synthesis_inputs(
    target_assignments: frozenset[Assignment],
    possible_assignments: frozenset[Assignment],
    props: tuple[str, ...],
) -> tuple[tuple[str, ...], set[Assignment], set[Assignment]]:
    if not props:
        raise ValueError("No propositions provided for formula synthesis.")
    if len(set(props)) != len(props):
        raise ValueError("Formula-synthesis propositions must be unique.")

    targets = set(target_assignments)
    possible = set(possible_assignments)
    if not targets <= possible:
        raise ValueError("Target assignments must be a subset of possible assignments.")

    all_props = set(props)
    unknown_props = {
        proposition
        for assignment in possible
        for proposition in assignment
        if proposition not in all_props
    }
    if unknown_props:
        raise ValueError(
            "Assignments contain propositions absent from props: "
            f"{sorted(unknown_props)!r}"
        )

    return tuple(sorted(all_props)), targets, possible


def _synthesize_sopform(
    target_assignments: frozenset[Assignment],
    possible_assignments: frozenset[Assignment],
    props: tuple[str, ...],
) -> BooleanNode:
    """Synthesize DNF via SymPy's full-truth-table Quine-McCluskey solver."""
    sorted_vars, targets, possible = _validate_synthesis_inputs(
        target_assignments, possible_assignments, props
    )

    sympy_vars = [Symbol(v) for v in sorted_vars]

    def to_bit_pattern(assignment: "Assignment") -> BitPattern:
        return tuple(1 if v in assignment else 0 for v in sorted_vars)

    target_patterns = {to_bit_pattern(a) for a in targets}
    universe_patterns = {to_bit_pattern(a) for a in possible}

    minterms = []
    dontcares = []

    for pattern in itertools.product([0, 1], repeat=len(sorted_vars)):
        if pattern in target_patterns:
            minterms.append(pattern)
        elif pattern not in universe_patterns:
            dontcares.append(pattern)

    expr = SOPform(sympy_vars, minterms, dontcares=dontcares)
    return sympy_to_graph(expr, list(sorted_vars))


def _minimal_hitting_sets(
    family: tuple[frozenset[int], ...],
) -> tuple[frozenset[int], ...]:
    found: set[frozenset[int]] = set()

    def search(partial: frozenset[int], remaining: tuple[frozenset[int], ...]) -> None:
        if any(result <= partial for result in found):
            return
        if not remaining:
            found.add(partial)
            return

        smallest = min(remaining, key=lambda item: (len(item), tuple(item)))
        for element in sorted(smallest):
            search(
                partial | {element},
                tuple(item for item in remaining if element not in item),
            )

    search(frozenset(), family)
    minimal = (item for item in found if not any(other < item for other in found))
    return tuple(sorted(minimal, key=lambda item: (len(item), tuple(item))))


def _cube_key(cube: Cube) -> tuple[int, ...]:
    return tuple(-1 if value is None else value for value in cube)


def _cube_covers(cube: Cube, minterm: BitPattern) -> bool:
    return all(
        value is None or value == minterm[index] for index, value in enumerate(cube)
    )


def _minimum_cover(
    on: set[BitPattern], prime_implicants: set[Cube]
) -> tuple[Cube, ...]:
    ordered_implicants = tuple(sorted(prime_implicants, key=_cube_key))
    coverage = {
        implicant: frozenset(
            minterm for minterm in on if _cube_covers(implicant, minterm)
        )
        for implicant in ordered_implicants
    }
    best_selection: tuple[Cube, ...] | None = None
    best_cost: tuple[int, int, tuple[tuple[int, ...], ...]] | None = None

    def search(remaining: frozenset[BitPattern], chosen: tuple[Cube, ...]) -> None:
        nonlocal best_cost, best_selection

        if not remaining:
            selection = tuple(sorted(chosen, key=_cube_key))
            cost = (
                len(selection),
                sum(sum(value is not None for value in cube) for cube in selection),
                tuple(_cube_key(cube) for cube in selection),
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_selection = selection
            return

        if best_cost is not None and len(chosen) >= best_cost[0]:
            return

        minterm = min(
            remaining,
            key=lambda item: (
                sum(item in coverage[implicant] for implicant in ordered_implicants),
                item,
            ),
        )
        candidates = sorted(
            (
                implicant
                for implicant in ordered_implicants
                if minterm in coverage[implicant]
            ),
            key=lambda implicant: (
                -len(coverage[implicant] & remaining),
                sum(value is not None for value in implicant),
                _cube_key(implicant),
            ),
        )
        for implicant in candidates:
            search(remaining - coverage[implicant], (*chosen, implicant))

    search(frozenset(on), ())
    if best_selection is None:
        raise RuntimeError("Failed to cover all target assignments.")
    return best_selection


def _synthesize_prime_implicant(
    target_assignments: frozenset[Assignment],
    possible_assignments: frozenset[Assignment],
    props: tuple[str, ...],
) -> BooleanNode:
    """Synthesize an exact minimum DNF from explicit ON/OFF assignments.

    A prime implicant through an ON minterm is a minimal hitting set of the
    literal families that distinguish it from every OFF point. After enumerating
    those implicants, an exact unate-cover search minimizes clauses, then literals.
    """
    sorted_vars, targets, possible = _validate_synthesis_inputs(
        target_assignments, possible_assignments, props
    )
    num_vars = len(sorted_vars)

    def bits(assignment: "Assignment") -> BitPattern:
        return tuple(1 if v in assignment else 0 for v in sorted_vars)

    on = {bits(assignment) for assignment in targets}
    off = {bits(assignment) for assignment in possible if assignment not in targets}

    if not on:
        return FalseNode()
    if not off:
        first_var = VarNode(sorted_vars[0])
        return MultiOrNode([first_var, NotNode(first_var)])

    prime_implicants: set[Cube] = set()
    sorted_off = tuple(sorted(off))
    for minterm in sorted(on):
        family = tuple(
            frozenset(
                index for index in range(num_vars) if minterm[index] != off_point[index]
            )
            for off_point in sorted_off
        )
        for hitting_set in _minimal_hitting_sets(family):
            prime_implicants.add(
                tuple(
                    minterm[index] if index in hitting_set else None
                    for index in range(num_vars)
                )
            )

    best_selection = _minimum_cover(on, prime_implicants)

    def cube_to_node(cube: Cube) -> BooleanNode:
        literals: list[BooleanNode] = []
        for index, value in enumerate(cube):
            if value is None:
                continue
            variable = VarNode(sorted_vars[index])
            literals.append(variable if value == 1 else NotNode(variable))
        if len(literals) == 1:
            return literals[0]
        return MultiAndNode(literals)

    nodes = [cube_to_node(cube) for cube in best_selection]
    if len(nodes) == 1:
        return nodes[0]
    return MultiOrNode(nodes)


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

    def __len__(self) -> int:
        return len(self.pos) + len(self.neg)


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


def push_down_nots(node: BooleanNode) -> BooleanNode:  # noqa: PLR0911
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
