from collections.abc import Iterable
from dataclasses import dataclass, field

from jaxolotl.ltl.logic.assignment import Assignment
from jaxolotl.ltl.progression.ltl_parser import LTLNode, parse
from jaxolotl.ltl.progression.progression import progress, simplify


class FormulaClosureGraph:
    """Represents the closure of an LTL formula under progression. Stores the closure
    as a graph."""

    def __init__(self, formula: str):
        self.formula = formula
        self.formula_node = simplify(parse(formula))
        self.nodes: dict[LTLNode, ClosureGraphNode] = {}

    @property
    def initial_node(self) -> "ClosureGraphNode":
        """Returns the initial node of the closure graph."""
        if self.formula_node not in self.nodes:
            raise ValueError("Closure graph has not been built yet.")
        return self.nodes[self.formula_node]

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the closure graph."""
        return len(self.nodes)

    @property
    def formula_graphs(self) -> list[LTLNode]:
        """Returns the list of formula nodes in the closure graph."""
        return [node.formula for node in self.nodes.values()]

    def build(self, assignments: Iterable[Assignment]):
        """Builds the closure graph for the formula under the given assignments.

        Args:
            assignments: The set of assignments to consider for progression.
        """
        initial_node = ClosureGraphNode(formula=self.formula_node)
        self.nodes[self.formula_node] = initial_node
        to_process = [initial_node]

        while to_process:
            current = to_process.pop()
            for assignment in assignments:
                progressed_formula = progress(current.formula, assignment)
                progressed_formula = simplify(progressed_formula)
                if progressed_formula not in self.nodes:
                    progressed_node = ClosureGraphNode(formula=progressed_formula)
                    self.nodes[progressed_formula] = progressed_node
                    to_process.append(progressed_node)
                current.edges[assignment] = self.nodes[progressed_formula]


@dataclass
class ClosureGraphNode:
    """Represents a node in the formula closure graph."""

    formula: LTLNode
    edges: dict[Assignment, "ClosureGraphNode"] = field(default_factory=dict)


if __name__ == "__main__":
    formula = "!a U (b & (!c U d))"
    assignments = {
        Assignment(frozenset({"a"})),
        Assignment(frozenset({"b"})),
        Assignment(frozenset({"c"})),
        Assignment(frozenset({"d"})),
    }
    closure_graph = FormulaClosureGraph(formula)
    closure_graph.build(assignments)
    print(f"Closure graph for formula: {formula}")
    for node in closure_graph.nodes.values():
        print(f"Node: {node.formula}")
        for assignment, target_node in node.edges.items():
            print(f"  --[{assignment}]--> {target_node.formula}")
