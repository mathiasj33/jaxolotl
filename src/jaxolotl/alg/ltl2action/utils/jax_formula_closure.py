from collections.abc import Iterable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from jaxolotl.alg.ltl2action.utils.formula_closure import FormulaClosureGraph
from jaxolotl.alg.ltl2action.utils.formula_processing import replace_implication
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.progression.ltl_parser import (
    AlwaysNode,
    AndNode,
    EventuallyNode,
    FalseNode,
    LTLNode,
    NotNode,
    OrNode,
    TrueNode,
    UntilNode,
    VarNode,
)


class JaxFormulaGraph(NamedTuple):
    """Jax representation of a formula graph."""

    nodes: jax.Array  # shape: (num_nodes,) int32
    edges: jax.Array  # shape: (num_edges, 2) int32
    edge_types: jax.Array  # shape: (num_edges,) int32
    node_mask: jax.Array  # shape: (num_nodes,) bool
    edge_mask: jax.Array  # shape: (num_edges,) bool


class StaticGraphTable:
    """Hashable, immutable table of unique formula graphs (numpy leaves).

    Stored in a *static* Equinox field so the small, constant table travels
    with the pytree structure — through curriculum save/load, environment
    state, and observations — without being traced, batched per sample, or
    serialized as a leaf. Content-based equality keeps treedefs (and JIT
    caches) stable across instances holding identical tables.
    """

    def __init__(self, graphs: JaxFormulaGraph):
        self.graphs = JaxFormulaGraph(*(np.asarray(leaf) for leaf in graphs))
        self._hash = hash(tuple(leaf.tobytes() for leaf in self.graphs))

    @property
    def num_graphs(self) -> int:
        return self.graphs.nodes.shape[0]

    def to_jax(self) -> JaxFormulaGraph:
        """Return the table as a batched JaxFormulaGraph (num_graphs, ...)."""
        return JaxFormulaGraph(*(jnp.asarray(leaf) for leaf in self.graphs))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, StaticGraphTable)
            and self._hash == other._hash
            and all(
                np.array_equal(a, b) for a, b in zip(self.graphs, other.graphs)
            )
        )


class JaxFormulaClosureGraph(eqx.Module):
    """Jax representation of a formula closure graph. If batched, uses -1 for padding.

    Graphs are stored either materialized per state (`graphs`) or deduplicated
    (`graph_indices` into a shared static `graph_table`); exactly one of the
    two representations is present.
    """

    num_states: jax.Array  # int32
    initial_state: jax.Array  # int32
    true_state: jax.Array  # int32
    false_state: jax.Array  # int32, -1 if not present
    transitions: jax.Array  # shape: (num_states, num_assignments) -> int32
    graphs: JaxFormulaGraph | None  # batched formula graphs (num_states, ...)
    graph_indices: jax.Array | None = None  # shape: (num_states,) int32
    graph_table: StaticGraphTable | None = eqx.field(static=True, default=None)

    def get_next_state(
        self, state: jax.Array, assignment_index: jax.Array
    ) -> jax.Array:
        """Get the next state given the current state and assignment.

        Args:
            state: Current state (int32).
            assignment_index: Index of the assignment (int32).

        Returns:
            next_state: Next state (int32)."""
        return self.transitions[state, assignment_index]

    def get_graph(self, state: jax.Array) -> JaxFormulaGraph:
        """Get the formula graph corresponding to a given state.

        Args:
            state: Current state (int32).

        Returns:
            graph: JaxFormulaGraph corresponding to the state.
        """
        assert self.graphs is not None, (
            "get_graph requires materialized graphs; this closure stores "
            "deduplicated graph_indices into a graph_table instead."
        )
        return jax.tree.map(lambda x: x[state], self.graphs)

    @classmethod
    def from_closure_graphs(
        cls,
        closures: list[FormulaClosureGraph],
        env: Environment | EnvWrapper,
    ) -> "JaxFormulaClosureGraph":
        """Convert a list of FormulaClosureGraphs to a batched Jax representation.

        Args:
            closures: The FormulaClosureGraphs to convert.
            env: The environment containing the assignments.
        """

        # Use numpy arrays and convert at the end for efficiency
        initial_states = np.zeros(len(closures), dtype=np.int32)
        num_assignments = len(env.assignments())
        assignment_to_index = {
            assignment: idx for idx, assignment in enumerate(env.assignments())
        }
        max_closure_nodes = max(closure.num_nodes for closure in closures)
        transitions = -np.ones(
            (len(closures), max_closure_nodes, num_assignments), dtype=np.int32
        )
        true_states = -np.ones(len(closures), dtype=np.int32)
        false_states = -np.ones(len(closures), dtype=np.int32)
        max_graph_nodes = 0
        max_graph_edges = 0
        for closure in closures:
            for node in closure.nodes.values():
                max_graph_nodes = max(max_graph_nodes, node.formula.num_nodes)
                max_graph_edges = max(max_graph_edges, node.formula.num_edges)
        graphs: list[JaxFormulaGraph] = []

        for i, closure in tqdm(
            enumerate(closures), desc="Converting closures", total=len(closures)
        ):
            if closure.initial_node is None:
                raise ValueError("FormulaClosureGraph not initialized.")

            assert len(set(closure.formula_graphs)) == closure.num_nodes

            formula_to_index = {
                formula: idx for idx, formula in enumerate(closure.formula_graphs)
            }
            initial_states[i] = formula_to_index[closure.initial_node.formula]

            true_state = -1
            false_state = -1
            for node in closure.nodes.values():
                node_idx = formula_to_index[node.formula]
                for assignment, target_node in node.edges.items():
                    assignment_idx = assignment_to_index[assignment]
                    transitions[i, node_idx, assignment_idx] = formula_to_index[
                        target_node.formula
                    ]
                    if isinstance(node.formula, TrueNode):
                        true_state = node_idx
                    elif isinstance(node.formula, FalseNode):
                        false_state = node_idx

            assert true_state >= 0, "True state not found."
            true_states[i] = true_state
            false_states[i] = false_state

            # Graphs
            current_closure_graphs = [None] * len(formula_to_index)
            for node in closure.nodes.values():
                node_idx = formula_to_index[node.formula]
                jax_graph = cls._graph_to_jax(
                    node.formula,
                    env.propositions,
                    max_graph_nodes,
                    max_graph_edges,
                )
                current_closure_graphs[node_idx] = jax_graph  # type: ignore
            assert all(g is not None for g in current_closure_graphs)

            current_jax_graphs = jax.tree.map(
                lambda *xs: jnp.stack(xs), *current_closure_graphs
            )

            # add padding graphs
            pad_size = max_closure_nodes - len(current_closure_graphs)
            current_jax_graphs = jax.tree.map(
                lambda x, pad_size=pad_size: jnp.pad(
                    x,
                    ((0, pad_size),) + ((0, 0),) * (x.ndim - 1),
                    constant_values=0,
                ),
                current_jax_graphs,
            )
            graphs.append(current_jax_graphs)

        batched_graphs = jax.tree.map(lambda *xs: jnp.stack(xs), *graphs)

        return cls(
            num_states=jnp.array(
                [closure.num_nodes for closure in closures], dtype=jnp.int32
            ),
            initial_state=jnp.array(initial_states, dtype=jnp.int32),
            true_state=jnp.array(true_states, dtype=jnp.int32),
            false_state=jnp.array(false_states, dtype=jnp.int32),
            transitions=jnp.array(transitions),
            graphs=batched_graphs,
        )

    @classmethod
    def _graph_to_jax(
        cls,
        graph: LTLNode,
        propositions: Iterable[str],
        max_nodes: int,
        max_edges: int,
    ) -> JaxFormulaGraph:
        """Convert a single formula graph to its Jax representation.

        Args:
            graph: The root node of the formula graph.
            max_nodes: Optional maximum number of nodes for padding.
            max_edges: Optional maximum number of edges for padding.

        Returns:
            JaxFormulaGraph representation of the formula graph.
        """
        graph = replace_implication(graph)  # ensure no implications

        nodes = []
        edges = []
        edge_types = []

        node_to_type = {
            t: idx
            for idx, t in enumerate(
                [
                    AndNode,
                    OrNode,
                    NotNode,
                    EventuallyNode,
                    AlwaysNode,
                    UntilNode,
                    TrueNode,
                    FalseNode,
                ]
            )
        }
        prop_to_type = {
            p: idx + len(node_to_type) for idx, p in enumerate(sorted(propositions))
        }
        edge_to_type = {
            "unary": 0,
            "binary_left": 1,
            "binary_right": 2,
        }

        def traverse(node: LTLNode) -> int:
            node_idx = len(nodes)
            if isinstance(node, VarNode):
                assert node.name in prop_to_type, f"Unknown proposition: {node.name}"
                nodes.append(prop_to_type[node.name])
            else:
                nodes.append(node_to_type[type(node)])  # type: ignore
            child_indices = []
            for c in node.children:
                child_idx = traverse(c)
                child_indices.append(child_idx)
            for idx in child_indices:
                edges.append((idx, node_idx))  # reverse edge
            if len(child_indices) == 2:
                edge_types.append(edge_to_type["binary_left"])
                edge_types.append(edge_to_type["binary_right"])
            elif len(child_indices) == 1:
                edge_types.append(edge_to_type["unary"])
            return node_idx

        root_idx = traverse(graph)
        assert root_idx == 0
        assert len(edge_types) == len(edges)

        num_nodes = len(nodes)
        num_edges = len(edges)
        if num_nodes > max_nodes:
            raise ValueError(
                f"Graph has {num_nodes} nodes, exceeds max_nodes {max_nodes}."
            )
        if num_edges > max_edges:
            raise ValueError(
                f"Graph has {num_edges} edges, exceeds max_edges {max_edges}."
            )

        # padding
        pad_nodes = max_nodes - num_nodes
        nodes += [-1] * pad_nodes  # Assuming 0 is a valid padding type_id
        pad_edges = max_edges - num_edges
        edges += [(0, 0)] * pad_edges  # Padding with dummy edges
        edge_types += [-1] * pad_edges  # Assuming 0 is a valid padding type_id

        node_mask = jnp.array(
            [1] * num_nodes + [0] * (max_nodes - num_nodes),
            dtype=bool,
        )
        edge_mask = jnp.array(
            [1] * num_edges + [0] * (max_edges - num_edges),
            dtype=bool,
        )
        return JaxFormulaGraph(
            nodes=jnp.array(nodes, dtype=jnp.int32),
            edges=jnp.array(edges, dtype=jnp.int32),
            edge_types=jnp.array(edge_types, dtype=jnp.int32),
            node_mask=node_mask,
            edge_mask=edge_mask,
        )
