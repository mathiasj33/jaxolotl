from typing import override

import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl.alg.curriculum import SampleBatcher
from jaxolotl.alg.ltl2action.utils.jax_formula_closure import (
    JaxFormulaClosureGraph,
    JaxFormulaGraph,
    StaticGraphTable,
)
from jaxolotl.alg.ltl2action.utils.preprocessing import preprocess_formulas
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper

# Above this many unique graphs, encoding the whole table once per forward
# pass would rival encoding each observation's graph individually, so the
# materialized per-state representation is kept instead.
MAX_UNIQUE_GRAPHS = 1024


class FormulaClosureBatcher(SampleBatcher[str, JaxFormulaClosureGraph]):
    """Batches formulas into a JaxFormulaClosureGraph.

    Curriculum samples repeat few distinct formulas, so each closure is built
    once and shared across duplicates. When the number of unique graphs across
    all closure states is small, graphs are additionally deduplicated into a
    static table with per-state indices, which lets the model encode each
    unique graph once per forward pass instead of once per observation.
    """

    @override
    @staticmethod
    def batch(
        samples: list[str],
        env: Environment | EnvWrapper,
        num_parallel: int = 1,
    ) -> JaxFormulaClosureGraph:
        unique_formulas = list(dict.fromkeys(samples))
        unique = preprocess_formulas(
            unique_formulas, env, verbose=True, num_parallel=num_parallel
        )
        formula_to_index = {f: i for i, f in enumerate(unique_formulas)}
        inverse = np.array([formula_to_index[f] for f in samples], dtype=np.int32)

        deduped = _dedup_graphs(unique.graphs)  # type: ignore
        if deduped is None:
            # Expand the materialized representation to one closure per sample.
            return jax.tree.map(lambda x: jnp.asarray(np.asarray(x)[inverse]), unique)

        graph_indices, graph_table = deduped
        expand = lambda x: jnp.asarray(np.asarray(x)[inverse])  # noqa: E731
        return JaxFormulaClosureGraph(
            num_states=expand(unique.num_states),
            initial_state=expand(unique.initial_state),
            true_state=expand(unique.true_state),
            false_state=expand(unique.false_state),
            transitions=expand(unique.transitions),
            graphs=None,
            graph_indices=jnp.asarray(graph_indices[inverse]),
            graph_table=graph_table,
        )


def _dedup_graphs(
    graphs: JaxFormulaGraph,
) -> tuple[np.ndarray, StaticGraphTable] | None:
    """Deduplicate batched (num_closures, num_states, ...) graphs.

    Returns per-(closure, state) indices into a table of unique graphs, or
    None when the table would exceed MAX_UNIQUE_GRAPHS.
    """
    graphs = JaxFormulaGraph(*(np.asarray(leaf) for leaf in graphs))  # type: ignore
    num_closures, num_states = graphs.nodes.shape[:2]
    key_to_index: dict[bytes, int] = {}
    members: list[JaxFormulaGraph] = []
    indices = np.zeros((num_closures, num_states), dtype=np.int32)
    for c in range(num_closures):
        for s in range(num_states):
            key = b"".join(
                np.ascontiguousarray(leaf[c, s]).tobytes() for leaf in graphs
            )
            index = key_to_index.setdefault(key, len(members))
            if index == len(members):
                members.append(JaxFormulaGraph(*(leaf[c, s] for leaf in graphs)))
                if len(members) > MAX_UNIQUE_GRAPHS:
                    return None
            indices[c, s] = index
    table = jax.tree.map(lambda *leaves: np.stack(leaves), *members)
    return indices, StaticGraphTable(table)
