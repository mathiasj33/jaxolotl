"""Utility functions for batching formulas into closure graphs."""

import jax
import jax.numpy as jnp
from tqdm import tqdm

import jaxolotl
from jaxolotl.alg.ltl2action.utils.formula_closure import FormulaClosureGraph
from jaxolotl.alg.ltl2action.utils.jax_formula_closure import JaxFormulaClosureGraph
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper


def preprocess_formulas(
    formulas: list[str], env: Environment | EnvWrapper, verbose: bool = False
) -> JaxFormulaClosureGraph:
    """Converts a list of LTL formulas into a batched JaxFormulaClosureGraph.

    Args:
        formulas: A list of formulas.

    Returns:
        A batched JaxFormulaClosureGraph.
    """

    closures: list[FormulaClosureGraph] = []
    it = tqdm(formulas, desc="Computing closures") if verbose else formulas
    for formula in it:
        closure = FormulaClosureGraph(formula)
        closure.build(env.assignments())
        closures.append(closure)
    return JaxFormulaClosureGraph.from_closure_graphs(closures, env)


def _pad_jax_closure_graphs(
    closures: list[JaxFormulaClosureGraph],
) -> JaxFormulaClosureGraph:
    """Pads a list of JaxFormulaClosureGraphs into a single batched JaxFormulaClosureGraph.

    Args:
        closures: A list of JaxFormulaClosureGraphs to pad.

    Returns:
        A batched JaxFormulaClosureGraph.
    """
    max_states = max(c.num_states for c in closures)

    def pad(graph: JaxFormulaClosureGraph) -> JaxFormulaClosureGraph:
        pad_size = max_states - graph.num_states
        padded_transitions = jnp.pad(
            graph.transitions,
            ((0, pad_size), (0, 0)),
            constant_values=-1,
        )
        padded_graphs = jax.tree.map(
            lambda x: jnp.pad(
                x,
                ((0, pad_size),) + ((0, 0),) * (x.ndim - 1),
                constant_values=-1,
            ),
            graph.graphs,
        )
        return JaxFormulaClosureGraph(
            num_states=graph.num_states,
            initial_state=graph.initial_state,
            true_state=graph.true_state,
            false_state=graph.false_state,
            transitions=padded_transitions,
            graphs=padded_graphs,
        )

    return jax.tree.map(
        lambda *args: jnp.stack(args),
        *[pad(c) for c in closures],
    )


if __name__ == "__main__":
    formulas = ["F green", "F (green & F red)", "!yellow U purple"]
    env, _ = jaxolotl.make("ZoneEnv")
    batched = preprocess_formulas(formulas, env)
    for arr in jax.tree.leaves(batched):
        print(arr)
