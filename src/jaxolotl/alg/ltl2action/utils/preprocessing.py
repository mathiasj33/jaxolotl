"""Utility functions for batching formulas into closure graphs."""

import jax
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


if __name__ == "__main__":
    formulas = ["F green", "F (green & F red)", "!yellow U purple"]
    env, _ = jaxolotl.make("ZoneEnv")
    batched = preprocess_formulas(formulas, env)
    for arr in jax.tree.leaves(batched):
        print(arr)
