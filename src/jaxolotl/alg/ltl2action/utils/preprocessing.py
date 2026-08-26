"""Utility functions for batching formulas into closure graphs."""

from typing import cast

import jax
from joblib import Parallel, delayed
from tqdm import tqdm

import jaxolotl
from jaxolotl.alg.ltl2action.utils.formula_closure import FormulaClosureGraph
from jaxolotl.alg.ltl2action.utils.jax_formula_closure import JaxFormulaClosureGraph
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.logic.assignment import Assignment


def preprocess_formulas(
    formulas: list[str],
    env: Environment | EnvWrapper,
    verbose: bool = False,
    num_parallel: int = 1,
) -> JaxFormulaClosureGraph:
    """Converts a list of LTL formulas into a batched JaxFormulaClosureGraph.

    Args:
        formulas: A list of formulas.
        env: The environment whose assignments define closure transitions.
        verbose: Whether to show a progress bar while dispatching formulas.
        num_parallel: Number of formula closures to construct in parallel.

    Returns:
        A batched JaxFormulaClosureGraph.
    """

    if num_parallel < 1:
        raise ValueError("num_parallel must be at least 1")
    it = tqdm(formulas, desc="Computing closures") if verbose else formulas
    assignments = tuple(env.assignments())
    closures = Parallel(n_jobs=num_parallel)(
        delayed(_build_formula_closure)(formula, assignments) for formula in it
    )
    closures = cast(list[FormulaClosureGraph], closures)
    return JaxFormulaClosureGraph.from_closure_graphs(closures, env)


def _build_formula_closure(
    formula: str, assignments: tuple[Assignment, ...]
) -> FormulaClosureGraph:
    closure = FormulaClosureGraph(formula)
    closure.build(assignments)
    return closure


if __name__ == "__main__":
    formulas = ["F green", "F (green & F red)", "!yellow U purple"]
    env, _ = jaxolotl.make("ZoneEnv")
    batched = preprocess_formulas(formulas, env)
    for arr in jax.tree.leaves(batched):
        print(arr)
