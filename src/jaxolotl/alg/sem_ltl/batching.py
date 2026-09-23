"""Batching/preprocessing utilities for SemLTL."""

from typing import cast, override

from joblib import Parallel, delayed
from tqdm import tqdm

from jaxolotl.alg.curriculum import SampleBatcher
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata.jax_semantic_ldba import JaxSemanticLDBA
from jaxolotl.ltl.automata.ldba import LDBA
from jaxolotl.ltl.automata.preprocessing import build_ldba


def preprocess_formulas(
    formulas: list[str],
    env: Environment | EnvWrapper,
    num_parallel: int = 1,
) -> JaxSemanticLDBA:
    """Build and batch SemML LDBAs, preserving the input formula order."""
    if num_parallel < 1:
        raise ValueError("num_parallel must be at least 1")
    # Curriculum samples repeat few distinct formulas; build each once and share
    # the (read-only) LDBA object across duplicates.
    unique_formulas = list(dict.fromkeys(formulas))
    unique_ldbas = Parallel(n_jobs=num_parallel)(
        delayed(build_ldba)(formula, env, backend="semml")
        for formula in tqdm(unique_formulas, desc="Building SemML LDBAs")
    )
    formula_to_ldba = dict(
        zip(unique_formulas, cast(list[LDBA], unique_ldbas), strict=True)
    )
    ldbas = [formula_to_ldba[formula] for formula in formulas]
    return JaxSemanticLDBA.from_ldbas(ldbas, env)


class SemanticLDBABatcher(SampleBatcher[str, JaxSemanticLDBA]):
    """Batches formulas into a JaxSemanticLDBA."""

    @override
    @staticmethod
    def batch(
        samples: list[str],
        env: Environment | EnvWrapper,
        num_parallel: int = 1,
    ) -> JaxSemanticLDBA:
        return preprocess_formulas(samples, env, num_parallel=num_parallel)
