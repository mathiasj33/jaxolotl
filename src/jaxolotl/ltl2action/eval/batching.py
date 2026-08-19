from typing import override

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl2action.curriculum.curriculum import SampleBatcher
from jaxolotl.ltl2action.utils.jax_formula_closure import JaxFormulaClosureGraph
from jaxolotl.ltl2action.utils.preprocessing import preprocess_formulas


class FormulaClosureBatcher(SampleBatcher[str, JaxFormulaClosureGraph]):
    """Batches reach-avoid sequences into a JaxReachAvoidSequence."""

    @override
    @staticmethod
    def batch(
        samples: list[str],
        env: Environment | EnvWrapper,
    ) -> JaxFormulaClosureGraph:
        return preprocess_formulas(samples, env, verbose=True)
