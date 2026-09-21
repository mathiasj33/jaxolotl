from jaxolotl.alg.common.reach_avoid import preprocessing
from jaxolotl.alg.gcrl_ltl.reach_avoid.jax_gcrl_sequence import JaxGCRLSequence
from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.eqx_utils.batching import pad_and_stack
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA


def preprocess_formulas(
    formulas: list[str], env: Environment | EnvWrapper
) -> tuple[JaxLDBA, JaxGCRLSequence]:
    return preprocessing.preprocess_formulas(
        formulas,
        env,
        JaxGCRLSequence.from_state_to_seqs,
        _batch_sequences,
        transform_sequences=preprocessing.to_boolean_sequences,
    )


def _batch_sequences(
    seqs: list[JaxGCRLSequence],
) -> JaxGCRLSequence:
    return pad_and_stack(seqs, JaxGCRLSequence.padding_values())
