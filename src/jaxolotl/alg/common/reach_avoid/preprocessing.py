"""Shared preprocessing pipeline for automata-based reach-avoid algorithms."""

from collections.abc import Callable, Sequence
from typing import cast

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA
from jaxolotl.ltl.automata.preprocessing import batch_ldbas, build_ldba
from jaxolotl.ltl.reach_avoid import path_search
from jaxolotl.ltl.reach_avoid.sequence import ReachAvoidSequence

type EnvironmentLike = Environment | EnvWrapper
type StateSequences[TSequence] = dict[int, list[TSequence]]
type SequenceTransform[TSequence] = Callable[
    [dict[int, list[ReachAvoidSequence]], EnvironmentLike],
    dict[int, list[TSequence]],
]
type StateEncoder[TSequence, TEncoded] = Callable[
    [StateSequences[TSequence], EnvironmentLike], TEncoded
]


def preprocess_formula[TSequence, TEncoded](
    formula: str,
    env: EnvironmentLike,
    encode_states: StateEncoder[TSequence, TEncoded],
    *,
    transform_sequences: SequenceTransform[TSequence] | None = None,
    num_loops: int = 2,
) -> tuple[JaxLDBA, TEncoded]:
    """Compile one formula and encode its reach-avoid choices by LDBA state."""
    ldba = build_ldba(formula, env)
    jax_ldba = JaxLDBA.from_ldba(ldba, env)
    state_to_sequences = path_search.compute_sequences(ldba, num_loops=num_loops)
    if transform_sequences is not None:
        transformed = transform_sequences(state_to_sequences, env)
    else:
        transformed = cast(StateSequences[TSequence], state_to_sequences)
    return jax_ldba, encode_states(transformed, env)


def preprocess_formulas[TSequence, TEncoded](
    formulas: Sequence[str],
    env: EnvironmentLike,
    encode_states: StateEncoder[TSequence, TEncoded],
    batch_encoded: Callable[[list[TEncoded]], TEncoded],
    *,
    transform_sequences: SequenceTransform[TSequence] | None = None,
    num_loops: int = 2,
) -> tuple[JaxLDBA, TEncoded]:
    """Compile and batch formulas using algorithm-specific sequence encoding."""
    if not formulas:
        raise ValueError("Cannot preprocess an empty formula batch.")

    ldbas: list[JaxLDBA] = []
    encoded: list[TEncoded] = []
    for formula in formulas:
        ldba, state_sequences = preprocess_formula(
            formula,
            env,
            encode_states,
            transform_sequences=transform_sequences,
            num_loops=num_loops,
        )
        ldbas.append(ldba)
        encoded.append(state_sequences)
    return batch_ldbas(ldbas), batch_encoded(encoded)
