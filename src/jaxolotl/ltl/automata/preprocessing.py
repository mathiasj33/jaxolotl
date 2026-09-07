"""Shared construction and batching helpers for LDBAs."""

import jax.numpy as jnp
import numpy as np

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA
from jaxolotl.ltl.automata.ldba import LDBA
from jaxolotl.ltl.automata.ltl2ldba import LDBABackend, ltl2ldba


def build_ldba(
    formula: str, env: Environment | EnvWrapper, backend: LDBABackend
) -> LDBA:
    """Build, prune, and complete an LDBA from a formula and environment."""
    ldba = ltl2ldba(formula, env.propositions, env.assignments(), backend)
    if backend == "semml":
        process_semantic_embeddings(ldba)
    ldba.prune(env.assignments())
    ldba.eliminate_forced_epsilons()
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


def process_semantic_embeddings(ldba: LDBA) -> None:
    """Concatenates breakpoint and master formula embeddings for each LDBA state."""
    for state in ldba.states:
        info = ldba.state_to_info[state]
        embeddings = info["embeddings"]
        if info["component"] == "initial":
            embedding = embeddings["formula_embedding"]
            embedding += [0.0] * len(embedding)  # empty breakpoint embedding
        else:
            embedding = embeddings["master_formula_embedding"]
            embedding += embeddings["breakpoint_formula_embedding"]
        info["embedding"] = np.array(embedding, dtype=np.float32)


def batch_ldbas(ldbas: list[JaxLDBA]) -> JaxLDBA:
    """Pad and stack JAX LDBAs along a new batch dimension."""
    if not ldbas:
        raise ValueError("Cannot batch an empty list of LDBAs.")

    num_assignments = ldbas[0].transitions.shape[1] - 1
    if any(ldba.transitions.shape[1] != num_assignments + 1 for ldba in ldbas):
        raise ValueError("All LDBAs must use the same assignment space.")

    state_counts = [int(ldba.num_states) for ldba in ldbas]
    max_states = max(state_counts)
    batch_size = len(ldbas)
    transitions = -jnp.ones(
        (batch_size, max_states, num_assignments + 1), dtype=jnp.int32
    )
    accepting = jnp.zeros((batch_size, max_states, num_assignments), dtype=jnp.bool_)
    sink_states = jnp.zeros((batch_size, max_states), dtype=jnp.bool_)
    initial_states = jnp.zeros((batch_size,), dtype=jnp.int32)

    for index, (ldba, num_states) in enumerate(zip(ldbas, state_counts, strict=True)):
        transitions = transitions.at[index, :num_states].set(ldba.transitions)
        accepting = accepting.at[index, :num_states].set(ldba.accepting)
        sink_states = sink_states.at[index, :num_states].set(ldba.sink_states)
        initial_states = initial_states.at[index].set(ldba.initial_state)

    return JaxLDBA(
        num_states=jnp.asarray(state_counts, dtype=jnp.int32),
        initial_state=initial_states,
        transitions=transitions,
        accepting=accepting,
        sink_states=sink_states,
        finite=jnp.asarray([ldba.finite for ldba in ldbas], dtype=jnp.bool_),
    )
