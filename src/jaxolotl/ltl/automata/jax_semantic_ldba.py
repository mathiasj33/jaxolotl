import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from jaxolotl.environments.environment import Environment
from jaxolotl.environments.wrappers.wrapper import EnvWrapper
from jaxolotl.ltl.automata.jax_ldba import JaxLDBA
from jaxolotl.ltl.automata.ldba import LDBA


class JaxSemanticLDBA(JaxLDBA):
    """Jax representation of an LDBA with semantic embeddings (as used in SemLTL).

    In contrast to a standard JaxLDBA, this class supports multiple epsilon transitions
    per state.
    """

    # transitions has shape (num_states, num_assignments) -> int32
    epsilon_transitions: jax.Array  # shape: (num_states, max_eps_transitions) -> int32
    embeddings: jax.Array  # shape: (num_states, embedding_dim), float32

    def get_embedding(self, state: jax.Array) -> jax.Array:
        """Get the semantic embedding for the given LDBA state.

        Args:
            state: LDBA state (int32).

        Returns:
            embedding: Semantic embedding of the state (float32)."""
        return self.embeddings[state]

    def get_next_epsilon_state(
        self, state: jax.Array, epsilon_index: jax.Array
    ) -> jax.Array:
        """Get the next LDBA state given the current state for the epsilon transition
        with the given index.
        Returns the same state if no epsilon transition exists.

        Args:
            state: Current LDBA state (int32).
            epsilon_index: Index of the epsilon transition (int32).

        Returns:
            next_state: Next LDBA state (int32)."""
        next_state = self.epsilon_transitions[state, epsilon_index]
        next_state = jnp.where(
            (epsilon_index < self.epsilon_transitions.shape[1]) & (next_state >= 0),
            next_state,
            state,
        )
        return next_state

    def get_next_epsilon_states(self, state: jax.Array) -> jax.Array:
        """Get all next LDBA states given the current state for all epsilon transitions.
        Returns -1 for non-existing epsilon transitions."""
        return self.epsilon_transitions[state]  # (max_eps_transitions,)

    def get_epsilon_embeddings(self, state: jax.Array) -> jax.Array:
        """Get the semantic embeddings for all epsilon transitions from the given LDBA state.

        Args:
            state: LDBA state (int32).

        Returns:
            epsilon_embeddings: Semantic embeddings of the epsilon transitions
                (max_eps_transitions, embedding_dim)."""
        epsilon_states = self.get_next_epsilon_states(state)  # (max_eps_transitions,)
        epsilon_embeddings = jnp.where(
            epsilon_states[:, None] >= 0,
            self.embeddings[epsilon_states],
            jnp.zeros_like(self.embeddings[0]),
        )
        return epsilon_embeddings  # (max_eps_transitions, embedding_dim)

    @classmethod
    def from_ldbas(  # noqa: PLR0912
        cls,
        ldbas: list[LDBA],
        env: Environment | EnvWrapper,
    ) -> "JaxSemanticLDBA":
        """Convert a list of LDBAs to a batched JaxSemanticLDBA.

        Args:
            ldbas: The LDBAs to convert.
            env: The environment containing the assignments.
        """

        if any(ldba.initial_state is None for ldba in ldbas):
            raise ValueError("One or more LDBAs not initialized.")

        # Determine maximum sizes for padding
        max_num_states = 0
        max_eps_transitions = 1  # at least one to avoid zero-size arrays
        max_embedding_dim = 0
        for ldba in ldbas:
            max_num_states = max(max_num_states, ldba.num_states)
            for state in ldba.states:
                eps_transitions = [
                    t for t in ldba.state_to_transitions[state] if t.is_epsilon()
                ]
                max_eps_transitions = max(max_eps_transitions, len(eps_transitions))
            for info in ldba.state_to_info.values():
                embedding_dim = info["embedding"].shape[0]
                max_embedding_dim = max(max_embedding_dim, embedding_dim)

        # Use numpy arrays and convert at the end for efficiency
        num_states = np.array([ldba.num_states for ldba in ldbas], dtype=np.int32)
        initial_states = np.array(
            [ldba.initial_state for ldba in ldbas], dtype=np.int32
        )
        finite = np.array(
            [ldba.is_finite_specification() for ldba in ldbas], dtype=bool
        )
        batch_size = len(ldbas)
        num_assignments = len(env.assignments())
        embedding_dtype = next(
            info["embedding"].dtype
            for ldba in ldbas
            for info in ldba.state_to_info.values()
        )

        # Allocate arrays
        transitions = -np.ones(
            (batch_size, max_num_states, num_assignments), dtype=np.int32
        )
        eps_transitions = -np.ones(
            (batch_size, max_num_states, max_eps_transitions), dtype=np.int32
        )
        accepting = np.zeros((batch_size, max_num_states, num_assignments), dtype=bool)
        sink_states = np.zeros((batch_size, max_num_states), dtype=bool)
        embeddings = -np.ones(
            (batch_size, max_num_states, max_embedding_dim), dtype=embedding_dtype
        )

        assignment_to_idx = {a: i for i, a in enumerate(env.assignments())}

        # Fill arrays
        for i, ldba in tqdm(
            enumerate(ldbas), desc="Processing LDBAs", total=len(ldbas)
        ):
            for state in range(ldba.num_states):
                scc = ldba.state_to_scc[state]
                if scc.bottom and not scc.accepting:
                    sink_states[i, state] = True
                if state in ldba.state_to_info:
                    # otherwise it's a sink state - embedding remains -1s
                    embedding = ldba.state_to_info[state]["embedding"]
                    embeddings[i, state, : embedding.shape[0]] = embedding
                eps_index = 0
                for eps_index, target in enumerate(
                    ldba.get_ordered_epsilon_transitions(state)
                ):
                    assert eps_index < max_eps_transitions, (
                        "Exceeded max epsilon transitions."
                    )
                    eps_transitions[i, state, eps_index] = target
                for transition in ldba.state_to_transitions[state]:
                    if transition.is_epsilon():
                        continue
                    for assignment in transition.valid_assignments:
                        index = assignment_to_idx[assignment]
                        transitions[i, state, index] = transition.target
                        if transition.accepting:
                            accepting[i, state, index] = True

            assert np.all(transitions[i, : ldba.num_states] >= 0), (
                "Incomplete LDBA transitions."
            )
        return cls(
            num_states=jnp.array(num_states),
            initial_state=jnp.array(initial_states),
            transitions=jnp.array(transitions),
            epsilon_transitions=jnp.array(eps_transitions),
            accepting=jnp.array(accepting),
            finite=jnp.array(finite),
            sink_states=jnp.array(sink_states),
            embeddings=jnp.array(embeddings),
        )
