"""StructLTL with tokenized formula representation."""

from typing import override

import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.alg.struct_ltl.reach_avoid.formula_tokenizer import Vocabulary
from jaxolotl.alg.struct_ltl.reach_avoid.jax_tokenized_reach_avoid_sequence import (
    JaxTokenizedReachAvoidSequence,
)
from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.gru_cell import GRUCell
from jaxolotl.networks.observation_encoder import ObservationEncoder
from jaxolotl.networks.sequence_encoder import SequenceEncoder
from jaxolotl.rl.actor_critic import ActorCritic


class TokenizedLTLModel(ActorCritic):
    """Ablation model using GRU over tokenized Boolean formulas.

    Instead of using structured DeepSets embeddings for clauses and disjuncts,
    this model represents formulas as token sequences and processes them with a GRU.
    The same GRU is applied independently to reach and avoid formulas, and the
    resulting embeddings are concatenated to form the step representation.
    Attention is used to encode the sequence of steps (matching struct_ltl).
    """

    env_net: ObservationEncoder
    token_embedding: nn.Embedding
    token_gru: GRUCell
    sequence_encoder: SequenceEncoder

    def __init__(
        self,
        obs_spec: ObservationSpec,
        act_space: Space,
        num_propositions: int,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)

        # Compute vocabulary size from propositions
        dummy_props = [f"p{i}" for i in range(num_propositions)]
        vocab_size = len(Vocabulary.from_propositions(dummy_props))

        # Environment network
        self.env_net = hydra.utils.instantiate(
            config.env_net, input_spec=obs_spec, key=env_key, _convert_="object"
        )

        # Token embedding
        key, emb_key = jax.random.split(key)
        embedding_dim = config.sequence.embedding_dim
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=embedding_dim,
            key=emb_key,
        )

        # Token-level GRU: processes tokens within reach or avoid formula
        key, token_gru_key = jax.random.split(key)
        token_hidden_size = config.sequence.token_hidden_size
        self.token_gru = GRUCell(
            input_size=embedding_dim,
            hidden_size=token_hidden_size,
            key=token_gru_key,
        )

        # Sequence-level encoder (attention or GRU)
        # Step embedding dimension = 2 * token_hidden_size (reach + avoid concatenated)
        step_embedding_dim = 2 * token_hidden_size

        key, encoder_key = jax.random.split(key)
        sequence_hidden_size = getattr(
            config.sequence, "sequence_hidden_size", 2 * token_hidden_size
        )
        self.sequence_encoder = SequenceEncoder(
            input_size=step_embedding_dim,
            hidden_size=sequence_hidden_size,
            mode=config.sequence.encoder,
            attention_config=getattr(config.sequence, "attention", None),
            key=encoder_key,
        )

        # Actor and critic
        joint_dim = self.env_net.output_size + self.sequence_encoder.output_size
        super().__init__(config, act_space, joint_dim, key)

    @override
    def _get_actor_context(self, obs: PyTree) -> PyTree:
        return obs.epsilon_mask

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = jax.vmap(self.env_net)(obs.features)
        emb = jax.vmap(self._compute_sequence_embedding)(obs.seq)
        return jnp.concatenate([x, emb], axis=-1)

    def _compute_sequence_embedding(
        self, seq: JaxTokenizedReachAvoidSequence
    ) -> jax.Array:
        """Compute sequence embedding using hierarchical GRU + attention encoding.

        The encoding is hierarchical:
        1. Token-level: Shared GRU processes reach and avoid tokens independently
        2. Step-level: Reach and avoid embeddings are concatenated
        3. Sequence-level: Attention (or GRU) encodes the sequence of steps

        Args:
            seq: A JaxTokenizedReachAvoidSequence containing separate reach/avoid tokens.

        Returns:
            Sequence embedding of shape (sequence_output_dim,).
        """
        # Compute step embeddings: reach and avoid encoded separately, then concatenated
        step_embeddings = self._compute_step_embeddings(seq)
        # step_embeddings shape: (max_length, 2 * token_hidden_size)

        # Encode sequence of steps
        return self.sequence_encoder(step_embeddings, seq.depth)

    def _compute_step_embeddings(
        self, seq: JaxTokenizedReachAvoidSequence
    ) -> jax.Array:
        """Compute step embeddings by encoding reach and avoid separately.

        Args:
            seq: JaxTokenizedReachAvoidSequence with separate reach/avoid tokens.

        Returns:
            Step embeddings of shape (max_length, 2 * token_hidden_size).
        """

        def encode_tokens(tokens: jax.Array) -> jax.Array:
            """Encode a token sequence with the shared GRU.

            Args:
                tokens: Token indices of shape (max_tokens,).

            Returns:
                Final hidden state of shape (token_hidden_size,).
            """
            # Embed tokens
            mask = tokens != -1
            token_embeddings = jax.vmap(self.token_embedding)(tokens) * mask[:, None]
            # token_embeddings shape: (max_tokens, embedding_dim)

            h0 = jnp.zeros((self.token_gru.hidden_size,))

            def gru_step(
                hidden: jax.Array,
                inputs: tuple[jax.Array, jax.Array],
            ) -> tuple[jax.Array, None]:
                token_emb, is_valid = inputs
                new_hidden = jax.lax.cond(
                    is_valid,
                    lambda: self.token_gru(token_emb, hidden),
                    lambda: hidden,
                )
                return new_hidden, None

            final_hidden, _ = jax.lax.scan(
                gru_step, h0, (token_embeddings, mask), unroll=8
            )
            return final_hidden

        # Encode all steps
        reach_embeddings = jax.vmap(encode_tokens)(seq.reach_tokens)
        avoid_embeddings = jax.vmap(encode_tokens)(seq.avoid_tokens)
        # shape: (max_length, token_hidden_size)
        step_embeddings = jnp.concatenate([reach_embeddings, avoid_embeddings], axis=-1)
        # shape: (max_length, 2 * token_hidden_size)
        return step_embeddings
