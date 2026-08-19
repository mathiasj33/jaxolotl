"""Ablation study model: GRU over tokenized Boolean formulas with attention.

This model replaces the structured clause/disjunct embeddings in StructLTLModel
with a simple GRU applied to a sequence of tokens representing the Boolean formula.
The same GRU is applied independently to reach and avoid formulas, producing
distinct embeddings that are concatenated to form the step representation.
Attention is used to encode the sequence of steps (matching struct_ltl).
"""

from math import prod
from typing import Literal, NamedTuple, override

import distrax
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
from jaxolotl.environments import spaces
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.alibi_attention import ALiBiAttention
from jaxolotl.networks.conv_net import ConvNet
from jaxolotl.networks.gru_cell import GRUCell
from jaxolotl.networks.mlp import MLP
from jaxolotl.networks.positional_attention import PositionalAttention
from jaxolotl.rl.actor.actor import Actor
from jaxolotl.rl.actor_critic import ActorCritic


class TokenizedLTLModel(ActorCritic):
    """Ablation model using GRU over tokenized Boolean formulas.

    Instead of using structured DeepSets embeddings for clauses and disjuncts,
    this model represents formulas as token sequences and processes them with a GRU.
    The same GRU is applied independently to reach and avoid formulas, and the
    resulting embeddings are concatenated to form the step representation.
    Attention is used to encode the sequence of steps (matching struct_ltl).
    """

    env_net: MLP | ConvNet
    token_embedding: nn.Embedding
    token_gru: GRUCell
    attention: ALiBiAttention | PositionalAttention | None
    gru: GRUCell | None
    actor: Actor
    critic: MLP

    _flatten_features: bool
    _sequence_encoder: Literal["gru", "attention"]

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_space: Space,
        num_propositions: int,
        key: jax.Array,
        **kwargs,
    ):
        """Initialize the TokenizedLTLModel.

        Args:
            obs_shape: Shape of the environment observation.
            act_space: Action space of the environment.
            num_propositions: Number of propositions in the environment.
                Used to compute vocabulary size.
            key: Random key for initialization.
            **kwargs: Additional configuration including:
                - env_net: Environment network configuration.
                - sequence: Sequence encoding configuration with:
                    - embedding_dim: Dimension of token embeddings.
                    - token_hidden_size: Hidden size of token-level GRU.
                    - encoder: "attention" or "gru" for sequence encoding.
                    - attention: Attention configuration (if encoder="attention").
                - actor: Actor network configuration.
                - critic: Critic network configuration.
        """
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)

        # Compute vocabulary size from propositions
        dummy_props = [f"p{i}" for i in range(num_propositions)]
        vocab_size = len(Vocabulary.from_propositions(dummy_props))

        # Environment network (same as StructLTLModel)
        is_conv = "ConvNet" in config.env_net._target_
        params = {"obs_shape": obs_shape} if is_conv else {"in_size": prod(obs_shape)}
        self.env_net = hydra.utils.instantiate(config.env_net, **params, key=env_key)
        self._flatten_features = not is_conv

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
        self._sequence_encoder = getattr(config.sequence, "encoder", "attention")
        step_embedding_dim = 2 * token_hidden_size

        key, encoder_key = jax.random.split(key)
        if self._sequence_encoder == "attention":
            self.gru = None
            self.attention = hydra.utils.instantiate(
                config.sequence.attention,
                input_dim=step_embedding_dim,
                key=encoder_key,
            )
            sequence_output_dim = step_embedding_dim
        elif self._sequence_encoder == "gru":
            sequence_hidden_size = getattr(
                config.sequence, "sequence_hidden_size", 2 * token_hidden_size
            )
            self.gru = GRUCell(
                input_size=step_embedding_dim,
                hidden_size=sequence_hidden_size,
                key=encoder_key,
            )
            self.attention = None
            sequence_output_dim = sequence_hidden_size
        else:
            raise ValueError(
                f"Unknown sequence encoder type: {self._sequence_encoder}. "
                f"Supported types are 'gru' and 'attention'."
            )

        # Actor and critic
        actor_key, critic_key = jax.random.split(key)
        joint_dim = self.env_net.output_size + sequence_output_dim
        params = self._get_actor_params_from_space(act_space)
        self.actor = hydra.utils.instantiate(
            config.actor, in_size=joint_dim, **params, key=actor_key
        )
        self.critic = hydra.utils.instantiate(
            config.critic,
            in_size=joint_dim,
            out_size=1,
            final_layer_activation=False,
            key=critic_key,
        )

    @staticmethod
    def _get_actor_params_from_space(act_space: Space) -> dict:
        """Returns a dict of parameters required to instantiate the actor."""
        if isinstance(act_space, spaces.Discrete):
            return {"num_actions": act_space.n}
        elif isinstance(act_space, spaces.Box):
            return {"action_dim": act_space.shape[0]}
        elif isinstance(act_space, spaces.Composite):
            return {
                "continuous_action_dim": act_space.continuous.shape[0],
                "num_discrete_actions": act_space.discrete.n,
            }
        else:
            raise NotImplementedError(
                f"Actor parameters extraction not implemented for space type "
                f"{type(act_space)}"
            )

    @override
    def _get_action(self, features: jax.Array, obs: PyTree) -> distrax.Distribution:
        return self.actor(features, obs.epsilon_mask)

    @override
    def _get_value(self, features: jax.Array) -> jax.Array:
        value = jax.vmap(self.critic)(features)
        return value.squeeze(-1)

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = (
            self.flatten_features(obs.features)
            if self._flatten_features
            else obs.features.features
        )
        x = jax.vmap(self.env_net)(x)
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
        if self._sequence_encoder == "attention":
            return self._encode_with_attention(step_embeddings, seq.depth)
        else:
            return self._encode_with_gru(step_embeddings, seq.depth)

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

    def _encode_with_attention(
        self, step_embeddings: jax.Array, depth: jax.Array
    ) -> jax.Array:
        """Encode the sequence using attention.

        The first step (index 0) is treated as the "current" step and used as query.
        All steps up to and including depth are used as keys/values.

        Args:
            step_embeddings: Sequence embeddings of shape (max_length, step_dim).
            depth: Current sequence depth (1-indexed).

        Returns:
            Encoded sequence representation of shape (step_dim,).
        """
        assert self.attention is not None
        max_length = step_embeddings.shape[0]

        # Current step is the first one (index 0)
        current_step = step_embeddings[0]

        # Create mask for valid sequence positions (up to and including depth)
        mask = jnp.arange(max_length) < depth

        # Apply attention: query = current step, keys/values = all steps
        return self.attention(current_step, step_embeddings, mask)

    def _encode_with_gru(
        self, step_embeddings: jax.Array, depth: jax.Array
    ) -> jax.Array:
        """Encode the sequence using GRU, processing from the end to the current step.

        Args:
            step_embeddings: Sequence embeddings of shape (max_length, step_dim).
            depth: Current sequence depth (1-indexed).

        Returns:
            Encoded sequence representation of shape (hidden_size,).
        """
        assert self.gru is not None
        gru = self.gru
        h0 = jnp.zeros((gru.hidden_size,))
        max_length = step_embeddings.shape[0]

        def gru_step(
            carry: tuple[jax.Array, int], inputs: jax.Array
        ) -> tuple[tuple[jax.Array, int], None]:
            hidden, step = carry
            hidden = jax.lax.cond(
                step <= depth,
                lambda: gru(inputs, hidden),
                lambda: hidden,
            )
            return (hidden, step - 1), None

        (final_hidden, _), _ = jax.lax.scan(
            gru_step, (h0, max_length), step_embeddings, reverse=True, unroll=8
        )
        return final_hidden

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
