from math import prod
from typing import Literal, NamedTuple, override

import distrax
import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.alg.struct_ltl.reach_avoid.jax_clause_reach_avoid_sequence import (
    JaxClauseReachAvoidSequence,
)
from jaxolotl.environments import spaces
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.alibi_attention import ALiBiAttention
from jaxolotl.networks.conv_net import ConvNet
from jaxolotl.networks.deep_sets import DeepSets
from jaxolotl.networks.gru_cell import GRUCell
from jaxolotl.networks.mlp import MLP
from jaxolotl.networks.positional_attention import PositionalAttention
from jaxolotl.rl.actor.actor import Actor
from jaxolotl.rl.actor_critic import ActorCritic


class StructLTLModel(ActorCritic):
    env_net: MLP | ConvNet
    embedding: nn.Embedding
    neg_linear: nn.Linear
    clause_mlp: DeepSets
    disjunct_mlp: DeepSets
    gru: GRUCell | None
    attention: ALiBiAttention | PositionalAttention | None
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
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)
        is_conv = "ConvNet" in config.env_net._target_
        params = {"obs_shape": obs_shape} if is_conv else {"in_size": prod(obs_shape)}
        self.env_net = hydra.utils.instantiate(config.env_net, **params, key=env_key)
        self._flatten_features = not is_conv
        key, emb_key = jax.random.split(key)
        embedding_dim = config.sequence.embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_propositions + 1,  # +1 for epsilon transitions
            embedding_size=embedding_dim,
            key=emb_key,
        )
        key, neg_key = jax.random.split(key)
        self.neg_linear = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            key=neg_key,
        )
        key, clause_key = jax.random.split(key)
        self.clause_mlp = hydra.utils.instantiate(
            config.sequence.clause_mlp,
            embedding_dim=embedding_dim,
            key=clause_key,
        )
        key, disjunct_key = jax.random.split(key)
        self.disjunct_mlp = hydra.utils.instantiate(
            config.sequence.disjunct_mlp,
            embedding_dim=embedding_dim,
            key=disjunct_key,
        )

        # Determine sequence encoder type (default to GRU for backward compatibility)
        self._sequence_encoder = getattr(config.sequence, "encoder", "gru")
        sequence_dim = 2 * config.sequence.disjunct_mlp.out_size

        key, encoder_key = jax.random.split(key)
        if self._sequence_encoder == "attention":
            self.gru = None
            self.attention = hydra.utils.instantiate(
                config.sequence.attention,
                input_dim=sequence_dim,
                key=encoder_key,
            )
        elif self._sequence_encoder == "gru":
            self.gru = GRUCell(
                input_size=sequence_dim,
                hidden_size=2 * embedding_dim,
                key=encoder_key,
            )
            self.attention = None
        else:
            raise ValueError(
                f"Unknown sequence encoder type: {self._sequence_encoder}. "
                f"Supported types are 'gru' and 'attention'."
            )

        actor_key, critic_key = jax.random.split(key)
        joint_dim = self.env_net.output_size + 2 * config.sequence.embedding_dim
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
        """Returns a dict of parameters required to instantiate the actor based on the
        action space."""
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
        self, seq: JaxClauseReachAvoidSequence
    ) -> jax.Array:
        def embed_clause(indices: jax.Array, neg_mask: jax.Array) -> jax.Array:
            # indices shape: (num_propositions,)
            mask = indices != -1
            embeddings = jax.vmap(self.embedding)(indices * mask)
            # embeddings shape: (num_propositions, embedding_dim)
            neg_embeddings = jax.vmap(self.neg_linear)(embeddings)
            embeddings = jnp.where(neg_mask[:, None], neg_embeddings, embeddings)
            embeddings = embeddings * mask[:, None]  # zero out padding embeddings
            return self.clause_mlp(embeddings)  # shape: (out_size,)

        def embed_disjunct(
            clauses: jax.Array, neg_masks: jax.Array, num_clauses: jax.Array
        ) -> jax.Array:
            # clauses shape: (max_clauses, num_propositions)
            clause_embeddings = jax.vmap(embed_clause)(clauses, neg_masks)
            mask = jnp.arange(clauses.shape[0]) < num_clauses
            clause_embeddings = clause_embeddings * mask[:, None]
            # clause_embeddings shape: (max_clauses, clause_mlp.out_size)
            return self.disjunct_mlp(clause_embeddings)  # shape: (out_size,)

        reach_emb = jax.vmap(embed_clause)(seq.reach_clauses, seq.reach_negatives)
        avoid_emb = jax.vmap(embed_disjunct)(
            seq.avoid_clauses, seq.avoid_negatives, seq.num_avoid_clauses
        )
        reach_avoid = jnp.concatenate([reach_emb, avoid_emb], axis=-1)
        # reach_avoid shape: (max_seq_length, 2 * disjunct_mlp.out_size)

        if self._sequence_encoder == "attention":
            return self._encode_with_attention(reach_avoid, seq.depth)
        else:
            return self._encode_with_gru(reach_avoid, seq.depth)

    def _encode_with_gru(self, reach_avoid: jax.Array, depth: jax.Array) -> jax.Array:
        """Encode the sequence using GRU, processing from the end to the current step.

        Args:
            reach_avoid: Sequence embeddings of shape (max_seq_length, seq_dim).
            depth: Current sequence depth (1-indexed).

        Returns:
            Encoded sequence representation of shape (hidden_size,).
        """
        assert self.gru is not None
        gru = self.gru  # local binding for type narrowing
        h0 = jnp.zeros((gru.hidden_size,))  # initial hidden state

        def gru_step(
            carry: tuple[jax.Array, int], inputs: jax.Array
        ) -> tuple[tuple[jax.Array, int], None]:
            hidden, step = carry
            hidden = jax.lax.cond(
                step <= depth, lambda: gru(inputs, hidden), lambda: hidden
            )
            return (hidden, step - 1), None

        max_seq_length = reach_avoid.shape[0]
        (final_hidden, _), _ = jax.lax.scan(
            gru_step, (h0, max_seq_length), reach_avoid, reverse=True, unroll=8
        )
        return final_hidden

    def _encode_with_attention(
        self, reach_avoid: jax.Array, depth: jax.Array
    ) -> jax.Array:
        """Encode the sequence using ALiBi attention.

        The first step (index 0) is treated as the "current" step and used as the query.
        All steps up to and including depth are used as keys/values.

        Args:
            reach_avoid: Sequence embeddings of shape (max_seq_length, seq_dim).
            depth: Current sequence depth (1-indexed).

        Returns:
            Encoded sequence representation of shape (seq_dim,).
        """
        assert self.attention is not None
        max_seq_length = reach_avoid.shape[0]

        # Current step is the first one (index 0)
        current_step = reach_avoid[0]

        # Create mask for valid sequence positions (up to and including depth)
        # depth is 1-indexed, so valid positions are 0 to depth-1 (inclusive)
        mask = jnp.arange(max_seq_length) < depth

        # Apply attention: query = current step, keys/values = all steps
        return self.attention(current_step, reach_avoid, mask)

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
