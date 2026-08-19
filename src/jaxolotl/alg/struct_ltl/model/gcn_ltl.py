"""StructLTL variant with a GCN encoder over Boolean formula graphs."""

from math import prod
from typing import Literal, NamedTuple, cast, override

import distrax
import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.alg.struct_ltl.reach_avoid.jax_clause_graph_reach_avoid_sequence import (
    NODE_TYPE_EPSILON,
    JaxGraphReachAvoidSequence,
    NodeData,
)
from jaxolotl.environments import spaces
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.alibi_attention import ALiBiAttention
from jaxolotl.networks.conv_net import ConvNet
from jaxolotl.networks.gcn import GCN, NodeFeatures
from jaxolotl.networks.gru_cell import GRUCell
from jaxolotl.networks.mlp import MLP
from jaxolotl.networks.positional_attention import PositionalAttention
from jaxolotl.rl.actor.actor import Actor
from jaxolotl.rl.actor_critic import ActorCritic


class GCNLTLModel(ActorCritic):
    """StructLTL variant that encodes each reach/avoid step with a GCN."""

    env_net: MLP | ConvNet
    prop_embedding: nn.Embedding
    type_embedding: nn.Embedding
    gcn: GCN
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

        key, prop_emb_key, type_emb_key = jax.random.split(key, 3)
        embedding_dim = config.sequence.embedding_dim
        self.prop_embedding = nn.Embedding(
            num_embeddings=num_propositions,
            embedding_size=embedding_dim,
            key=prop_emb_key,
        )
        self.type_embedding = nn.Embedding(
            num_embeddings=NODE_TYPE_EPSILON + 1,
            embedding_size=embedding_dim,
            key=type_emb_key,
        )

        key, gcn_key = jax.random.split(key)
        self.gcn = hydra.utils.instantiate(
            config.sequence.gcn,
            in_size=embedding_dim,
            out_size=embedding_dim,
            key=gcn_key,
        )

        self._sequence_encoder = getattr(config.sequence, "encoder", "gru")
        sequence_dim = 2 * embedding_dim

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

    def _compute_sequence_embedding(self, seq: JaxGraphReachAvoidSequence) -> jax.Array:
        reach_root_features = self._get_root_features(seq.reach_graphs)
        avoid_root_features = self._get_root_features(seq.avoid_graphs)
        reach_avoid = jnp.concatenate(
            [reach_root_features, avoid_root_features], axis=-1
        )

        if self._sequence_encoder == "attention":
            return self._encode_with_attention(reach_avoid, seq.depth)
        else:
            return self._encode_with_gru(reach_avoid, seq.depth)

    def _get_root_features(self, graph) -> jax.Array:
        nodes = cast(NodeData, graph.nodes)
        prop_idx = nodes["prop_idx"]
        type_idx = nodes["type_idx"]
        node_mask = nodes["mask"]

        is_prop = prop_idx != -1
        prop_emb = jax.vmap(self.prop_embedding)(prop_idx * is_prop) * is_prop[:, None]

        is_type = type_idx != -1
        type_emb = jax.vmap(self.type_embedding)(type_idx * is_type) * is_type[:, None]

        node_features = (prop_emb + type_emb) * node_mask[:, None]
        graph_with_features = graph._replace(
            nodes={"features": node_features, "mask": node_mask}
        )
        processed_graph = self.gcn(graph_with_features)
        processed_nodes = cast(NodeFeatures, processed_graph.nodes)
        output_node_features = processed_nodes["features"]
        root_indices = jnp.concatenate([jnp.array([0]), jnp.cumsum(graph.n_node[:-1])])
        return output_node_features[root_indices]

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
