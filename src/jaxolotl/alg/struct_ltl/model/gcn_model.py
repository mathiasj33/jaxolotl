"""StructLTL variant with a GCN encoder over Boolean formula graphs."""

from typing import cast, override

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
from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.gcn import GCN, NodeFeatures
from jaxolotl.networks.observation_encoder import ObservationEncoder
from jaxolotl.networks.sequence_encoder import SequenceEncoder
from jaxolotl.rl.actor_critic import ActorCritic


class GCNLTLModel(ActorCritic):
    """StructLTL variant that encodes each reach/avoid step with a GCN."""

    env_net: ObservationEncoder
    prop_embedding: nn.Embedding
    type_embedding: nn.Embedding
    gcn: GCN
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
        self.env_net = hydra.utils.instantiate(
            config.env_net, input_spec=obs_spec, key=env_key, _convert_="object"
        )

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

        sequence_dim = 2 * embedding_dim

        key, encoder_key = jax.random.split(key)
        self.sequence_encoder = SequenceEncoder(
            input_size=sequence_dim,
            hidden_size=2 * embedding_dim,
            mode=config.sequence.encoder,
            attention_config=getattr(config.sequence, "attention", None),
            key=encoder_key,
        )

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

    def _compute_sequence_embedding(self, seq: JaxGraphReachAvoidSequence) -> jax.Array:
        reach_root_features = self._get_root_features(seq.reach_graphs)
        avoid_root_features = self._get_root_features(seq.avoid_graphs)
        reach_avoid = jnp.concatenate(
            [reach_root_features, avoid_root_features], axis=-1
        )

        return self.sequence_encoder(reach_avoid, seq.depth)

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
