from typing import override

import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.alg.common.reach_avoid.jax_sequence import (
    JaxReachAvoidSequence,
)
from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.deep_sets import DeepSets
from jaxolotl.networks.observation_encoder import ObservationEncoder
from jaxolotl.networks.sequence_encoder import SequenceEncoder
from jaxolotl.rl.actor_critic import ActorCritic


class DeepLTLModel(ActorCritic):
    env_net: ObservationEncoder
    embedding: nn.Embedding
    deep_sets: DeepSets
    sequence_encoder: SequenceEncoder

    def __init__(
        self,
        obs_spec: ObservationSpec,
        act_space: Space,
        num_assignments: int,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)
        self.env_net = hydra.utils.instantiate(
            config.env_net, input_spec=obs_spec, key=env_key, _convert_="object"
        )
        key, emb_key = jax.random.split(key)
        embedding_dim = config.sequence.embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_assignments + 1,  # +1 for epsilon transitions
            embedding_size=embedding_dim,
            key=emb_key,
        )
        key, ds_key = jax.random.split(key)
        self.deep_sets = hydra.utils.instantiate(
            config.sequence.deep_sets,
            embedding_dim=embedding_dim,
            key=ds_key,
        )
        key, encoder_key = jax.random.split(key)
        self.sequence_encoder = SequenceEncoder(
            input_size=2 * config.sequence.deep_sets.out_size,
            hidden_size=2 * embedding_dim,
            mode="gru",
            attention_config=None,
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

    def _compute_sequence_embedding(self, seq: JaxReachAvoidSequence) -> jax.Array:
        def embed_assignment_set(indices: jax.Array) -> jax.Array:
            # indices shape: (num_assignments,)
            mask = indices != -1
            embeddings = jax.vmap(self.embedding)(indices * mask) * (mask[:, None])
            # embeddings shape: (num_assignments, embedding_dim)
            return self.deep_sets(embeddings)  # shape: (out_size,)

        reach_emb = jax.vmap(embed_assignment_set)(seq.reach)
        avoid_emb = jax.vmap(embed_assignment_set)(seq.avoid)
        reach_avoid = jnp.concatenate([reach_emb, avoid_emb], axis=-1)
        return self.sequence_encoder(reach_avoid, seq.depth)
