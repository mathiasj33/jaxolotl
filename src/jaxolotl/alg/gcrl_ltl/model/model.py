from typing import override

import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.environments import spaces
from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.networks.observation_encoder import ObservationEncoder
from jaxolotl.rl.actor_critic import ActorCritic


class GCRLModel(ActorCritic):
    """Proposition-conditioned actor/value/Q model."""

    env_net: ObservationEncoder
    embedding: nn.Embedding

    def __init__(
        self,
        obs_spec: ObservationSpec,
        act_space: spaces.Discrete,
        num_propositions: int,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)
        self.env_net = hydra.utils.instantiate(
            config.env_net, input_spec=obs_spec, key=env_key, _convert_="object"
        )
        embedding_dim = config.embedding_dim
        key, emb_key = jax.random.split(key)
        self.embedding = nn.Embedding(
            num_embeddings=num_propositions, embedding_size=embedding_dim, key=emb_key
        )
        joint_dim = self.env_net.output_size + embedding_dim
        super().__init__(config, act_space, joint_dim, key)

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = jax.vmap(self.env_net)(obs.features)
        embedding = jax.vmap(self.embedding)(obs.goal)
        return jnp.concatenate([x, embedding], axis=-1)
