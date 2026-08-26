from typing import override

import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.observation_encoder import ObservationEncoder
from jaxolotl.rl.actor_critic import ActorCritic


class SemLTLModel(ActorCritic):
    env_net: ObservationEncoder
    projection: nn.Linear

    def __init__(
        self,
        obs_spec: ObservationSpec,
        act_space: Space,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)
        self.env_net = hydra.utils.instantiate(
            config.env_net, input_spec=obs_spec, key=env_key, _convert_="object"
        )
        key, proj_key = jax.random.split(key)
        self.projection = nn.Linear(
            in_features=config.semantic.embedding_size,
            out_features=config.embedding_dim,
            key=proj_key,
        )
        joint_dim = self.env_net.output_size + config.embedding_dim
        super().__init__(config, act_space, joint_dim, key)

    @override
    def _get_actor_context(self, obs: PyTree) -> PyTree:
        return obs.epsilon_mask

    @override
    def _compute_common_features(self, obs: PyTree) -> tuple[jax.Array, jax.Array]:
        """Computes observation features and state embeddings for current and epsilon
        states.

        Returns:
            A tuple of:
                - state embedding: (batch_size, embedding_dim)
                - epsilon state embeddings: (batch_size, max_eps_transitions, embedding_dim)
        """
        x = jax.vmap(self.env_net)(obs.features)
        emb = jax.vmap(self.projection)(obs.embedding)
        # obs.epsilon_embeddings: (batch_size,max_eps_transitions,embedding_dim)
        epsilon_emb = jax.vmap(jax.vmap(self.projection))(obs.epsilon_embeddings)
        x_tiled = jnp.tile(x[:, None, :], (1, epsilon_emb.shape[1], 1))
        return jnp.concatenate([x, emb], axis=-1), jnp.concatenate(
            [x_tiled, epsilon_emb], axis=-1
        )

    @override
    def _get_value(self, features: jax.Array) -> jax.Array:
        return jax.vmap(self.critic)(features[0]).squeeze(-1)
