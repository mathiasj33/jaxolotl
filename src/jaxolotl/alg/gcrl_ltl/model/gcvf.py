import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.networks.mlp import MLP
from jaxolotl.networks.observation_encoder import ObservationEncoder


class GCVF(eqx.Module):
    """Goal-conditioned value function for GCRL-LTL."""

    env_net: ObservationEncoder
    embedding: nn.Embedding
    value_net: MLP

    def __init__(
        self,
        obs_spec: ObservationSpec,
        env_net_config: DictConfig,
        value_net_config: DictConfig,
        embedding_dim: int,
        num_propositions: int,
        key: jax.Array,
    ):
        key, env_key = jax.random.split(key)
        self.env_net = hydra.utils.instantiate(
            env_net_config, input_spec=obs_spec, key=env_key, _convert_="object"
        )
        key, emb_key = jax.random.split(key)
        self.embedding = nn.Embedding(
            num_embeddings=num_propositions, embedding_size=embedding_dim, key=emb_key
        )
        key, value_key = jax.random.split(key)
        self.value_net = hydra.utils.instantiate(
            value_net_config,
            in_size=self.env_net.output_size + 2 * embedding_dim,
            out_size=1,
            final_layer_activation=False,
            key=value_key,
        )

    def __call__(self, obs: PyTree, target_goal: jax.Array) -> jax.Array:
        x = jax.vmap(self.env_net)(obs.features)
        embedding = jax.vmap(self.embedding)(obs.goal)
        target_embedding = jax.vmap(self.embedding)(target_goal)
        concat = jnp.concatenate([x, embedding, target_embedding], axis=-1)
        return jax.vmap(self.value_net)(concat).squeeze(-1)


def build_gcvf_models(
    obs_spec: ObservationSpec,
    num_propositions: int,
    env_net_config: DictConfig,
    value_net_config: DictConfig,
    embedding_dim: int,
    keys: jax.Array,
) -> GCVF:
    return eqx.filter_vmap(GCVF, in_axes=(None, None, None, None, None, 0))(
        obs_spec,
        env_net_config,
        value_net_config,
        embedding_dim,
        num_propositions,
        keys,
    )
