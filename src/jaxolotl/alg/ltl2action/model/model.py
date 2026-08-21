from typing import override

import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.alg.ltl2action.utils.jax_formula_closure import JaxFormulaGraph
from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.observation_encoder import ObservationEncoder
from jaxolotl.networks.rgcn import RGCN
from jaxolotl.rl.actor_critic import ActorCritic


class LTL2ActionModel(ActorCritic):
    env_net: ObservationEncoder
    embedding: nn.Embedding
    rgcn: RGCN

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
        key, emb_key = jax.random.split(key)
        embedding_dim = config.rgcn.embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_propositions + 8,  # &, |, !, F, G, U, tt, ff
            embedding_size=embedding_dim,
            key=emb_key,
        )
        key, rgcn_key = jax.random.split(key)
        self.rgcn = hydra.utils.instantiate(
            config.rgcn,
            in_size=embedding_dim,
            out_size=embedding_dim,
            num_relations=3,  # unary, binary_left, binary_right
            key=rgcn_key,
        )
        joint_dim = self.env_net.output_size + embedding_dim
        super().__init__(config, act_space, joint_dim, key)

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = jax.vmap(self.env_net)(obs.features)
        emb = jax.vmap(self._compute_root_features)(obs.graph)
        return jnp.concatenate([x, emb], axis=-1)

    def _compute_root_features(self, graph: JaxFormulaGraph) -> jax.Array:
        """Embeds graph nodes, runs RGCN, and extracts root node features.

        Args:
            graph: A single (non-batched) JaxFormulaGraph.
        """
        embeddings = jax.vmap(self.embedding)(graph.nodes) * graph.node_mask[:, None]
        features = self.rgcn(graph, embeddings)
        return features[0]  # root node is always node 0
