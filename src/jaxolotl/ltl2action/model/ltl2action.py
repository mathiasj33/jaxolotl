from math import prod
from typing import NamedTuple, override

import distrax
import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.environments import spaces
from jaxolotl.environments.spaces import Space
from jaxolotl.ltl2action.utils.jax_formula_closure import JaxFormulaGraph
from jaxolotl.networks.conv_net import ConvNet
from jaxolotl.networks.mlp import MLP
from jaxolotl.networks.rgcn import RGCN
from jaxolotl.rl.actor.actor import Actor
from jaxolotl.rl.actor_critic import ActorCritic


class LTL2ActionModel(ActorCritic):
    env_net: MLP | ConvNet
    embedding: nn.Embedding
    rgcn: RGCN
    actor: Actor
    critic: MLP

    _flatten_features: bool

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
        is_conv = "ConvNet" in config.env_net._target_  # TODO: clean up
        params = {"obs_shape": obs_shape} if is_conv else {"in_size": prod(obs_shape)}
        self.env_net = hydra.utils.instantiate(config.env_net, **params, key=env_key)
        self._flatten_features = not is_conv
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
        actor_key, critic_key = jax.random.split(key)
        joint_dim = self.env_net.output_size + embedding_dim
        params = self._get_actor_params_from_space(act_space)
        self.actor = hydra.utils.instantiate(  # TODO: reduce code duplication between different models
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
        return self.actor(features, None)

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

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
