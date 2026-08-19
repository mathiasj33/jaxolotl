from typing import Any, override

import distrax
import hydra
import jax
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.environments import spaces
from jaxolotl.environments.spaces import Space
from jaxolotl.genz_ltl.model.observation_reduction import (
    ObservationReductionFunction,
)
from jaxolotl.networks.mlp import MLP
from jaxolotl.rl.actor.actor import Actor
from jaxolotl.rl.actor_critic import ActorCritic


class GenZLTLModel(ActorCritic):
    """GenZ-LTL model."""

    env_net: MLP
    actor: Actor
    critic: MLP
    cost_critic: MLP
    lagrangian: MLP
    observation_reduction_fn: ObservationReductionFunction[Any, Any]

    def __init__(
        self,
        act_space: Space,
        key: jax.Array,
        obs_shape: tuple[int, ...],
        num_assignments: int,
        env_params: Any,
        **kwargs,
    ):
        del obs_shape, num_assignments
        config = DictConfig(kwargs)

        key, env_key, actor_key, critic_key, cost_key, lag_key = jax.random.split(
            key, 6
        )
        self.observation_reduction_fn = hydra.utils.instantiate(
            config.observation_reduction_fn
        )
        obs_size = self.observation_reduction_fn.output_size(env_params)
        self.env_net = hydra.utils.instantiate(
            config.env_net, in_size=obs_size, key=env_key
        )
        in_size = self.env_net.output_size

        params = self._get_actor_params_from_space(act_space)
        self.actor = hydra.utils.instantiate(
            config.actor,
            in_size=in_size,
            **params,
            key=actor_key,
        )
        self.critic = hydra.utils.instantiate(
            config.critic,
            in_size=in_size,
            out_size=1,
            final_layer_activation=False,
            key=critic_key,
        )
        self.cost_critic = hydra.utils.instantiate(
            config.cost_critic,
            in_size=in_size,
            out_size=1,
            final_layer_activation=False,
            key=cost_key,
        )
        self.lagrangian = hydra.utils.instantiate(
            config.lagrangian,
            in_size=in_size,
            out_size=1,
            final_layer_activation=False,
            key=lag_key,
        )

    @staticmethod
    def _get_actor_params_from_space(act_space: Space) -> dict:
        if isinstance(act_space, spaces.Discrete):
            return {"num_actions": act_space.n}
        if isinstance(act_space, spaces.Box):
            return {"action_dim": act_space.shape[0]}
        if isinstance(act_space, spaces.Composite):
            return {
                "continuous_action_dim": act_space.continuous.shape[0],
                "num_discrete_actions": act_space.discrete.n,
            }
        raise NotImplementedError(f"Unsupported action space {type(act_space)}")

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        reduced_obs = jax.vmap(self.observation_reduction_fn)(obs.features, obs.subgoal)
        return jax.vmap(self.env_net)(reduced_obs)

    @override
    def _get_action(self, features: jax.Array, obs: PyTree) -> distrax.Distribution:
        return self.actor(features, None)

    @override
    def _get_value(self, features: jax.Array) -> jax.Array:
        return jax.vmap(self.critic)(features).squeeze(-1)

    def get_cost_value(self, obs: PyTree) -> jax.Array:
        features = self._compute_common_features(obs)
        return jax.vmap(self.cost_critic)(features).squeeze(-1)

    def get_lagrangian(self, obs: PyTree) -> jax.Array:
        features = self._compute_common_features(obs)
        return jax.vmap(self.lagrangian)(features).squeeze(-1)
