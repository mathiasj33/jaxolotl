from typing import Any, override

import hydra
import jax
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.alg.genz_ltl.model.observation_reduction import (
    ObservationReductionFunction,
)
from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.mlp import MLP
from jaxolotl.networks.observation_encoder import ObservationEncoder
from jaxolotl.rl.actor_critic import ActorCritic


class GenZLTLModel(ActorCritic):
    """GenZ-LTL model."""

    env_net: ObservationEncoder
    cost_critic: MLP
    lagrangian: MLP
    observation_reduction_fn: ObservationReductionFunction[Any, Any]

    def __init__(
        self,
        obs_spec: ObservationSpec,
        act_space: Space,
        num_assignments: int,
        num_propositions: int,
        env_params: Any,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)

        env_key, ac_key, cost_key, lag_key = jax.random.split(key, 4)
        self.observation_reduction_fn = hydra.utils.instantiate(
            config.observation_reduction_fn
        )
        reduced_spec = self.observation_reduction_fn.output_spec(
            obs_spec, env_params, num_assignments, num_propositions
        )
        self.env_net = hydra.utils.instantiate(
            config.env_net,
            input_spec=reduced_spec,
            key=env_key,
            _convert_="object",
        )
        super().__init__(config, act_space, self.env_net.output_size, ac_key)
        self.cost_critic = hydra.utils.instantiate(
            config.cost_critic,
            in_size=self.env_net.output_size,
            out_size=1,
            final_layer_activation=False,
            key=cost_key,
        )
        self.lagrangian = hydra.utils.instantiate(
            config.lagrangian,
            in_size=self.env_net.output_size,
            out_size=1,
            final_layer_activation=False,
            key=lag_key,
        )

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        reduced_obs = jax.vmap(self.observation_reduction_fn)(obs.features, obs.subgoal)
        return jax.vmap(self.env_net)(reduced_obs)

    def get_cost_value(self, obs: PyTree) -> jax.Array:
        features = self._compute_common_features(obs)
        return jax.vmap(self.cost_critic)(features).squeeze(-1)

    def get_lagrangian(self, obs: PyTree) -> jax.Array:
        features = self._compute_common_features(obs)
        return jax.vmap(self.lagrangian)(features).squeeze(-1)
