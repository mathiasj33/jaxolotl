from abc import abstractmethod

import distrax
import equinox as eqx
import hydra
import jax
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.environments import spaces
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.mlp import MLP
from jaxolotl.rl.actor.actor import Actor


class ActorCritic(eqx.Module):
    """Abstract base class for actor-critic models."""

    actor: Actor
    critic: MLP

    def __init__(
        self,
        config: DictConfig,
        act_space: Space,
        in_size: int,
        key: jax.Array,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = hydra.utils.instantiate(
            config.actor,
            in_size=in_size,
            **self._actor_kwargs_from_space(act_space),
            key=actor_key,
        )
        self.critic = hydra.utils.instantiate(
            config.critic,
            in_size=in_size,
            out_size=1,
            final_layer_activation=config.critic.get("final_layer_activation", False),
            key=critic_key,
        )

    def _actor_kwargs_from_space(self, act_space: Space) -> dict[str, int]:
        """Return actor constructor arguments for an action space."""
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

    def __call__(self, obs: PyTree) -> tuple[distrax.Distribution, jax.Array]:
        """Forward pass through the actor and critic networks.

        Args:
            obs: Batched observations.

        Returns:
            A tuple of (action distribution, state value).
        """
        features = self._compute_common_features(obs)
        dist = self.actor(features, self._get_actor_context(obs))
        value = self._get_value(features)
        return dist, value

    def get_action(self, obs: PyTree) -> distrax.Distribution:
        """Get action distribution from the actor network.

        Args:
            obs: Batched observations.

        Returns:
            Batched action distribution.
        """
        return self.actor(
            self._compute_common_features(obs), self._get_actor_context(obs)
        )

    def _get_actor_context(self, obs: PyTree) -> PyTree | None:
        """Return optional observation-dependent information required by the actor."""
        del obs
        return None

    def get_value(self, obs: PyTree) -> jax.Array:
        """Get state value from the critic network.

        Args:
            obs: Batched observations.

        Returns:
            State values.
        """
        return self._get_value(self._compute_common_features(obs))

    def _get_value(self, features: jax.Array) -> jax.Array:
        """Get state value from the critic network.

        Args:
            features: Batched features.

        Returns:
            Batched state values.
        """
        return jax.vmap(self.critic)(features).squeeze(-1)

    @abstractmethod
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        """Compute common features from observations for actor and critic.

        Args:
            obs: Batched observations.

        Returns:
            Common features.
        """
        pass
