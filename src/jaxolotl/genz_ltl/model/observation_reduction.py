"""Subgoal-induced observation reduction for GenZ-LTL.

Initial implementation supports ZoneEnv only.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple, override

import jax
import jax.numpy as jnp

from jaxolotl.environments.zone_env import zone_env
from jaxolotl.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import JaxReachAvoidSubgoal


class ObservationReductionFunction[TObsFeatures: NamedTuple, TEnvParams](ABC):
    """A function that reduces environment observations based on the current subgoal."""

    @abstractmethod
    def __call__(
        self, features: TObsFeatures, subgoal: JaxReachAvoidSubgoal
    ) -> jax.Array:
        """Reduce the given features based on the subgoal."""
        raise NotImplementedError()

    @abstractmethod
    def output_size(self, params: TEnvParams) -> int:
        """The size of the reduced observation vector."""
        raise NotImplementedError()


class GenericObservationReduction(ObservationReductionFunction[NamedTuple, NamedTuple]):
    """A generic observation reduction that concatenates all observation features and
    encodes the subgoal as a bitvector (see section 4.1 of the GenZ-LTL paper)."""

    def __init__(self, output_size: int):
        self._output_size = output_size

    @override
    def __call__(
        self, features: NamedTuple, subgoal: JaxReachAvoidSubgoal
    ) -> jax.Array:
        """Concatenate all features into a single vector."""
        vector = self._flatten_features(features)
        vector = jnp.concatenate(
            [vector, subgoal.reach_one_hot, subgoal.avoid_one_hot],
            axis=0,
        )
        return vector

    def _flatten_features(self, features: NamedTuple) -> jax.Array:
        return jnp.concatenate([v.flatten() for v in jax.tree.leaves(features)], axis=0)

    @override
    def output_size(self, _: NamedTuple) -> int:
        return self._output_size


class ZoneEnvObservationReduction(
    ObservationReductionFunction[zone_env.ObsFeatures, zone_env.EnvParams]
):
    """Observation reduction for ZoneEnv."""

    @override
    def __call__(
        self, features: zone_env.ObsFeatures, subgoal: JaxReachAvoidSubgoal
    ) -> jax.Array:
        """Reduce ZoneEnv observations to [agent_obs, reach_obs, avoid_obs].

        Returns:
            reduced feature vector of shape (5 + 2*num_bins,)
        """
        agent_obs = jnp.concatenate(
            [features.acceleration, features.velocity, features.angular_velocity],
            axis=0,
        )
        reach_obs = features.lidar[subgoal.reach]
        avoid_lidars = jnp.where(
            jnp.reshape(subgoal.avoid != -1, (-1, 1)), features.lidar[subgoal.avoid], 0
        )
        avoid_obs = jnp.max(avoid_lidars, axis=0)
        return jnp.concatenate([agent_obs, reach_obs, avoid_obs], axis=0)

    @override
    def output_size(self, params: zone_env.EnvParams) -> int:
        """Output size of the reduced observation vector."""
        return 5 + 2 * params.num_lidar_bins
