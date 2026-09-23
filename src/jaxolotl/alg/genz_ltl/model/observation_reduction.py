"""Subgoal-induced observation reduction for GenZ-LTL."""

from abc import ABC, abstractmethod
from typing import NamedTuple, override

import jax
import jax.numpy as jnp

from jaxolotl.alg.genz_ltl.reach_avoid.jax_reach_avoid_subgoal import (
    JaxReachAvoidSubgoal,
)
from jaxolotl.environments.letter_world import letter_world
from jaxolotl.environments.observation_spec import (
    ArraySpec,
    ObservationSpec,
)
from jaxolotl.environments.zone_env import zone_env
from jaxolotl.networks.observation_encoder import flatten_observation


class ReducedObservation(NamedTuple):
    reduced: jax.Array


class ObservationReductionFunction[TObsFeatures: NamedTuple, TEnvParams](ABC):
    """A function that reduces environment observations based on the current subgoal."""

    @abstractmethod
    def __call__(
        self, features: TObsFeatures, subgoal: JaxReachAvoidSubgoal
    ) -> ReducedObservation:
        """Reduce the given features based on the subgoal."""
        raise NotImplementedError()

    @abstractmethod
    def output_spec(
        self,
        input_spec: ObservationSpec,
        params: TEnvParams,
        num_assignments: int,
        num_propositions: int,
    ) -> ObservationSpec:
        """Return the array specification produced by this reduction."""
        raise NotImplementedError()


class GenericObservationReduction(ObservationReductionFunction[NamedTuple, NamedTuple]):
    """A generic observation reduction that concatenates all observation features and
    encodes the subgoal as a bitvector (see Section 4.1 of the GenZ-LTL paper)."""

    @override
    def __call__(
        self, features: NamedTuple, subgoal: JaxReachAvoidSubgoal
    ) -> ReducedObservation:
        """Concatenate all features into a single vector."""
        vector = flatten_observation(features)
        vector = jnp.concatenate(
            [vector, subgoal.reach_one_hot, subgoal.avoid_one_hot],
            axis=0,
        )
        return ReducedObservation(reduced=vector)

    @override
    def output_spec(
        self,
        input_spec: ObservationSpec,
        params: NamedTuple,
        num_assignments: int,
        num_propositions: int,
    ) -> ObservationSpec:
        del params
        size = input_spec.flat_observation_size + num_propositions + num_assignments
        return ObservationSpec(reduced=ArraySpec((size,), jnp.float32))


class LetterWorldObservationReduction(
    ObservationReductionFunction[letter_world.ObsFeatures, letter_world.EnvParams]
):
    """Reduce LetterWorld observations to reach, avoid, and agent locations."""

    @override
    def __call__(
        self, features: letter_world.ObsFeatures, subgoal: JaxReachAvoidSubgoal
    ) -> ReducedObservation:
        """Return a single-channel grid induced by the current subgoal."""
        grid = features.grid
        num_letters = grid.shape[-1] - 1
        letter_grid = grid[..., :num_letters]

        reach_index = jnp.clip(subgoal.reach, 0, num_letters - 1)
        reach_is_letter = (subgoal.reach >= 0) & (subgoal.reach < num_letters)
        reach_mask = (letter_grid[..., reach_index] > 0) & reach_is_letter

        avoid_indices = jnp.clip(subgoal.avoid, 0, num_letters - 1)
        avoid_is_letter = (subgoal.avoid >= 0) & (subgoal.avoid < num_letters)
        avoid_mask = jnp.any(
            (letter_grid[..., avoid_indices] > 0) & avoid_is_letter,
            axis=-1,
        )
        agent_mask = grid[..., -1] > 0

        reduced = jnp.zeros(grid.shape[:2], dtype=jnp.float32)
        reduced = jnp.where(avoid_mask, 0.5, reduced)
        reduced = jnp.where(reach_mask, 1.0, reduced)
        reduced = jnp.where(agent_mask, 0.2, reduced)
        return ReducedObservation(reduced=reduced[..., None])

    @override
    def output_spec(
        self,
        input_spec: ObservationSpec,
        params: letter_world.EnvParams,
        num_assignments: int,
        num_propositions: int,
    ) -> ObservationSpec:
        del input_spec, num_assignments, num_propositions
        return ObservationSpec(
            reduced=ArraySpec((params.grid_size, params.grid_size, 1), jnp.float32)
        )


class ZoneEnvObservationReduction(
    ObservationReductionFunction[zone_env.ObsFeatures, zone_env.EnvParams]
):
    """Observation reduction for ZoneEnv."""

    @override
    def __call__(
        self, features: zone_env.ObsFeatures, subgoal: JaxReachAvoidSubgoal
    ) -> ReducedObservation:
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
        return ReducedObservation(
            reduced=jnp.concatenate([agent_obs, reach_obs, avoid_obs], axis=0)
        )

    @override
    def output_spec(
        self,
        input_spec: ObservationSpec,
        params: zone_env.EnvParams,
        num_assignments: int,
        num_propositions: int,
    ) -> ObservationSpec:
        del input_spec, num_assignments, num_propositions
        return ObservationSpec(
            reduced=ArraySpec((5 + 2 * params.num_lidar_bins,), jnp.float32)
        )
