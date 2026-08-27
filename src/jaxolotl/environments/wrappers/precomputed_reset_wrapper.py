from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl import eqx_utils
from jaxolotl.environments.environment import Environment, EnvObservation
from jaxolotl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class PrecomputedResetWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]):
    """Reset from compact descriptors loaded from disk.

    Descriptors are materialized only after selection. This is identical to
    storing states for array-only environments, while avoiding serialization of
    large derived simulator state for MJX environments.
    """

    reset_descriptors: eqx.Module  # batched compact reset descriptors
    num_reset_states: int

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, TResetOptions]
        ),
        params: TEnvParams,
        path: str | Path,
    ):
        super().__init__(env, uses_state=False)
        self.num_reset_states = eqx_utils.load_metadata(path)["batch_dim"]
        descriptor_template = env.sample_reset(jax.random.key(0), None, params=params)
        descriptor_template = jax.tree.map(  # Add batch dimension
            lambda x: jnp.zeros((self.num_reset_states,) + x.shape, dtype=x.dtype),
            descriptor_template,
        )
        self.reset_descriptors = eqx_utils.load(path, descriptor_template)

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        _state: WrapperState | None,
        params: TEnvParams,
        options: (
            TResetOptions | None
        ) = None,  # note: currently ignored for precomputed resets
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        del options
        descriptor = self._sample_random_reset_descriptor(key)
        state = self._env.materialize(descriptor, params)
        obs = self._env.compute_obs(state, params)
        return state, obs

    def _sample_random_reset_descriptor(self, key: jax.Array) -> eqx.Module:
        index = jax.random.randint(
            key, shape=(), minval=0, maxval=self.num_reset_states
        )
        return jax.tree.map(lambda x: x[index], self.reset_descriptors)
