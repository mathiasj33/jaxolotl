from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.testing as npt

from jaxolotl import eqx_utils
from jaxolotl.environments import spaces
from jaxolotl.environments.environment import Environment, EnvParams
from jaxolotl.environments.observation_spec import ArraySpec, ObservationSpec
from jaxolotl.environments.wrappers.precomputed_reset_wrapper import (
    PrecomputedResetWrapper,
)
from jaxolotl.ltl.logic.assignment import Assignment


@dataclass(frozen=True)
class _Params(EnvParams):
    pass


class _Descriptor(eqx.Module):
    value: jax.Array


class _State(eqx.Module):
    value: jax.Array
    derived: jax.Array


class _Features(NamedTuple):
    value: jax.Array


class _ResetOptions(NamedTuple):
    pass


class _DescriptorEnv(
    Environment[_State, _Descriptor, _Params, _Features, _ResetOptions]
):
    def __init__(self):
        super().__init__(_Params(max_steps_in_episode=1), ("ready",))

    def _sample_reset(self, key, descriptor, params, options=None):
        del descriptor, params, options
        return _Descriptor(jax.random.uniform(key, (2,)))

    def materialize(self, descriptor, params):
        del params
        return _State(descriptor.value, jnp.outer(descriptor.value, descriptor.value))

    def _step(self, key, state, action, params):
        del key, action, params
        return state, jnp.zeros(()), jnp.asarray(False), {}

    def _compute_obs(self, state, params):
        del params
        return _Features(state.value)

    def compute_propositions(self, state, params):
        del state, params
        return -jnp.ones((1,), dtype=jnp.int32)

    def _observation_spec(self, params):
        del params
        return ObservationSpec(value=ArraySpec((2,), jnp.float32))

    def _action_space(self, params):
        del params
        return spaces.Discrete(shape=(1,), n=1)

    @staticmethod
    def assignments():
        return [Assignment(frozenset())]

    def get_renderer(self, params, **kwargs):
        raise NotImplementedError


def test_reset_materializes_compact_descriptor():
    env = _DescriptorEnv()
    descriptor = env.sample_reset(jax.random.key(0), None, env.default_params)
    state, obs = env.reset(jax.random.key(0), None, env.default_params)

    assert isinstance(descriptor, _Descriptor)
    assert isinstance(state, _State)
    npt.assert_allclose(state.value, descriptor.value)
    npt.assert_allclose(state.derived, jnp.outer(state.value, state.value))
    npt.assert_allclose(obs.features.value, state.value)


def test_precomputed_resets_store_descriptors_and_materialize(tmp_path):
    env = _DescriptorEnv()
    descriptors = _Descriptor(jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32))
    path = tmp_path / "resets.eqx"
    eqx_utils.save(path, descriptors, metadata={"batch_dim": 2})

    wrapped = PrecomputedResetWrapper(env, env.default_params, path)
    state, obs = wrapped.reset(jax.random.key(1), None, env.default_params)

    assert isinstance(wrapped.reset_descriptors, _Descriptor)
    assert isinstance(state, _State)
    npt.assert_allclose(state.derived, jnp.outer(state.value, state.value))
    npt.assert_allclose(obs.features.value, state.value)
    assert bool(jnp.any(jnp.all(descriptors.value == state.value, axis=1)))
