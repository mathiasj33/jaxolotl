from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.environments import spaces
from jaxolotl.environments.environment import Environment, EnvParams
from jaxolotl.environments.observation_spec import ArraySpec, ObservationSpec
from jaxolotl.environments.wrappers.time_limit_wrapper import TimeLimitWrapper
from jaxolotl.ltl.logic.assignment import Assignment


@dataclass(frozen=True)
class _Params(EnvParams):
    truncate_on_step: int | None = None


class _State(eqx.Module):
    step: jax.Array


class _Features(NamedTuple):
    step: jax.Array


class _Options(NamedTuple):
    pass


class _TruncatingEnv(Environment[_State, _State, _Params, _Features, _Options]):
    def __init__(self, *, terminate_on_step: int | None = None):
        super().__init__(_Params(max_steps_in_episode=2), ())
        self.terminate_on_step = terminate_on_step

    terminate_on_step: int | None

    def _sample_reset(self, key, descriptor, params, options=None):
        del key, descriptor, params, options
        return _State(jnp.asarray(0))

    def _step(self, key, state, action, params):
        del key, action, params
        next_state = _State(state.step + 1)
        terminated = next_state.step == self.terminate_on_step
        return next_state, jnp.asarray(0.0), terminated, {}

    def step(self, key, state, action, params):
        transition = super().step(key, state, action, params)
        truncated = (
            jnp.asarray(False)
            if params.truncate_on_step is None
            else transition.state.step == params.truncate_on_step
        )
        return transition._replace(truncated=truncated)

    def _compute_obs(self, state, params):
        del params
        return _Features(state.step)

    def compute_propositions(self, state, params):
        del state, params
        return jnp.empty((0,), dtype=jnp.int32)

    def _observation_spec(self, params):
        del params
        return ObservationSpec(step=ArraySpec((), jnp.int32))

    def _action_space(self, params):
        del params
        return spaces.Discrete(shape=(), n=1)

    @staticmethod
    def assignments():
        return [Assignment(frozenset())]

    def get_renderer(self, params, **kwargs):
        raise NotImplementedError


def _step(wrapper, params):
    state, _ = wrapper.reset(jax.random.key(0), None, params)
    return wrapper.step(jax.random.key(1), state, 0, params)


def test_time_limit_truncates_without_terminating_by_default() -> None:
    env = _TruncatingEnv()
    params = _Params(max_steps_in_episode=1)

    transition = _step(TimeLimitWrapper(env), params)

    assert bool(transition.truncated)
    assert not bool(transition.terminated)


def test_time_limit_can_treat_truncation_as_termination() -> None:
    env = _TruncatingEnv()
    params = _Params(max_steps_in_episode=1)

    transition = _step(TimeLimitWrapper(env, treat_trunc_as_term=True), params)

    assert bool(transition.truncated)
    assert bool(transition.terminated)


def test_time_limit_preserves_wrapped_environment_truncation() -> None:
    env = _TruncatingEnv()
    params = _Params(max_steps_in_episode=5, truncate_on_step=1)

    transition = _step(TimeLimitWrapper(env), params)

    assert bool(transition.truncated)
    assert not bool(transition.terminated)
