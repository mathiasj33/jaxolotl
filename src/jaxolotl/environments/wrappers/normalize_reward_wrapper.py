from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class NormalizeRewardEnvState(WrapperState):
    accumulated: jax.Array  # float
    mean: jax.Array  # float
    m2: jax.Array  # float
    var: jax.Array  # float
    count: jax.Array  # float


class NormalizeRewardWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]):
    """Normalize the rewards returned by the environment on a per-episode basis."""

    gamma: jax.Array  # float
    eps: jax.Array  # float

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, TResetOptions]
        ),
        gamma: float = 0.99,
        eps: float = 1e-8,
    ):
        super().__init__(env)
        self.gamma = jnp.array(gamma, dtype=jnp.float32)
        self.eps = jnp.array(eps, dtype=jnp.float32)

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: NormalizeRewardEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[NormalizeRewardEnvState, EnvObservation[TObsFeatures]]:
        re_state, obs = super().reset(key, state, params, options)
        return self._wrap_reset_state(re_state), obs

    def _wrap_reset_state(self, env_state: Any) -> NormalizeRewardEnvState:
        return NormalizeRewardEnvState(
            state=env_state,
            accumulated=jnp.array(0.0, dtype=jnp.float32),
            mean=jnp.array(0.0, dtype=jnp.float32),
            m2=jnp.array(0.0, dtype=jnp.float32),
            var=jnp.array(1.0, dtype=jnp.float32),
            count=jnp.array(1e-4, dtype=jnp.float32),
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: NormalizeRewardEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[NormalizeRewardEnvState, TObsFeatures]:
        transition = super().step(key, state, action, params)
        reward = transition.reward
        accumulated = (
            state.accumulated * self.gamma * (1.0 - transition.terminated) + reward
        )
        delta = accumulated - state.mean
        count = state.count + 1.0
        mean = state.mean + delta / count
        delta2 = accumulated - mean
        m2 = state.m2 + delta * delta2
        var = m2 / count
        normalized_reward = reward / jnp.sqrt(var + self.eps)
        new_state = NormalizeRewardEnvState(
            state=transition.state,
            accumulated=accumulated,
            mean=mean,
            m2=m2,
            var=var,
            count=count,
        )
        return EnvTransition(
            state=new_state,
            observation=transition.observation,
            reward=normalized_reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            terminal_observation=transition.terminal_observation,
            propositions=transition.propositions,
            info=transition.info,
        )
