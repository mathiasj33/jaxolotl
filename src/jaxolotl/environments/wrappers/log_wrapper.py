from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class LogEnvState(WrapperState):
    step: jax.Array  # int
    total_step: jax.Array  # int
    ret: jax.Array  # float
    pos_ret: jax.Array  # float


class LogWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]):
    """Log the episode returns and lengths to the info dict."""

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]
            | Environment[Any, TEnvParams, TObsFeatures, TResetOptions]
        ),
    ):
        super().__init__(env)

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: LogEnvState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[LogEnvState, EnvObservation[TObsFeatures]]:
        re_state, obs = super().reset(key, state, params, options)
        return self._wrap_reset_state(re_state), obs

    def _wrap_reset_state(self, state: Any) -> LogEnvState:
        return LogEnvState(
            state=state,
            step=jnp.array(0, dtype=jnp.int32),
            total_step=jnp.array(0, dtype=jnp.int32),
            ret=jnp.array(0.0, dtype=jnp.float32),
            pos_ret=jnp.array(0.0, dtype=jnp.float32),
        )

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: LogEnvState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[LogEnvState, TObsFeatures]:
        transition = super().step(key, state, action, params)
        ret = transition.reward + state.ret
        pos_ret = jax.nn.relu(transition.reward) + state.pos_ret
        length = state.step + 1
        total_step = state.total_step + 1
        stage = transition.state.curriculum_stage + 1
        log_state = LogEnvState(
            step=(state.step + 1) * (1 - transition.done),
            state=transition.state,
            total_step=state.total_step + 1,
            ret=ret * (1.0 - transition.done),
            pos_ret=pos_ret * (1.0 - transition.done),
        )
        info = {
            "episode_return": ret,
            "positive_return": pos_ret,
            "episode_length": length,
            "total_step": total_step,
            "curriculum_stage": stage,
            "done": transition.done,
        } | transition.info
        return EnvTransition(
            state=log_state,
            observation=transition.observation,
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            terminal_observation=transition.terminal_observation,
            propositions=transition.propositions,
            info=info,
        )
