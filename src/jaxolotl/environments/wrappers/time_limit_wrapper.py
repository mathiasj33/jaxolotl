from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition
from jaxolotl.environments.wrappers.wrapper import EnvWrapper, WrapperState


class TimeLimitState(WrapperState):
    timestep: jax.Array  # int


class TimeLimitWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]):
    """Truncate episodes at the environment time limit.

    By default, reaching the limit is not a termination: value-based algorithms may
    bootstrap from the terminal observation. ``treat_trunc_as_term`` retains the
    legacy Gym-style behaviour for callers that explicitly require it.
    """

    treat_trunc_as_term: bool

    def __init__(
        self,
        env: (
            EnvWrapper[TEnvParams, TObsFeatures, TResetOptions]
            | Environment[Any, Any, TEnvParams, TObsFeatures, TResetOptions]
        ),
        treat_trunc_as_term: bool = False,
    ):
        super().__init__(env)
        self.treat_trunc_as_term = treat_trunc_as_term

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: TimeLimitState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[TimeLimitState, EnvObservation[TObsFeatures]]:
        re_state, obs = super().reset(key, state, params, options)
        return TimeLimitState(
            state=re_state, timestep=jnp.array(0, dtype=jnp.int32)
        ), obs

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: TimeLimitState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[TimeLimitState, TObsFeatures]:
        transition = super().step(key, state, action, params)
        next_state = TimeLimitState(
            timestep=state.timestep + 1,
            state=transition.state,
        )
        time_limit_reached = next_state.timestep >= params.max_steps_in_episode  # type: ignore
        truncated = jnp.logical_or(transition.truncated, time_limit_reached)
        terminated = transition.terminated
        if self.treat_trunc_as_term:
            terminated = jnp.logical_or(terminated, truncated)
        return transition._replace(
            state=next_state, terminated=terminated, truncated=truncated
        )  # type: ignore
