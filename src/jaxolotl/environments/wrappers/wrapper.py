"""Environment wrappers.

Adapted from gymnax
(https://github.com/RobertTLange/gymnax/blob/main/gymnax/wrappers/purerl.py).
"""

import dataclasses
from typing import Any, NamedTuple

import equinox as eqx
import jax

from jaxolotl.environments.environment import Environment, EnvObservation, EnvTransition


class WrapperState(eqx.Module):
    """Base class for wrapper states. Wrappers can add fields to this class to maintain
    their own state."""

    state: eqx.Module  # the state of the wrapped environment / previous wrapper

    def unwrapped(self) -> eqx.Module:
        """Recursively unwraps the environment to get the base environment."""
        if isinstance(self.state, WrapperState):
            return self.state.unwrapped()
        return self.state

    def replace_in_chain[T: WrapperState](
        self, state_type: type[T], **changes: Any
    ) -> "WrapperState":
        """Recursively replaces the state of the first wrapper of the given type in the chain."""
        if isinstance(self, state_type):
            return dataclasses.replace(self, **changes)
        if isinstance(self.state, WrapperState):
            new_state = self.state.replace_in_chain(state_type, **changes)
            return dataclasses.replace(self, state=new_state)
        raise ValueError(f"No wrapper of type {state_type} found in chain.")

    def __getattr__(self, name):
        return getattr(self.state, name)


class EnvWrapper[
    TEnvParams,
    TObsFeatures: NamedTuple,
    TResetOptions: NamedTuple,
](eqx.Module):
    """Base class for environment wrappers."""

    _env: "EnvWrapper[TEnvParams, TObsFeatures, TResetOptions] | Environment[Any, Any, TEnvParams, TObsFeatures, TResetOptions]"
    uses_state: bool  # whether the wrapper uses custom state that needs to be unwrapped

    def __init__(
        self,
        env: "EnvWrapper[TEnvParams, TObsFeatures, TResetOptions] | Environment[Any, Any, TEnvParams, TObsFeatures, TResetOptions]",
        uses_state: bool = True,
    ):
        self._env = env
        self.uses_state = uses_state

    @eqx.filter_jit
    def reset(
        self,
        key: jax.Array,
        state: WrapperState | None,
        params: TEnvParams,
        options: TResetOptions | None = None,
    ) -> tuple[WrapperState, EnvObservation[TObsFeatures]]:
        if state is None:
            return self._env.reset(key, None, params, options)
        if self.uses_state:
            state = state.state  # type: ignore
        return self._env.reset(key, state, params, options)

    @eqx.filter_jit
    def step(
        self,
        key: jax.Array,
        state: WrapperState,
        action: int | float | jax.Array,
        params: TEnvParams,
    ) -> EnvTransition[WrapperState, TObsFeatures]:
        if self.uses_state:
            state = state.state  # type: ignore
        return self._env.step(key, state, action, params)

    # provide proxy access to regular attributes of wrapped environment
    def __getattr__(self, name):
        return getattr(self._env, name)
