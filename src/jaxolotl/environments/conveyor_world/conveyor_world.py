"""An implementation of the ConveyorWorld environment.

The agent must move and collect objects. It cannot move against the conveyor belt direction.
"""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxolotl.environments import environment, spaces
from jaxolotl.environments.observation_spec import ArraySpec, ObservationSpec
from jaxolotl.ltl.logic.assignment import Assignment

if TYPE_CHECKING:
    from jaxolotl.environments.renderer.renderer import BaseRenderer


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    pass


class EnvState(eqx.Module):
    position: jax.Array  # shape: (2,)


class ObsFeatures(NamedTuple):
    position: jax.Array


class ResetOptions(NamedTuple):
    pass


class ConveyorWorld(
    environment.Environment[EnvState, EnvParams, ObsFeatures, ResetOptions]
):
    default_params = EnvParams(max_steps_in_episode=50)
    propositions = ("parcel", "wrench", "hammer")

    # Actions: right, down, left, up
    _index_to_action = jnp.array([[1, 0], [0, -1], [-1, 0], [0, 1]], dtype=jnp.int32)

    _grid_size = 9

    _start_pos = jnp.array([0, 4], dtype=jnp.int32)

    _parcel_pos = jnp.array([[7, 1], [7, 7]], dtype=jnp.int32)
    _wrench_pos = jnp.array([[8, 1]], dtype=jnp.int32)
    _hammer_pos = jnp.array([[8, 7]], dtype=jnp.int32)

    def __init__(self, **kwargs):
        params = dataclasses.asdict(self.default_params) | kwargs
        super().__init__(
            default_params=EnvParams(**params), propositions=self.propositions
        )

    @override
    def _observation_spec(self, params: EnvParams) -> ObservationSpec:
        return ObservationSpec(position=ArraySpec(shape=(2,), dtype=jnp.float32))

    @override
    def _action_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Discrete(n=4)

    @override
    def _reset(
        self,
        key: jax.Array,
        state: EnvState | None,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        return EnvState(position=self._start_pos)

    def _get_wall_mask(self) -> jax.Array:
        """Returns a boolean mask of shape (grid_size, grid_size) where True is a wall."""
        walls = jnp.zeros((self._grid_size, self._grid_size), dtype=jnp.bool)

        # Bottom-left block: x=0..2, y=0
        walls = walls.at[0:5, 0].set(True)

        # Top-left block: x=0..2, y=8
        walls = walls.at[0:5, 8].set(True)

        # Center C-shape/Block obstacle: x=3..6, y=2..6
        walls = walls.at[4:9, 3:6].set(True)
        walls = walls.at[3, 2:7].set(True)
        walls = walls.at[3:6, 2].set(True)
        walls = walls.at[3:6, 6].set(True)

        return walls

    @override
    def _step(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        current_pos = state.position
        move_vec = self._index_to_action[action]

        # --- Conveyor Belt Logic ---
        # Belt 1 (Bottom): y=1, x=3..6, Direction: Right (+x)
        # Belt 2 (Top):    y=7, x=3..6, Direction: Right (+x)

        x, y = current_pos[0], current_pos[1]

        on_bottom_belt = (y == 1) & (x >= 3) & (x <= 5)
        on_top_belt = (y == 7) & (x >= 3) & (x <= 5)
        on_conveyor = on_bottom_belt | on_top_belt

        # Action mapping: 0=Right, 1=Down, 2=Left, 3=Up
        trying_to_move_left = action == 2

        # If on conveyor and trying to move left (against flow), movement is blocked
        move_vec = jnp.where(
            on_conveyor & trying_to_move_left,
            jnp.array([0, 0], dtype=jnp.int32),
            move_vec,
        )

        # --- Movement & Wall Collision ---
        target_pos = current_pos + move_vec

        # Clip to grid bounds
        target_pos = jnp.clip(target_pos, 0, self._grid_size - 1)

        # Check walls
        walls = self._get_wall_mask()
        is_wall = walls[target_pos[0], target_pos[1]]

        # If wall, stay in current position
        next_pos = jnp.where(is_wall, current_pos, target_pos)

        next_state = EnvState(position=next_pos)

        return (
            next_state,
            jnp.zeros((), dtype=jnp.float32),
            jnp.zeros((), dtype=jnp.bool),
            {},
        )

    @override
    def _compute_obs(self, state: EnvState, params: EnvParams) -> ObsFeatures:
        """Compute the observation for a given state."""
        return ObsFeatures(position=state.position)

    @override
    def compute_propositions(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute which proposition is currently satisfied.

        Returns an int32 array containing indices of active propositions.
        """
        pos = state.position

        # Helper to check if pos is in a list of target locations
        def is_at(targets):
            return jnp.any(jnp.all(targets == pos, axis=1))

        p_parcel = is_at(self._parcel_pos)
        p_wrench = is_at(self._wrench_pos)
        p_hammer = is_at(self._hammer_pos)

        active_props = jnp.stack([p_parcel, p_wrench, p_hammer])
        return jnp.nonzero(active_props, size=len(self.propositions), fill_value=-1)[0]

    @staticmethod
    @override
    def assignments() -> list[Assignment]:
        """Returns all possible assignments in the environment."""
        assignments = [
            Assignment(frozenset({prop})) for prop in ConveyorWorld.propositions
        ]
        assignments.append(Assignment(frozenset()))  # empty assignment
        return assignments

    @override
    def get_renderer(
        self, env_params: EnvParams, **kwargs
    ) -> "BaseRenderer[ObsFeatures, ResetOptions]":
        """Returns a renderer for the environment."""
        from .renderer import ConveyorWorldRenderer  # noqa: PLC0415

        return ConveyorWorldRenderer(
            title="ConveyorWorld",
            screen_size=600,
            grid_size=self._grid_size,
        )

    @override
    def plot_trajectories(
        self,
        trajs: EnvState,
        lengths: jax.Array,
        params: EnvParams,
        **plotting_kwargs,
    ) -> None:
        raise NotImplementedError(
            "Trajectory plotting not implemented for ConveyorWorld."
        )
