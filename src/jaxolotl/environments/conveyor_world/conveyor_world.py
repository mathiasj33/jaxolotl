"""An implementation of the ConveyorWorld environment.

The agent must move and collect objects. It cannot move against the conveyor
belt direction.

The world consists of two mirrored corridors (top and bottom) whose entrances
are guarded by conveyor belts. Both corridors contain the same chain of shared
objects, but each ends in a different final object. Since the belts prevent
moving back out of a corridor, the agent must commit to one corridor at the
start of the episode — long before the final object is reached. Tasks are
reach-chains of length ``chain_length`` that only differ in the final object,
so solving them requires non-myopic reasoning.

``ConveyorWorld-k`` (see ``registration.py``) has chains of length ``k`` over
``k + 1`` propositions named ``"a"``, ``"b"``, ... — the first ``k - 1`` are
shared between the corridors, the last two are the corridor-specific finals.

``ConveyorWorldSimple-k`` is a simplified layout for easier exploration: the
agent starts in the middle of a single walled-in column with a length-``k``
belt directly above (flowing up) and below (flowing down), one chain item per
belt cell. The only free choice is the first move (up or down); after that the
belt prevents backing out and the agent can only advance to the corridor's
final item or stall.
"""

import dataclasses
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxolotl.environments import environment, spaces
from jaxolotl.environments.observation_spec import ArraySpec, ObservationSpec
from jaxolotl.ltl.logic.assignment import Assignment

if TYPE_CHECKING:
    from jaxolotl.environments.renderer.renderer import BaseRenderer

_MAX_CHAIN_LENGTH = 64


def _proposition_names(n: int) -> tuple[str, ...]:
    """Returns n proposition names: "a".."z", then "aa", "ab", ..."""
    letters = string.ascii_lowercase
    names = []
    for i in range(n):
        if i < len(letters):
            names.append(letters[i])
        else:
            j = i - len(letters)
            names.append(letters[j // len(letters)] + letters[j % len(letters)])
    return tuple(names)


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    chain_length: int
    simplified: bool = False


class EnvState(eqx.Module):
    position: jax.Array  # shape: (2,)


class ObsFeatures(NamedTuple):
    position: jax.Array


class ResetOptions(NamedTuple):
    pass


class ConveyorWorld(
    environment.Environment[EnvState, EnvState, EnvParams, ObsFeatures, ResetOptions]
):
    default_params = EnvParams(max_steps_in_episode=50, chain_length=2)

    # (num_propositions,) tuple of (num_locations_i, 2) int32 arrays
    _prop_positions: tuple[jax.Array, ...]
    _grid_width: int
    _grid_height: int
    _start_pos: jax.Array
    # (grid_width, grid_height) boolean wall mask
    _walls: jax.Array
    # (grid_width, grid_height) belt flow direction as an action index; -1 = no belt
    _belt_flow: jax.Array

    # Actions: right, down, left, up
    _index_to_action = jnp.array([[1, 0], [0, -1], [-1, 0], [0, 1]], dtype=jnp.int32)

    # Standard layout: corridors at y=1 (bottom) and y=7 (top); belts on x=3..5
    # move right.
    _corridor_ys = (1, 7)
    _belt_xs = (3, 4, 5)
    # Objects start at x=6+1; the first belt-free corridor cell (x=6) stays empty.
    _first_item_x = 7

    def __init__(self, chain_length: int = 2, simplified: bool = False, **kwargs):
        if not 1 <= chain_length <= _MAX_CHAIN_LENGTH:
            raise ValueError(
                f"chain_length must be in [1, {_MAX_CHAIN_LENGTH}], got {chain_length}."
            )
        values = dataclasses.asdict(self.default_params) | kwargs
        values["chain_length"] = chain_length
        values["simplified"] = simplified
        if simplified and "max_steps_in_episode" not in kwargs:
            # Scale the time limit with the corridor length (min solve = k
            # steps): at 4k steps a uniform-random committed policy completes
            # ~50% of episodes, keeping exploration difficulty flat across k.
            values["max_steps_in_episode"] = max(
                self.default_params.max_steps_in_episode, 4 * chain_length
            )
        params = EnvParams(**values)

        propositions = _proposition_names(chain_length + 1)
        if simplified:
            self._build_simplified_layout(chain_length)
        else:
            self._build_standard_layout(chain_length)

        super().__init__(default_params=params, propositions=propositions)

    def _build_standard_layout(self, chain_length: int) -> None:
        self._grid_width = self._first_item_x + chain_length
        self._grid_height = 9
        self._start_pos = jnp.array([0, 4], dtype=jnp.int32)

        walls = np.zeros((self._grid_width, self._grid_height), dtype=bool)
        # Bottom-left block: x=0..4, y=0
        walls[0:5, 0] = True
        # Top-left block: x=0..4, y=8
        walls[0:5, 8] = True
        # Center block separating the corridors: x=4.., y=3..5
        walls[4:, 3:6] = True
        walls[3, 2:7] = True
        walls[3:6, 2] = True
        walls[3:6, 6] = True
        self._walls = jnp.asarray(walls)

        belt_flow = np.full((self._grid_width, self._grid_height), -1, dtype=np.int32)
        for y in self._corridor_ys:
            belt_flow[self._belt_xs[0] : self._belt_xs[-1] + 1, y] = 0  # right
        self._belt_flow = jnp.asarray(belt_flow)

        bottom_y, top_y = self._corridor_ys
        prop_positions = []
        # Shared chain objects appear in both corridors.
        for i in range(chain_length - 1):
            x = self._first_item_x + i
            prop_positions.append(
                jnp.array([[x, bottom_y], [x, top_y]], dtype=jnp.int32)
            )
        # The final object differs between the corridors.
        final_x = self._grid_width - 1
        prop_positions.append(jnp.array([[final_x, bottom_y]], dtype=jnp.int32))
        prop_positions.append(jnp.array([[final_x, top_y]], dtype=jnp.int32))
        self._prop_positions = tuple(prop_positions)

    def _build_simplified_layout(self, chain_length: int) -> None:
        # A single walled-in column: the agent starts in the middle with a
        # length-k belt above (flowing up) and below (flowing down), one chain
        # item per belt cell. The final items sit at the two corridor ends.
        k = chain_length
        self._grid_width = 3
        self._grid_height = 2 * k + 3
        start_y = k + 1
        self._start_pos = jnp.array([1, start_y], dtype=jnp.int32)

        walls = np.ones((self._grid_width, self._grid_height), dtype=bool)
        walls[1, 1 : self._grid_height - 1] = False
        self._walls = jnp.asarray(walls)

        belt_flow = np.full((self._grid_width, self._grid_height), -1, dtype=np.int32)
        belt_flow[1, start_y + 1 : start_y + k + 1] = 3  # up
        belt_flow[1, start_y - k : start_y] = 1  # down
        self._belt_flow = jnp.asarray(belt_flow)

        prop_positions = []
        # Shared chain objects appear in both corridors, at distance i+1.
        for i in range(k - 1):
            d = i + 1
            prop_positions.append(
                jnp.array([[1, start_y - d], [1, start_y + d]], dtype=jnp.int32)
            )
        # The final object differs between the corridors.
        prop_positions.append(jnp.array([[1, start_y - k]], dtype=jnp.int32))
        prop_positions.append(jnp.array([[1, start_y + k]], dtype=jnp.int32))
        self._prop_positions = tuple(prop_positions)

    @override
    def _observation_spec(self, params: EnvParams) -> ObservationSpec:
        return ObservationSpec(position=ArraySpec(shape=(2,), dtype=jnp.float32))

    @override
    def _action_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Discrete(n=4)

    @override
    def _sample_reset(
        self,
        key: jax.Array,
        state: EnvState | None,
        params: EnvParams,
        options: ResetOptions | None = None,
    ) -> EnvState:
        return EnvState(position=self._start_pos)

    def _get_wall_mask(self) -> jax.Array:
        """Returns a boolean mask of shape (grid_width, grid_height) where True is a wall."""
        return self._walls

    def _get_belt_mask(self) -> jax.Array:
        """Returns a boolean mask of shape (grid_width, grid_height) where True is a belt."""
        return self._belt_flow >= 0

    def item_map(self) -> dict[tuple[int, int], str]:
        """Returns a mapping from grid position to the proposition placed there."""
        items = {}
        for prop, positions in zip(
            self.propositions, self._prop_positions, strict=True
        ):
            for pos in np.asarray(positions):
                items[(int(pos[0]), int(pos[1]))] = prop
        return items

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
        # A belt blocks movement against its flow direction.
        flow = self._belt_flow[current_pos[0], current_pos[1]]
        against_flow = (flow >= 0) & (action == (flow + 2) % 4)
        move_vec = jnp.where(
            against_flow,
            jnp.array([0, 0], dtype=jnp.int32),
            move_vec,
        )

        # --- Movement & Wall Collision ---
        target_pos = current_pos + move_vec

        # Clip to grid bounds
        target_pos = jnp.clip(
            target_pos,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.array([self._grid_width - 1, self._grid_height - 1], dtype=jnp.int32),
        )

        # Check walls
        is_wall = self._walls[target_pos[0], target_pos[1]]

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

        active_props = jnp.stack(
            [is_at(positions) for positions in self._prop_positions]
        )
        return jnp.nonzero(active_props, size=len(self.propositions), fill_value=-1)[0]

    def assignments(self) -> list[Assignment]:  # type: ignore[override]
        """Returns all possible assignments in the environment."""
        assignments = [Assignment(frozenset({prop})) for prop in self.propositions]
        assignments.append(Assignment(frozenset()))  # empty assignment
        return assignments

    @override
    def get_renderer(
        self, env_params: EnvParams, **kwargs
    ) -> "BaseRenderer[ObsFeatures]":
        """Returns a renderer for the environment."""
        from .renderer import ConveyorWorldRenderer  # noqa: PLC0415

        return ConveyorWorldRenderer(
            title="ConveyorWorld",
            screen_size=600,
            grid_width=self._grid_width,
            grid_height=self._grid_height,
            walls=np.asarray(self._walls),
            belt_flow=np.asarray(self._belt_flow),
            items=self.item_map(),
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
