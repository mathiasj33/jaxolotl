"""A 2D renderer for the warehouse environment based on pygame."""

import math
from functools import partial
from typing import override

import jax
import jax.numpy as jnp
import pygame
from pygame import gfxdraw

from jaxolotl.environments.renderer.renderer import ContinuousTimeRenderer
from jaxolotl.environments.warehouse_env.warehouse_env import (
    EnvState,
    ObsFeatures,
    ResetOptions,
    WarehouseParams,
)

_DO_NOTHING_INDEX = 0
_PICKUP_VASE_INDEX = 1
_PICKUP_CRATE_INDEX = 2
_DROP_VASE_INDEX = 3
_DROP_CRATE_INDEX = 4


class Renderer(ContinuousTimeRenderer[ObsFeatures, ResetOptions]):
    def __init__(
        self,
        params: WarehouseParams,
        screen_size: int = 800,
        grid_size: int = 50,
        show_lidar: bool = False,
    ):
        super().__init__("Warehouse Environment", screen_size)

        self._params = params
        self._screen_size = screen_size
        self.draw_lidar = show_lidar

        self._background = pygame.Surface(self._screen.get_size())
        self._lidar_surface = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)

        self._world_to_screen_scale = screen_size / params.world_size
        self._agent_radius_px = int(0.1 * self._world_to_screen_scale)
        self._object_radius_px = int(params.pickup_radius * self._world_to_screen_scale)

        # Checkerboard background
        self.grid_size = grid_size
        self._grid_color_1 = (248, 250, 252)
        self._grid_color_2 = (241, 245, 249)
        self._render_background()

        # Regions
        self._region_a_color = (239, 68, 68, 150)  # red-500
        self._region_b_color = (34, 197, 94, 150)  # green-500
        self._door_color = (168, 85, 247, 150)  # purple-500
        self._render_regions()

        # Agent color
        self._agent_color = (59, 130, 246)  # blue-500
        self._agent_heading_color = (59, 130, 246, 180)  # blue-500 with alpha

        # Colors
        self._vase_color = (234, 179, 8, 220)  # yellow-500
        self._crate_color = (139, 69, 19, 220)  # saddlebrown

    def _render_background(self):
        """Draw checkerboard background."""
        self._background.fill(self._grid_color_1)
        for y in range(0, self._screen_size, self.grid_size):
            for x in range(0, self._screen_size, self.grid_size):
                if (y // self.grid_size + x // self.grid_size) % 2 == 1:
                    rect = pygame.Rect(x, y, self.grid_size, self.grid_size)
                    self._background.fill(self._grid_color_2, rect)

    def _render_regions(self):
        self._draw_rect_region(
            self._params.region_a,
            self._region_a_color,
            border_top_right_radius=10,
            border_bottom_right_radius=10,
        )
        self._draw_rect_region(
            self._params.region_b, self._region_b_color, border_top_left_radius=10
        )
        self._draw_rect_region(
            self._params.door_region, self._door_color, border_bottom_left_radius=10
        )

    def _draw_rect_region(self, region_coords, color, **kwargs):
        x_min, x_max, y_min, y_max = region_coords
        screen_pos = self._world_to_screen(jnp.array([[x_min, y_max], [x_max, y_min]]))
        top_left = screen_pos[0]
        bottom_right = screen_pos[1]
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        rect_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        rect = pygame.Rect(0, 0, width, height)
        pygame.draw.rect(rect_surface, color, rect, **kwargs)
        self._background.blit(rect_surface, (top_left[0], top_left[1]))

    def _render_objects(self, state: EnvState):
        # Vases
        for i in range(self._params.num_vases):
            if state.vase_available[i]:
                pos = self._world_to_screen_single(state.vase_positions[i])
                self._draw_circle(
                    self._screen, self._vase_color, pos, self._object_radius_px
                )

        # Crates
        for i in range(self._params.num_crates):
            if state.crate_available[i]:
                pos = self._world_to_screen_single(state.crate_positions[i])
                self._draw_circle(
                    self._screen, self._crate_color, pos, self._object_radius_px
                )

    def _draw_circle(self, surface, color, position, radius):
        """Draw an anti-aliased filled circle."""
        gfxdraw.aacircle(surface, int(position[0]), int(position[1]), radius, color)
        gfxdraw.filled_circle(
            surface, int(position[0]), int(position[1]), radius, color
        )

    @partial(jax.jit, static_argnames=("self",))
    def _world_to_screen(self, pos: jax.Array) -> jax.Array:
        """Convert world coordinates to screen coordinates."""
        pos = pos * self._world_to_screen_scale
        pos = pos.at[:, 1].set(self._screen_size - pos[:, 1])
        return pos.astype(jnp.int32)

    def _world_to_screen_single(self, pos: jax.Array) -> jax.Array:
        """Convert a single world coordinate to screen coordinates."""
        pos = pos * self._world_to_screen_scale
        pos = pos.at[1].set(self._screen_size - pos[1])
        return pos.astype(jnp.int32)

    @override
    def render_with_interpolation(
        self,
        state: EnvState,
        previous_state: EnvState,
        obs: ObsFeatures | None,
        alpha: float,
    ):
        """Render the environment state."""
        self._screen.blit(self._background, (0, 0))

        self._render_objects(state)

        # Interpolation for smooth rendering
        interpolated_position = (
            previous_state.position * (1.0 - alpha) + state.position * alpha
        )
        angle_diff = (state.angle - previous_state.angle + jnp.pi) % (
            2 * jnp.pi
        ) - jnp.pi
        interpolated_angle = previous_state.angle + alpha * angle_diff

        self._draw_agent(interpolated_position, interpolated_angle, state)
        if self.draw_lidar and obs:
            self._draw_lidar(interpolated_position, obs, state)

        pygame.display.flip()

    @override
    def _format_obs(self, obs: ObsFeatures) -> str:
        """Neatly formats the observations and propositions into a single string."""
        if not isinstance(obs, ObsFeatures):
            return ""

        output = []
        output.append(f"Type: {type(obs).__name__}\n")
        for field, value in obs._asdict().items():
            if not isinstance(value, jax.Array):
                output.append(f"  {field}: {value}\n")
                continue

            if value.ndim == 2:
                output.append(f"  {field}: shape {value.shape}\n")
                if field == "lidar":
                    output.append(self._format_lidar_field(value))
                else:
                    output.append(f"    {value}\n")
            else:
                with jnp.printoptions(precision=2, suppress=True):
                    output.append(f"  {field}: {value}\n")
        return "".join(output)

    def _format_lidar_field(self, value: jax.Array) -> str:
        lines = []
        num_objects, num_bins = value.shape

        # Header
        header_parts = [f"{'Bin':>3}"]
        header_parts.extend([f"{f'O{i}':>5}" for i in range(num_objects)])
        lines.append(f"    {' | '.join(header_parts)}\n")

        # Separator
        separator_parts = [f"{'-' * 3}"]
        separator_parts.extend([f"{'-' * 5}" for _ in range(num_objects)])
        lines.append(f"    {'-+-'.join(separator_parts)}\n")

        # Data rows
        for i in range(num_bins):
            row_parts = [f"{i:3d}"]
            row_parts.extend([f"{value[j, i]:5.2f}" for j in range(num_objects)])
            lines.append(f"    {' | '.join(row_parts)}\n")

        return "".join(lines)

    def _draw_agent(self, position: jax.Array, angle: jax.Array, state: EnvState):
        # Draw agent heading as a rectangle
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        rect_w = 0.02
        rect_l = 0.2
        corners = jnp.array(
            [
                [0, -rect_w / 2],
                [rect_l, -rect_w / 2],
                [rect_l, rect_w / 2],
                [0, rect_w / 2],
            ]
        )

        # Rotate and translate corners
        rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotated_corners = jnp.dot(corners, rotation_matrix.T)
        translated_corners = rotated_corners + position

        agent_and_corners = jnp.vstack([position, translated_corners])
        screen_positions = self._world_to_screen(agent_and_corners)
        agent_pos = screen_positions[0]
        self._draw_circle(
            self._screen, self._agent_color, agent_pos, self._agent_radius_px
        )
        corners = screen_positions[1:]
        gfxdraw.filled_polygon(self._screen, corners, self._agent_heading_color)
        gfxdraw.aapolygon(self._screen, corners, self._agent_heading_color)

        # Draw carried object
        radius = self._object_radius_px // 2
        if state.carrying_vase_idx != -1:
            self._draw_circle(self._screen, self._vase_color, agent_pos, radius - 3)
        if state.carrying_crate_idx != -1:
            self._draw_circle(self._screen, self._crate_color, agent_pos, radius - 5)

    def _draw_lidar(
        self,
        position: jax.Array,
        obs: ObsFeatures,
        state: EnvState,
    ):
        self._lidar_surface.fill((0, 0, 0, 0))
        points = self._compute_lidar_points(position, obs, state)  # (C, num_bins, 3)
        screen_pos = self._world_to_screen(points.reshape(-1, 3)[:, :2]).reshape(
            points.shape[0], points.shape[1], 2
        )
        points = points.at[:, :, :2].set(screen_pos)

        colors = [self._vase_color, self._crate_color]

        for color, lidar_points in zip(colors, points, strict=True):
            for point in lidar_points:
                pos = point[:2]
                strength = point[2]
                if strength > 0.0:
                    exp_strength = math.exp(-2 * (1 - strength))
                    faded_color = (
                        color[0],
                        color[1],
                        color[2],
                        int(exp_strength * 255),
                    )
                    pygame.draw.circle(self._lidar_surface, faded_color, pos, 2)
        self._screen.blit(self._lidar_surface, (0, 0))

    @staticmethod
    @jax.jit
    def _compute_lidar_points(
        position: jax.Array,
        obs: ObsFeatures,
        state: EnvState,
    ) -> jax.Array:
        # obs.lidar shape (C, num_bins)
        num_bins = obs.lidar.shape[1]
        bin_size = 2 * jnp.pi / num_bins
        bin_idx = state.angle // bin_size
        normalized_angle = bin_idx * bin_size
        points = jnp.zeros((*obs.lidar.shape, 3))
        angles = (jnp.arange(num_bins) / num_bins) * 2 * jnp.pi + normalized_angle
        angles = angles % (2 * jnp.pi)
        directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        for i, signals in enumerate(obs.lidar):
            positions = position + directions * (0.2 + i * 0.1)
            points = points.at[i].set(
                jnp.concatenate([positions, signals[:, None]], axis=-1)
            )
        return points

    @override
    def get_action(
        self, keys: pygame.key.ScancodeWrapper
    ) -> tuple[jax.Array, jax.Array]:
        """Gets an action from user input."""

        force = 0.0
        angular_velocity = 0.0
        discrete_action = _DO_NOTHING_INDEX

        if keys[pygame.K_w]:
            force = 1.0
        if keys[pygame.K_s]:
            force = -1.0
        if keys[pygame.K_a]:
            angular_velocity = 1.0
        if keys[pygame.K_d]:
            angular_velocity = -1.0

        if keys[pygame.K_e]:  # pickup vase
            discrete_action = _PICKUP_VASE_INDEX
        if keys[pygame.K_r]:  # pickup crate
            discrete_action = _PICKUP_CRATE_INDEX
        if keys[pygame.K_e] and keys[pygame.K_LSHIFT]:  # drop vase
            discrete_action = _DROP_VASE_INDEX
        if keys[pygame.K_r] and keys[pygame.K_LSHIFT]:  # drop crate
            discrete_action = _DROP_CRATE_INDEX

        cont_action = jnp.array([force, angular_velocity])
        return cont_action, jnp.array(discrete_action)
