"""MuJoCo offscreen rendering blitted into the shared pygame render loop."""

import os
from typing import override

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pygame

from jaxolotl.environments.franka_zone_env.franka_zone_env import (
    ZONE_RGBA,
    EnvParams,
    EnvState,
    FrankaZoneEnv,
    ObsFeatures,
    ResetOptions,
    build_model,
)
from jaxolotl.environments.renderer.renderer import ContinuousTimeRenderer


def _add_zone_geom(
    scene: mujoco.MjvScene,  # type: ignore
    center: np.ndarray,
    radius: float,
    rgba: np.ndarray,
    label: str,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(  # type: ignore
        scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # type: ignore
        size=np.array([radius, 0.0, 0.0]),
        pos=np.asarray(center, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.geoms[scene.ngeom].label = label
    scene.ngeom += 1


class Renderer(ContinuousTimeRenderer[ObsFeatures, ResetOptions]):
    def __init__(
        self,
        env: FrankaZoneEnv,
        params: EnvParams,
        screen_size: int = 800,
        camera_elevation: float = -20.0,
        camera_azimuth: float = 135.0,
    ):
        # SDL reads this before creating its window. It prevents a blank pygame
        # window when SDL and MuJoCo both use OpenGL.
        os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        super().__init__("Franka Zone Environment", screen_size)
        self._env = env
        self._params = params
        self._model = build_model()
        self._model.vis.global_.offwidth = max(
            self._model.vis.global_.offwidth, screen_size
        )
        self._model.vis.global_.offheight = max(
            self._model.vis.global_.offheight, screen_size
        )
        self._data = mujoco.MjData(self._model)  # type: ignore
        try:
            self._renderer = mujoco.Renderer(
                self._model, height=screen_size, width=screen_size
            )
        except mujoco.FatalError as error:  # type: ignore
            raise RuntimeError(
                "MuJoCo could not create a rendering context. Its backend is fixed "
                "when mujoco is imported; export MUJOCO_GL=egl before starting "
                "the process on a headless machine."
            ) from error

        self._camera = mujoco.MjvCamera()  # type: ignore
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE  # type: ignore
        self._camera.lookat[:] = (0.4, 0.0, 0.35)
        self._camera.distance = 1.65
        self._camera.azimuth = camera_azimuth
        self._camera.elevation = camera_elevation
        self._legend_font = pygame.font.Font(None, 20)

    def _draw_legend(self, reached: np.ndarray) -> None:
        rows_per_column = 8
        for index, name in enumerate(self._env.propositions):
            column, row = divmod(index, rows_per_column)
            x = 12 + column * 120
            y = 12 + row * 23
            rgb = tuple((ZONE_RGBA[index, :3] * 255).astype(np.uint8))
            pygame.draw.circle(self._screen, rgb, (x + 7, y + 7), 6)
            if reached[index]:
                pygame.draw.circle(self._screen, (255, 255, 255), (x + 7, y + 7), 8, 2)
            text = self._legend_font.render(name, True, (245, 245, 245))
            self._screen.blit(text, (x + 19, y - 1))

    @override
    def render_with_interpolation(
        self,
        state: EnvState,
        previous_state: EnvState,  # noqa: ARG002
        obs: ObsFeatures | None,  # noqa: ARG002
        alpha: float,  # noqa: ARG002
    ):
        self._data.qpos[:] = np.asarray(state.data.qpos)
        mujoco.mj_forward(self._model, self._data)  # type: ignore
        self._renderer.update_scene(self._data, camera=self._camera)

        grasp_center = np.asarray(state.ee_pose[:3, 3])
        centers = np.asarray(state.zone_centers)
        reached = np.linalg.norm(centers - grasp_center, axis=-1) < (
            self._params.zone_radius
        )
        for index, (center, is_reached) in enumerate(
            zip(centers, reached, strict=True)
        ):
            rgba = ZONE_RGBA[index].copy()
            rgba[3] = 0.62 if is_reached else rgba[3]
            _add_zone_geom(
                self._renderer.scene,
                center,
                self._params.zone_radius,
                rgba,
                "",
            )

        # pygame uses (x, y); rendered RGB arrays use (row, column).
        frame = pygame.surfarray.make_surface(self._renderer.render().swapaxes(0, 1))
        self._screen.blit(frame, (0, 0))
        self._draw_legend(reached)
        pygame.display.flip()

    @override
    def _format_obs(self, obs: ObsFeatures) -> str:
        with jnp.printoptions(precision=2, suppress=True):
            return (
                f"Type: {type(obs).__name__}\n  arm: {obs.arm}\n  zones:\n{obs.zones}\n"
            )

    @override
    def get_action(self, keys: pygame.key.ScancodeWrapper) -> jax.Array:
        """W/S, E/D, R/F translate; T/G, Y/H, U/J rotate in base axes."""
        return jnp.asarray(
            [
                float(keys[pygame.K_w]) - float(keys[pygame.K_s]),
                float(keys[pygame.K_e]) - float(keys[pygame.K_d]),
                float(keys[pygame.K_r]) - float(keys[pygame.K_f]),
                float(keys[pygame.K_t]) - float(keys[pygame.K_g]),
                float(keys[pygame.K_y]) - float(keys[pygame.K_h]),
                float(keys[pygame.K_u]) - float(keys[pygame.K_j]),
            ],
            dtype=jnp.float32,
        )

    @override
    def close(self):
        self._renderer.close()
        super().close()
