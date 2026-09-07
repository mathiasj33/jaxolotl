from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from jaxolotl.environments.conveyor_world.conveyor_world import EnvState, ObsFeatures
from jaxolotl.environments.renderer.renderer import DiscreteTimeRenderer


class ConveyorWorldRenderer(DiscreteTimeRenderer[ObsFeatures]):
    """Renderer for the ConveyorWorld environment."""

    def __init__(
        self,
        title: str = "ConveyorWorld",
        screen_size: int = 600,
        grid_size: int = 9,
    ):
        super().__init__(title=title, screen_size=screen_size)
        self.grid_size = grid_size
        self.cell_size = screen_size // self.grid_size

        # Colors
        self.bg_color = (250, 250, 250)
        self.wall_color = (60, 60, 60)
        self.conveyor_color = (220, 230, 255)  # Light blue
        self.grid_line_color = (200, 200, 200)

        # Item Colors
        self.item_colors = {
            "P": (255, 165, 0),  # Orange for Parcel
            "W": (100, 100, 100),  # Grey for Wrench
            "H": (139, 69, 19),  # Brown for Hammer
        }
        self.agent_color = (0, 200, 0)  # Green

        # Fonts
        self.font = pygame.font.Font(None, int(self.cell_size * 0.6))
        self.arrow_font = pygame.font.Font(None, int(self.cell_size * 0.8))

        # Precompute Static Map Elements
        self._walls = self._create_wall_mask()
        self._conveyors = self._create_conveyor_mask()  # Stores direction strings
        self._items = self._create_item_map()

        # Canvas for double buffering
        self._canvas = pygame.Surface((self.screen_size, self.screen_size))

    def _create_wall_mask(self) -> np.ndarray:
        """Replicates the wall logic from ConveyorWorld using numpy."""
        walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Bottom-left block
        walls[0:5, 0] = True
        # Top-left block
        walls[0:5, 8] = True
        # Center C-shape
        walls[4:9, 3:6] = True
        walls[3, 2:7] = True
        walls[3:6, 2] = True
        walls[3:6, 6] = True

        return walls

    def _create_conveyor_mask(self) -> np.ndarray:
        """Returns a grid where cells contain direction strings (e.g., '>>')."""
        conveyors = np.full((self.grid_size, self.grid_size), None, dtype=object)

        # Bottom Belt (y=1, x=3..5) -> Right
        conveyors[3:6, 1] = ">>"
        # Top Belt (y=7, x=3..5) -> Right
        conveyors[3:6, 7] = ">>"

        return conveyors

    def _create_item_map(self) -> dict:
        """Returns a dict mapping (x,y) tuples to item characters."""
        items = {}
        # Coordinates taken from ConveyorWorld definition
        items[(7, 1)] = "P"
        items[(7, 7)] = "P"
        items[(8, 1)] = "W"
        items[(8, 7)] = "H"
        return items

    def _get_rect(self, x: int, y: int) -> pygame.Rect:
        """Converts Env (x, y) to PyGame Rect.

        Env: (0,0) is Bottom-Left.
        PyGame: (0,0) is Top-Left.
        """
        col = x
        row = (self.grid_size - 1) - y
        return pygame.Rect(
            col * self.cell_size,
            row * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    @override
    def render(
        self,
        state: EnvState,
        _,
    ):
        """Renders the environment state."""
        # 1. Clear Canvas
        self._canvas.fill(self.bg_color)

        # 2. Draw Static Elements (Grid, Walls, Conveyors, Items)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = self._get_rect(x, y)

                # Draw Walls
                if self._walls[x, y]:
                    pygame.draw.rect(self._canvas, self.wall_color, rect)
                    continue  # Skip drawing anything else on a wall

                # Draw Conveyors
                belt_dir = self._conveyors[x, y]
                if belt_dir:
                    pygame.draw.rect(self._canvas, self.conveyor_color, rect)
                    # Render arrows
                    text_surf = self.arrow_font.render(belt_dir, True, (50, 100, 200))
                    text_rect = text_surf.get_rect(center=rect.center)
                    self._canvas.blit(text_surf, text_rect)

                # Draw Grid Lines
                pygame.draw.rect(self._canvas, self.grid_line_color, rect, 1)

                # Draw Items
                if (x, y) in self._items:
                    item_char = self._items[(x, y)]
                    color = self.item_colors.get(item_char, (0, 0, 0))

                    # Draw a small circle background for the item
                    center = rect.center
                    radius = int(self.cell_size * 0.35)
                    pygame.draw.circle(self._canvas, (240, 240, 240), center, radius)
                    pygame.draw.circle(self._canvas, color, center, radius, 2)

                    # Draw Text
                    text_surf = self.font.render(item_char, True, color)
                    self._canvas.blit(text_surf, text_surf.get_rect(center=center))

        # 3. Draw Agent
        agent_pos = np.array(state.position, dtype=int)
        ax, ay = agent_pos[0], agent_pos[1]

        agent_rect = self._get_rect(ax, ay)
        center = agent_rect.center
        radius = int(self.cell_size * 0.4)

        # Draw Agent Body
        pygame.draw.circle(self._canvas, self.agent_color, center, radius)
        # Draw Agent Border
        pygame.draw.circle(self._canvas, (0, 100, 0), center, radius, 2)

        # 4. Blit to screen
        self._screen.blit(self._canvas, (0, 0))
        pygame.display.flip()

    @override
    def get_action(self, key: int) -> jax.Array:
        """Gets an action from user input.

        Mappings:
        D -> Right (0)
        S -> Down (1)
        A -> Left (2)
        W -> Up (3)
        """
        mapping = {
            pygame.K_d: 0,
            pygame.K_s: 1,
            pygame.K_a: 2,
            pygame.K_w: 3,
        }
        if key not in mapping:
            raise ValueError(f"Invalid key pressed: {key}")
        return jnp.array(mapping[key], dtype=jnp.int32)
