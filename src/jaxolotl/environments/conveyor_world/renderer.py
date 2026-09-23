from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from jaxolotl.environments.conveyor_world.conveyor_world import EnvState, ObsFeatures
from jaxolotl.environments.renderer.renderer import DiscreteTimeRenderer

# Distinct colors assigned to propositions in order (cycled if exhausted).
_ITEM_COLORS = [
    (255, 165, 0),  # orange
    (100, 100, 100),  # grey
    (139, 69, 19),  # brown
    (200, 0, 200),  # magenta
    (0, 120, 200),  # blue
    (200, 0, 0),  # red
    (0, 150, 90),  # green
    (150, 120, 0),  # olive
]


class ConveyorWorldRenderer(DiscreteTimeRenderer[ObsFeatures]):
    """Renderer for the ConveyorWorld environment.

    The static layout (walls, belts, items) is provided by the environment.
    """

    # Arrow glyphs per belt flow direction (action index: right, down, left, up)
    _FLOW_ARROWS = (">>", "vv", "<<", "^^")

    def __init__(
        self,
        walls: np.ndarray,
        belt_flow: np.ndarray,
        items: dict[tuple[int, int], str],
        title: str = "ConveyorWorld",
        screen_size: int = 600,
        grid_width: int = 9,
        grid_height: int = 9,
    ):
        super().__init__(title=title, screen_size=screen_size)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = screen_size // max(grid_width, grid_height)

        # Colors
        self.bg_color = (250, 250, 250)
        self.wall_color = (60, 60, 60)
        self.conveyor_color = (220, 230, 255)  # Light blue
        self.grid_line_color = (200, 200, 200)

        # Item colors, assigned per proposition in first-seen order
        prop_names = sorted(set(items.values()))
        self.item_colors = {
            prop: _ITEM_COLORS[i % len(_ITEM_COLORS)]
            for i, prop in enumerate(prop_names)
        }
        self.agent_color = (0, 200, 0)  # Green

        # Fonts
        self.font = pygame.font.Font(None, int(self.cell_size * 0.6))
        self.arrow_font = pygame.font.Font(None, int(self.cell_size * 0.8))

        # Static map elements (provided by the environment)
        self._walls = walls
        self._belt_flow = belt_flow
        self._items = items

        # Canvas for double buffering; centered on the (square) screen
        self._canvas = pygame.Surface(
            (grid_width * self.cell_size, grid_height * self.cell_size)
        )
        self._canvas_offset = (
            (self.screen_size - grid_width * self.cell_size) // 2,
            (self.screen_size - grid_height * self.cell_size) // 2,
        )

    def _get_rect(self, x: int, y: int) -> pygame.Rect:
        """Converts Env (x, y) to PyGame Rect.

        Env: (0,0) is Bottom-Left.
        PyGame: (0,0) is Top-Left.
        """
        col = x
        row = (self.grid_height - 1) - y
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
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = self._get_rect(x, y)

                # Draw Walls
                if self._walls[x, y]:
                    pygame.draw.rect(self._canvas, self.wall_color, rect)
                    continue  # Skip drawing anything else on a wall

                # Draw Conveyors
                if self._belt_flow[x, y] >= 0:
                    pygame.draw.rect(self._canvas, self.conveyor_color, rect)
                    # Render arrows in the flow direction
                    arrow = self._FLOW_ARROWS[self._belt_flow[x, y]]
                    text_surf = self.arrow_font.render(arrow, True, (50, 100, 200))
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
        self._screen.fill(self.bg_color)
        self._screen.blit(self._canvas, self._canvas_offset)
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
