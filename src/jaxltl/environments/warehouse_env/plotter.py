"""Plotting utilities for WarehouseEnv environment."""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

from jaxltl.environments.zone_env_nm.plotter import FancyAxes

# Color definitions matching the renderer
_REGION_A_COLOR = "#ef4444"  # red-500
_REGION_B_COLOR = "#22c55e"  # green-500
_DOOR_COLOR = "#a855f7"  # purple-500
_VASE_COLOR = "#eab308"  # yellow-500
_CRATE_COLOR = "#8b4513"  # saddlebrown
_AGENT_COLOR = "#3b82f6"  # blue-500
_PATH_COLOR = "#22c55e"  # green for path
_CARRYING_VASE_PATH_COLOR = "#fbbf24"  # amber-400 (lighter yellow)
_CARRYING_CRATE_PATH_COLOR = "#a0522d"  # sienna (lighter brown)


def draw_region(
    ax: Axes,
    region: tuple[float, float, float, float],
    color: str,
    alpha: float = 0.3,
):
    """Draw a rectangular region.

    Args:
        ax: Matplotlib axes
        region: (x_min, x_max, y_min, y_max)
        color: Fill color
        alpha: Transparency
    """
    x_min, x_max, y_min, y_max = region
    width = x_max - x_min
    height = y_max - y_min
    rect = mpatches.FancyBboxPatch(
        (x_min, y_min),
        width,
        height,
        boxstyle="round,rounding_size=0.15",
        fc=to_rgba(color, alpha),
        ec=color,
        linewidth=1.5,
    )
    ax.add_patch(rect)


def draw_object(
    ax: Axes,
    center: tuple[float, float],
    color: str,
    radius: float = 0.2,
    alpha: float = 0.8,
):
    """Draw a circular object (vase or crate)."""
    circ = mpatches.Circle(
        center, radius, fc=to_rgba(color, alpha), ec=color, linewidth=1.5
    )
    ax.add_patch(circ)


def draw_pickup_marker(
    ax: Axes,
    position: tuple[float, float],
    color: str,
    size: float = 0.15,
):
    """Draw an upward-pointing triangle marker for pickup events."""
    triangle = mpatches.Polygon(
        [
            (position[0], position[1] + size),
            (position[0] - size * 0.7, position[1] - size * 0.5),
            (position[0] + size * 0.7, position[1] - size * 0.5),
        ],
        fc=to_rgba(color, 0.9),
        ec="black",
        linewidth=1.5,
        zorder=15,
    )
    ax.add_patch(triangle)


def draw_drop_marker(
    ax: Axes,
    position: tuple[float, float],
    color: str,
    size: float = 0.15,
):
    """Draw a downward-pointing triangle marker for drop events."""
    triangle = mpatches.Polygon(
        [
            (position[0], position[1] - size),
            (position[0] - size * 0.7, position[1] + size * 0.5),
            (position[0] + size * 0.7, position[1] + size * 0.5),
        ],
        fc=to_rgba(color, 0.9),
        ec="black",
        linewidth=1.5,
        zorder=15,
    )
    ax.add_patch(triangle)


def draw_start_marker(
    ax: Axes,
    center: tuple[float, float],
    color: str = "orange",
    size: float = 0.18,
):
    """Draw a diamond marker for the starting position."""
    diamond = mpatches.Polygon(
        [
            (center[0], center[1] + size),
            (center[0] + size, center[1]),
            (center[0], center[1] - size),
            (center[0] - size, center[1]),
        ],
        fc=to_rgba(color, 0.9),
        ec="black",
        linewidth=1.5,
        zorder=20,
    )
    ax.add_patch(diamond)


def draw_path_segment(
    ax: Axes,
    points: list[tuple[float, float]],
    color: str,
    linewidth: float = 3,
    alpha: float = 0.8,
):
    """Draw a path segment."""
    if len(points) < 2:  # noqa: PLR2004
        return
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    ax.plot(
        x, y, color=to_rgba(color, alpha), linewidth=linewidth, solid_capstyle="round"
    )


def draw_trajectory(
    ax: Axes,
    positions: list[tuple[float, float]],
    carrying_vase: list[bool],
    carrying_crate: list[bool],
    pickup_events: list[dict],
    drop_events: list[dict],
    linewidth: float = 3,
):
    """Draw a trajectory with colored segments based on carrying status.

    Args:
        ax: Matplotlib axes
        positions: List of (x, y) positions
        carrying_vase: List of booleans indicating if carrying a vase at each step
        carrying_crate: List of booleans indicating if carrying a crate at each step
        pickup_events: List of dicts with 'position' and 'type' ('vase' or 'crate')
        drop_events: List of dicts with 'position' and 'type' ('vase' or 'crate')
        linewidth: Line width for the path
    """
    if len(positions) < 2:  # noqa: PLR2004
        return

    # Draw path segments with different colors based on carrying status
    current_segment = [positions[0]]
    current_carrying_vase = carrying_vase[0]
    current_carrying_crate = carrying_crate[0]

    def get_color(cv: bool, cc: bool) -> str:
        if cv and cc:
            return "#ff6b6b"  # Both - use a distinct color
        elif cv:
            return _CARRYING_VASE_PATH_COLOR
        elif cc:
            return _CARRYING_CRATE_PATH_COLOR
        else:
            return _PATH_COLOR

    for i in range(1, len(positions)):
        cv, cc = carrying_vase[i], carrying_crate[i]
        if cv != current_carrying_vase or cc != current_carrying_crate:
            # Carrying status changed - draw current segment and start new one
            current_segment.append(positions[i])
            draw_path_segment(
                ax,
                current_segment,
                get_color(current_carrying_vase, current_carrying_crate),
                linewidth,
            )
            current_segment = [positions[i]]
            current_carrying_vase = cv
            current_carrying_crate = cc
        else:
            current_segment.append(positions[i])

    # Draw final segment
    if len(current_segment) >= 2:  # noqa: PLR2004
        draw_path_segment(
            ax,
            current_segment,
            get_color(current_carrying_vase, current_carrying_crate),
            linewidth,
        )

    # Draw pickup markers
    for event in pickup_events:
        color = _VASE_COLOR if event["type"] == "vase" else _CRATE_COLOR
        draw_pickup_marker(ax, event["position"], color)

    # Draw drop markers
    for event in drop_events:
        color = _VASE_COLOR if event["type"] == "vase" else _CRATE_COLOR
        draw_drop_marker(ax, event["position"], color)


def draw_checkerboard(
    ax: Axes, world_size: float, grid_size: float = 0.6, margin: float = 0.1
):
    """Draw a subtle checkerboard pattern as background.

    Args:
        ax: Matplotlib axes
        world_size: Size of the world
        grid_size: Size of each checkerboard cell
        margin: Extra margin to cover beyond the world bounds
    """
    color_light = "#fafafa"  # Very light gray
    color_dark = "#f0f0f0"  # Slightly darker gray

    # Start before origin and extend past world_size to cover margins
    start = -grid_size
    end = world_size + grid_size + margin
    num_cells = int((end - start) / grid_size) + 1

    for i in range(num_cells):
        for j in range(num_cells):
            x = start + i * grid_size
            y = start + j * grid_size
            # Offset the pattern calculation to maintain consistency
            color = color_light if (i + j) % 2 == 0 else color_dark
            rect = mpatches.Rectangle(
                (x, y),
                grid_size,
                grid_size,
                fc=color,
                ec="none",
                zorder=0,
            )
            ax.add_patch(rect)


def setup_axis(ax: Axes, world_size: float):
    """Set up the axis limits, aspect ratio, grid, and hide axes."""
    margin = 0.1
    ax.set_xlim(-margin, world_size + margin)
    ax.set_ylim(-margin, world_size + margin)
    ax.set_aspect("equal")

    # # Add subtle grid
    # ax.grid(
    #     True, which="both", color="gray", linestyle="dashed", linewidth=0.5, alpha=0.4
    # )
    ax.set_axisbelow(True)

    # Hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Hide ticks and labels
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )


def draw_single_trajectory(  # noqa: PLR0913
    ax: Axes,
    positions: list[tuple[float, float]],
    initial_vase_positions: list[tuple[float, float]],
    initial_crate_positions: list[tuple[float, float]],
    carrying_vase_idx: list[int],
    carrying_crate_idx: list[int],
    region_a: tuple[float, float, float, float],
    region_b: tuple[float, float, float, float],
    door_region: tuple[float, float, float, float],
    world_size: float,
    pickup_radius: float = 0.2,
):
    """Draw a single trajectory on the given axes.

    Args:
        ax: Matplotlib axes
        positions: List of agent (x, y) positions
        initial_vase_positions: Initial positions of vases
        initial_crate_positions: Initial positions of crates
        carrying_vase_idx: List of carried vase indices (-1 if not carrying)
        carrying_crate_idx: List of carried crate indices (-1 if not carrying)
        region_a: Region A bounds (x_min, x_max, y_min, y_max)
        region_b: Region B bounds
        door_region: Door region bounds
        world_size: Size of the world
        pickup_radius: Radius of objects
    """
    setup_axis(ax, world_size)

    # Draw checkerboard background
    draw_checkerboard(ax, world_size)

    # Draw regions
    draw_region(ax, region_a, _REGION_A_COLOR, alpha=0.25)
    draw_region(ax, region_b, _REGION_B_COLOR, alpha=0.25)
    draw_region(ax, door_region, _DOOR_COLOR, alpha=0.25)

    # Draw initial object positions
    for pos in initial_vase_positions:
        draw_object(ax, pos, _VASE_COLOR, radius=pickup_radius, alpha=0.5)
    for pos in initial_crate_positions:
        draw_object(ax, pos, _CRATE_COLOR, radius=pickup_radius, alpha=0.5)

    # Compute carrying status and events
    carrying_vase = [idx != -1 for idx in carrying_vase_idx]
    carrying_crate = [idx != -1 for idx in carrying_crate_idx]

    # Detect pickup and drop events
    pickup_events = []
    drop_events = []

    for i in range(1, len(positions)):
        # Vase pickup/drop
        prev_vase, curr_vase = carrying_vase_idx[i - 1], carrying_vase_idx[i]
        if prev_vase == -1 and curr_vase != -1:
            pickup_events.append({"position": positions[i], "type": "vase"})
        elif prev_vase != -1 and curr_vase == -1:
            drop_events.append({"position": positions[i], "type": "vase"})

        # Crate pickup/drop
        prev_crate, curr_crate = carrying_crate_idx[i - 1], carrying_crate_idx[i]
        if prev_crate == -1 and curr_crate != -1:
            pickup_events.append({"position": positions[i], "type": "crate"})
        elif prev_crate != -1 and curr_crate == -1:
            drop_events.append({"position": positions[i], "type": "crate"})

    # Draw trajectory
    draw_trajectory(
        ax, positions, carrying_vase, carrying_crate, pickup_events, drop_events
    )

    # Draw start marker
    if positions:
        draw_start_marker(ax, positions[0])


def draw_trajectories(  # noqa: PLR0913
    positions: list[list[tuple[float, float]]],
    initial_vase_positions: list[list[tuple[float, float]]],
    initial_crate_positions: list[list[tuple[float, float]]],
    carrying_vase_idx: list[list[int]],
    carrying_crate_idx: list[list[int]],
    region_a: tuple[float, float, float, float],
    region_b: tuple[float, float, float, float],
    door_region: tuple[float, float, float, float],
    world_size: float,
    pickup_radius: float,
    num_cols: int,
    num_rows: int,
):
    """Draw multiple trajectories in a grid layout.

    Args:
        positions: List of trajectories, each a list of (x, y) positions
        initial_vase_positions: Initial vase positions for each trajectory
        initial_crate_positions: Initial crate positions for each trajectory
        carrying_vase_idx: Carrying vase index for each step of each trajectory
        carrying_crate_idx: Carrying crate index for each step of each trajectory
        region_a: Region A bounds
        region_b: Region B bounds
        door_region: Door region bounds
        world_size: Size of the world
        pickup_radius: Radius of objects
        num_cols: Number of columns in the grid
        num_rows: Number of rows in the grid
    """
    num_trajs = len(positions)
    if num_cols * num_rows < num_trajs:
        raise ValueError(
            f"Grid size {num_rows}x{num_cols} too small for {num_trajs} trajectories"
        )

    fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows))

    for i in range(num_trajs):
        ax = fig.add_subplot(
            num_rows,
            num_cols,
            i + 1,
            axes_class=FancyAxes,
            edgecolor="gray",
            linewidth=0.5,
        )

        draw_single_trajectory(
            ax,
            positions[i],
            initial_vase_positions[i],
            initial_crate_positions[i],
            carrying_vase_idx[i],
            carrying_crate_idx[i],
            region_a,
            region_b,
            door_region,
            world_size,
            pickup_radius,
        )

    # Add legend
    legend_elements = [
        mpatches.Patch(color=_REGION_A_COLOR, alpha=0.3, label="Region A"),
        mpatches.Patch(color=_REGION_B_COLOR, alpha=0.3, label="Region B"),
        mpatches.Patch(color=_DOOR_COLOR, alpha=0.3, label="Door"),
        mpatches.Patch(color=_PATH_COLOR, label="Path (empty hands)"),
        mpatches.Patch(color=_CARRYING_VASE_PATH_COLOR, label="Path (carrying vase)"),
        mpatches.Patch(color=_CARRYING_CRATE_PATH_COLOR, label="Path (carrying crate)"),
        mlines.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=_VASE_COLOR,
            markeredgecolor="black",
            markersize=10,
            label="Vase pickup",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor=_VASE_COLOR,
            markeredgecolor="black",
            markersize=10,
            label="Vase drop",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=_CRATE_COLOR,
            markeredgecolor="black",
            markersize=10,
            label="Crate pickup",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor=_CRATE_COLOR,
            markeredgecolor="black",
            markersize=10,
            label="Crate drop",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="d",
            color="w",
            markerfacecolor="orange",
            markeredgecolor="black",
            markersize=10,
            label="Start",
        ),
    ]

    plt.tight_layout(pad=4)
    plt.savefig("traj.pdf", dpi=300)
    plt.show()
