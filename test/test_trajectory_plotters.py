from unittest.mock import Mock

import matplotlib.pyplot as plt
import pytest

from jaxltl.environments.warehouse_env.plotter import (
    draw_trajectories as draw_warehouse_trajectories,
)
from jaxltl.environments.zone_env_nm.plotter import (
    draw_trajectories as draw_zone_trajectories,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_zone_trajectory_saving_is_optional(monkeypatch, tmp_path):
    savefig = Mock()
    monkeypatch.setattr(plt, "savefig", savefig)
    monkeypatch.setattr(plt, "show", Mock())

    kwargs = {
        "zone_positions": [[(0.0, 0.0)]],
        "colors": [["red"]],
        "paths": [[(0.0, 0.0), (1.0, 1.0)]],
        "num_cols": 1,
        "num_rows": 1,
    }
    draw_zone_trajectories(**kwargs)
    savefig.assert_not_called()

    save_path = tmp_path / "zone-trajectories.pdf"
    draw_zone_trajectories(**kwargs, save_path=str(save_path))
    savefig.assert_called_once_with(str(save_path), dpi=300, format="pdf")


def test_warehouse_trajectory_saving_is_optional(monkeypatch, tmp_path):
    savefig = Mock()
    monkeypatch.setattr(plt, "savefig", savefig)
    monkeypatch.setattr(plt, "show", Mock())

    kwargs = {
        "positions": [[(0.0, 0.0), (1.0, 1.0)]],
        "initial_vase_positions": [[]],
        "initial_crate_positions": [[]],
        "carrying_vase_idx": [[-1, -1]],
        "carrying_crate_idx": [[-1, -1]],
        "region_a": (0.0, 1.0, 0.0, 1.0),
        "region_b": (1.0, 2.0, 1.0, 2.0),
        "door_region": (0.0, 0.5, 0.0, 0.5),
        "world_size": 3.0,
        "pickup_radius": 0.2,
        "num_cols": 1,
        "num_rows": 1,
    }
    draw_warehouse_trajectories(**kwargs)
    savefig.assert_not_called()

    save_path = tmp_path / "warehouse-trajectories.pdf"
    draw_warehouse_trajectories(**kwargs, save_path=str(save_path))
    savefig.assert_called_once_with(str(save_path), dpi=300, format="pdf")
