"""Plotting utilities."""

import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import numpy as np


def smooth(row, radius):
    """
    Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
    """
    y = np.ones(radius)
    z = np.ones(len(row))
    return np.convolve(row, y, "same") / np.convolve(z, y, "same")


class StaticColorAxisBBox(mpatches.FancyBboxPatch):
    def set_edgecolor(self, color):
        if hasattr(self, "_original_edgecolor"):
            return
        self._original_edgecolor = color
        self._set_edgecolor(color)  # type: ignore

    def set_linewidth(self, w):
        del w
        super().set_linewidth(1.5)


class FancyAxes(maxes.Axes):
    name = "fancy_box_axes"
    _edgecolor: str

    def __init__(self, *args, **kwargs):
        self._edgecolor = kwargs.pop("edgecolor", None)
        self._linewidth = kwargs.pop("linewidth", None)
        super().__init__(*args, **kwargs)

    def _gen_axes_patch(self):
        return StaticColorAxisBBox(
            (0, 0),
            1.0,
            1.0,
            boxstyle="round, rounding_size=0.06, pad=0",
            edgecolor=self._edgecolor,
            linewidth=5,
        )
