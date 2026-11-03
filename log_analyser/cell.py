"""Selecting log points around one cell of a table grid, and plotting them."""
from __future__ import annotations

import os

import numpy as np

from .axis import Axis
from .datalog import Log, LogError


def _axis_at(vals: np.ndarray, t: float) -> float:
    """Axis value at fractional node index ``t``: linear within a step, beyond the ends
    the first/last step is continued."""
    n = vals.size
    k = int(np.clip(np.floor(t), 0, n - 2))
    return float(vals[k] + (t - k) * (vals[k + 1] - vals[k]))


def _bounds(axis: Axis, value: int, radius: float) -> tuple[float, float]:
    vals = axis.values.astype(np.float64)
    idx = np.flatnonzero(axis.values == value)
    if idx.size == 0:
        raise LogError(f"{value} is not a node of {axis!r}")
    i = int(idx[0])
    return _axis_at(vals, i - radius), _axis_at(vals, i + radius)


def cell_bounds(x_axis: Axis, y_axis: Axis, x_value: int, y_value: int,
                overlap: float = 0.5) -> tuple[float, float, float, float]:
    """``(x_lo, x_hi, y_lo, y_hi)`` around node ``(x_value, y_value)``.

    ``overlap`` is the capture radius in grid steps, counted along each axis by node
    index (so it follows non-linear axes): 0.5 reaches the midpoint to the neighbour
    (cells tile without overlap), 1 the neighbouring node, 1.5 the midpoint between the
    1st and 2nd neighbours, 2 the 2nd neighbour. Beyond an axis end the end step is
    continued. All points inside the window count equally.
    """
    if overlap <= 0:
        raise LogError(f"overlap must be > 0, got {overlap}")
    x_lo, x_hi = _bounds(x_axis, x_value, overlap)
    y_lo, y_hi = _bounds(y_axis, y_value, overlap)
    return x_lo, x_hi, y_lo, y_hi


def cell_points(log: Log, x_axis: Axis, y_axis: Axis, x_value: int, y_value: int,
                overlap: float = 0.5, x: str = "RPM", y: str = "MAP") -> Log:
    """Rows of ``log`` whose ``(x, y)`` fall inside the cell (bounds inclusive)."""
    x_lo, x_hi, y_lo, y_hi = cell_bounds(x_axis, y_axis, x_value, y_value, overlap)
    xs, ys = log.column(x), log.column(y)
    return log.where((xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi))


def plot_cell(log: Log, x_axis: Axis, y_axis: Axis, x_value: int, y_value: int,
              what: str = "CORR", vs: str = "RPM", path: str | os.PathLike | None = None,
              overlap: float = 0.5, x: str = "RPM", y: str = "MAP", **scatter_kw):
    """Scatter ``what`` against ``vs`` for the points of one cell; save to ``path`` if given."""
    import matplotlib
    if path is not None and not os.environ.get("DISPLAY") and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = cell_points(log, x_axis, y_axis, x_value, y_value, overlap, x, y)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_kw.setdefault("s", 6)
    ax.scatter(pts.column(vs), pts.column(what), **scatter_kw)
    ax.set_xlabel(f"{vs} [{log.units[vs]}]" if log.units.get(vs) else vs)
    ax.set_ylabel(f"{what} [{log.units[what]}]" if log.units.get(what) else what)
    ax.set_title(f"{x}={x_value}, {y}={y_value} (overlap {overlap}): {len(pts)} points")
    ax.grid(True, alpha=0.3)
    if path is not None:
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
    return fig
