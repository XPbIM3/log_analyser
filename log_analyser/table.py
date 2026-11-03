"""2-D calibration tables over two axes."""
from __future__ import annotations

from enum import Enum

import numpy as np

from .axis import Axis

DEFAULT_SIZE = 16


class TableKind(Enum):
    """Semantic type of the data stored in a table: (short name, unit, default digits)."""

    AFR = ("AFR", "AFR", 1)
    LAMBDA = ("Lambda", "λ", 3)
    VE = ("VE", "%", 0)
    IGN = ("IGN", "deg", 0)
    CORR = ("CORR", "ratio", 3)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def unit(self) -> str:
        return self.value[1]

    @property
    def digits(self) -> int:
        """Default number of decimals when the table is written to text formats."""
        return self.value[2]


class TableError(ValueError):
    """Raised when a table is inconsistent."""


def _segment(nodes: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the left node of the segment containing ``q`` and the position in it (0..1), clamped."""
    q = np.clip(q, nodes[0], nodes[-1])
    i = np.clip(np.searchsorted(nodes, q, side="right") - 1, 0, nodes.size - 2)
    t = (q - nodes[i]) / (nodes[i + 1] - nodes[i])
    return i, t


class Table:
    """Values over ``x`` (columns, usually RPM) and ``y`` (rows, usually load/kPa).

    ``values.shape == (len(y), len(x))`` — row ``i`` corresponds to ``y[i]``,
    column ``j`` to ``x[j]`` (same layout as TunerStudio / the legacy code).
    """

    def __init__(self, kind: TableKind, x: Axis, y: Axis, values, caption: str = "") -> None:
        if not isinstance(kind, TableKind):
            raise TableError(f"kind must be TableKind, got {kind!r}")
        if not isinstance(x, Axis) or not isinstance(y, Axis):
            raise TableError(f"x and y must be Axis instances, got {type(x).__name__}, {type(y).__name__}")
        try:
            arr = np.array(values, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TableError(f"values are not numeric: {e}") from None
        if arr.ndim != 2:
            raise TableError(f"values must be 2-D, got shape {arr.shape}")
        expected = (len(y), len(x))
        if arr.shape != expected:
            raise TableError(f"values shape {arr.shape} != (len(y), len(x)) = {expected}")
        if not np.all(np.isfinite(arr)):
            raise TableError("values must be finite")

        arr.flags.writeable = False
        self._kind = kind
        self._x = x
        self._y = y
        self._values = arr
        self._caption = str(caption)

    @property
    def kind(self) -> TableKind:
        return self._kind

    @property
    def caption(self) -> str:
        return self._caption

    @property
    def x(self) -> Axis:
        return self._x

    @property
    def y(self) -> Axis:
        return self._y

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    @property
    def is_default_size(self) -> bool:
        return self.shape == (DEFAULT_SIZE, DEFAULT_SIZE)

    def lookup(self, x, y) -> np.ndarray:
        """Bilinear interpolation at points ``(x, y)`` (x along columns, y along rows).

        Points outside the grid are clamped to the edge values (no extrapolation).
        Accepts scalars or arrays of equal shape; returns an array of that shape.
        """
        xq = np.asarray(x, dtype=np.float64)
        yq = np.asarray(y, dtype=np.float64)
        if xq.shape != yq.shape:
            raise TableError(f"x and y must have the same shape, got {xq.shape} and {yq.shape}")
        xs = self._x.values.astype(np.float64)
        ys = self._y.values.astype(np.float64)
        xi, tx = _segment(xs, xq)
        yi, ty = _segment(ys, yq)
        v = self._values
        top = v[yi, xi] * (1 - tx) + v[yi, xi + 1] * tx
        bottom = v[yi + 1, xi] * (1 - tx) + v[yi + 1, xi + 1] * tx
        return top * (1 - ty) + bottom * ty

    def __repr__(self) -> str:
        rows, cols = self.shape
        return f"Table({self._kind.label}, {self._caption!r}, {rows}x{cols})"

    def resample(self, x: Axis, y: Axis, caption: str | None = None) -> "Table":
        """This table on new axes: every node of ``x`` × ``y`` gets :meth:`lookup` (bilinear, clamped)."""
        xx, yy = np.meshgrid(x.values.astype(np.float64), y.values.astype(np.float64))
        return Table(self._kind, x, y, self.lookup(xx, yy),
                     caption=self._caption if caption is None else caption)
