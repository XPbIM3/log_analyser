"""Table axes (breakpoints) with ECU-specific quantization checks."""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

Curve = Callable[[np.ndarray], np.ndarray]


class AxisError(ValueError):
    """Raised when axis values or construction parameters are invalid."""


def _check_quantum(owner: str, q) -> int:
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q <= 0:
        raise AxisError(f"{owner}: quantum must be a positive int, got {q!r}")
    return int(q)


class Axis:
    """Strictly increasing 1-D set of integer breakpoints, each a multiple of ``quantum``.

    Subclasses fix the semantics (``name``/``unit``), the default ``quantum`` and
    ``min_value``; ``quantum`` can be overridden per instance for ECUs with other
    restrictions.
    """

    name: str = "axis"
    unit: str = ""
    quantum: int = 1
    min_value: int = 0
    MIN_POINTS = 2

    def __init__(self, values, quantum: int | None = None) -> None:
        cls_name = type(self).__name__
        self.quantum = _check_quantum(cls_name, self.quantum if quantum is None else quantum)

        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as e:
            raise AxisError(f"{cls_name}: values are not numeric: {e}") from None

        if arr.ndim != 1:
            raise AxisError(f"{cls_name}: expected 1-D values, got shape {arr.shape}")
        if arr.size < self.MIN_POINTS:
            raise AxisError(f"{cls_name}: need at least {self.MIN_POINTS} points, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise AxisError(f"{cls_name}: values must be finite, got {arr.tolist()}")

        non_int = arr[arr != np.round(arr)]
        if non_int.size:
            raise AxisError(f"{cls_name}: values must be integers, bad: {non_int.tolist()}")
        ints = arr.astype(np.int64)

        below = ints[ints < self.min_value]
        if below.size:
            raise AxisError(f"{cls_name}: values must be >= {self.min_value}, bad: {below.tolist()}")

        bad = ints[ints % self.quantum != 0]
        if bad.size:
            raise AxisError(f"{cls_name}: values must be multiples of {self.quantum}, bad: {bad.tolist()}")

        diffs = np.diff(ints)
        if np.any(diffs == 0):
            dups = sorted({int(v) for v in ints[1:][diffs == 0]})
            raise AxisError(f"{cls_name}: duplicate values: {dups}")
        if np.any(diffs < 0):
            i = int(np.argmax(diffs < 0))
            raise AxisError(
                f"{cls_name}: values must be increasing, at index {i + 1}: {ints[i]} > {ints[i + 1]}"
            )

        ints.flags.writeable = False
        self._values = ints

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_curve(cls, start, stop, n: int, curve: Curve, quantum: int | None = None) -> "Axis":
        """Axis from ``start`` to ``stop`` with ``n`` points distributed by ``curve``.

        ``curve`` maps ``t = linspace(0, 1, n)`` to ``u`` in [0, 1] (``u[0] == 0``,
        ``u[-1] == 1``, non-decreasing); points are ``start + (stop - start) * u``
        snapped to ``quantum``. If snapping makes neighbours collide, points are
        pushed apart so every step is at least ``quantum`` and the ends stay exact.
        """
        name = cls.__name__
        q = _check_quantum(name, cls.quantum if quantum is None else quantum)

        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < cls.MIN_POINTS:
            raise AxisError(f"{name}: n must be an int >= {cls.MIN_POINTS}, got {n!r}")
        for label, v in (("start", start), ("stop", stop)):
            if not np.isfinite(v) or v != round(v) or int(round(v)) % q != 0:
                raise AxisError(f"{name}: {label}={v} must be an integer multiple of {q}")
        start, stop = int(round(start)), int(round(stop))
        if start >= stop:
            raise AxisError(f"{name}: start ({start}) must be < stop ({stop})")
        if (n - 1) * q > stop - start:
            raise AxisError(
                f"{name}: {n} points with step >= {q} do not fit in [{start}, {stop}] "
                f"(need span >= {(n - 1) * q})"
            )

        t = np.linspace(0.0, 1.0, n)
        try:
            u = np.asarray(curve(t), dtype=float)
        except (TypeError, ValueError) as e:
            raise AxisError(f"{name}: curve failed: {e}") from None
        if u.shape != t.shape or not np.all(np.isfinite(u)):
            raise AxisError(f"{name}: curve must return {n} finite values, got shape {u.shape}")
        if not (np.isclose(u[0], 0.0) and np.isclose(u[-1], 1.0)):
            raise AxisError(f"{name}: curve must map 0->0 and 1->1, got {u[0]:g} and {u[-1]:g}")
        if np.any(np.diff(u) < -1e-12):
            raise AxisError(f"{name}: curve must be non-decreasing")

        raw = start + (stop - start) * u
        return cls(cls._snap(raw, start, stop, q), quantum=quantum)

    @classmethod
    def linspace(cls, start, stop, n: int, quantum: int | None = None) -> "Axis":
        """Evenly spaced axis, points snapped to the nearest multiple of ``quantum``."""
        return cls.from_curve(start, stop, n, lambda t: t, quantum=quantum)

    @classmethod
    def gamma(cls, start, stop, n: int, gamma: float, quantum: int | None = None) -> "Axis":
        """Axis along ``u = t ** gamma``.

        ``gamma > 1``: denser at the start (low RPM / low load), sparser at the end;
        ``gamma == 1``: linear; ``0 < gamma < 1``: denser at the end.
        """
        if not np.isfinite(gamma) or gamma <= 0:
            raise AxisError(f"{cls.__name__}: gamma must be > 0, got {gamma!r}")
        return cls.from_curve(start, stop, n, lambda t: t ** gamma, quantum=quantum)

    @staticmethod
    def _snap(raw: np.ndarray, start: int, stop: int, q: int) -> np.ndarray:
        """Round to multiples of ``q`` keeping exact ends and a minimum step of ``q``.

        Caller guarantees ``(n - 1) * q <= stop - start``.
        """
        v = (np.round(raw / q).astype(np.int64) * q).clip(start, stop)
        v[0], v[-1] = start, stop
        for i in range(1, v.size):          # push up so each step >= q
            v[i] = max(v[i], v[i - 1] + q)
        v[-1] = stop
        for i in range(v.size - 2, -1, -1):  # pull down so we never pass stop
            v[i] = min(v[i], v[i + 1] - q)
        return v

    # ------------------------------------------------------------------ accessors

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def steps(self) -> np.ndarray:
        """Distances between neighbouring breakpoints (``len(self) - 1`` values)."""
        return np.diff(self._values)

    def __len__(self) -> int:
        return self._values.size

    def __iter__(self) -> Iterator[int]:
        return (int(v) for v in self._values)

    def __getitem__(self, idx):
        return self._values[idx]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Axis):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.quantum == other.quantum
            and np.array_equal(self._values, other._values)
        )

    def __hash__(self) -> int:
        return hash((type(self), self.quantum, self._values.tobytes()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._values.tolist()}, quantum={self.quantum})"


class RpmAxis(Axis):
    name = "rpm"
    unit = "RPM"
    quantum = 50


class KpaAxis(Axis):
    name = "kpa"
    unit = "kPa"
    quantum = 2
