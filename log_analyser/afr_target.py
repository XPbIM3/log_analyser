"""Generating an AFR target table from a few macro parameters.

Two S-shaped transitions (smootherstep) from ``lean`` to ``rich``: along load —
lean up to ``kpa_lean``, rich from ``kpa_rich`` — and, optionally, along RPM —
lean up to ``rpm_lean``, rich from ``rpm_rich``. They are combined with max:
a cell is as rich as the richer of the two ("high load OR high RPM").
Without the RPM transition the table is constant along RPM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .axis import KpaAxis, RpmAxis
from .table import Table, TableKind


class AfrFillError(ValueError):
    """Raised when AFR fill parameters are inconsistent."""


@dataclass(frozen=True)
class AfrFill:
    """Parameters of the AFR(RPM, kPa) map.

    ``rpm_lean``/``rpm_rich`` are both ``None`` (no RPM enrichment) or both set.
    ``step`` is the rounding quantum of the resulting values (0 disables rounding).
    """

    lean: float = 15.0
    rich: float = 12.5
    kpa_lean: float = 55
    kpa_rich: float = 95
    rpm_lean: float | None = None
    rpm_rich: float | None = None
    step: float = 0.1

    def __post_init__(self) -> None:
        names = ["lean", "rich", "kpa_lean", "kpa_rich", "step"]
        if (self.rpm_lean is None) != (self.rpm_rich is None):
            raise AfrFillError("rpm_lean and rpm_rich must be both set or both None")
        if self.rpm_lean is not None:
            names += ["rpm_lean", "rpm_rich"]
        for name in names:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise AfrFillError(f"{name} must be a finite number, got {v!r}")
        if self.rich <= 0:
            raise AfrFillError(f"rich must be > 0, got {self.rich}")
        if self.rich > self.lean:
            raise AfrFillError(f"rich ({self.rich}) must be <= lean ({self.lean})")
        if self.kpa_lean >= self.kpa_rich:
            raise AfrFillError(f"kpa_lean ({self.kpa_lean}) must be < kpa_rich ({self.kpa_rich})")
        if self.rpm_lean is not None and self.rpm_lean >= self.rpm_rich:
            raise AfrFillError(f"rpm_lean ({self.rpm_lean}) must be < rpm_rich ({self.rpm_rich})")
        if self.step < 0:
            raise AfrFillError(f"step must be >= 0, got {self.step}")


def smootherstep(t: np.ndarray) -> np.ndarray:
    """``6t^5 - 15t^4 + 10t^3`` on [0, 1]: 0 -> 0, 1 -> 1, zero 1st and 2nd derivatives at the ends."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _transition(v, lo: float, hi: float) -> np.ndarray:
    """0 up to ``lo``, 1 from ``hi``, smootherstep between."""
    return smootherstep((np.asarray(v, dtype=np.float64) - lo) / (hi - lo))


def _round(afr: np.ndarray, step: float) -> np.ndarray:
    if step <= 0:
        return afr
    afr = np.round(afr / step) * step
    return np.round(afr, max(0, -math.floor(math.log10(step)) + 1))


def afr_map(rpm, kpa, fill: AfrFill, rounded: bool = True) -> np.ndarray:
    """Target AFR at (``rpm``, ``kpa``) (broadcast together); rounded to ``fill.step`` if ``rounded``."""
    rpm, kpa = np.broadcast_arrays(np.asarray(rpm, dtype=np.float64), np.asarray(kpa, dtype=np.float64))
    s = _transition(kpa, fill.kpa_lean, fill.kpa_rich)
    if fill.rpm_lean is not None:
        s = np.maximum(s, _transition(rpm, fill.rpm_lean, fill.rpm_rich))
    afr = fill.lean - (fill.lean - fill.rich) * s
    return _round(afr, fill.step) if rounded else afr


def afr_curve(kpa, fill: AfrFill, rounded: bool = True) -> np.ndarray:
    """Target AFR along load only (the RPM transition ignored)."""
    k = np.asarray(kpa, dtype=np.float64)
    afr = fill.lean - (fill.lean - fill.rich) * _transition(k, fill.kpa_lean, fill.kpa_rich)
    return _round(afr, fill.step) if rounded else afr


def afr_target_table(rpm: RpmAxis, kpa: KpaAxis, fill: AfrFill,
                     caption: str = "AFR Target Table") -> Table:
    """AFR target table over ``rpm`` (columns) and ``kpa`` (rows)."""
    values = afr_map(rpm.values[None, :], kpa.values[:, None], fill)
    return Table(TableKind.AFR, rpm, kpa, values, caption=caption)
