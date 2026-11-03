"""Generating an AFR target table from a few macro parameters.

AFR is constant along RPM and changes along load (kPa): a lean plateau up to
``kpa_lean``, a rich plateau from ``kpa_rich``, and a smooth S-shaped transition
(smootherstep) between them.
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
    """Parameters of the AFR(kPa) curve.

    ``bias`` reshapes the transition: ``t -> t ** bias``; ``bias > 1`` delays the
    enrichment towards ``kpa_rich``, ``bias < 1`` brings it closer to ``kpa_lean``.
    ``step`` is the rounding quantum of the resulting values (0 disables rounding).
    """

    lean: float = 15.0
    rich: float = 12.5
    kpa_lean: float = 55
    kpa_rich: float = 95
    bias: float = 1.0
    step: float = 0.1

    def __post_init__(self) -> None:
        for name in ("lean", "rich", "kpa_lean", "kpa_rich", "bias", "step"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise AfrFillError(f"{name} must be a finite number, got {v!r}")
        if self.rich <= 0:
            raise AfrFillError(f"rich must be > 0, got {self.rich}")
        if self.rich > self.lean:
            raise AfrFillError(f"rich ({self.rich}) must be <= lean ({self.lean})")
        if self.kpa_lean >= self.kpa_rich:
            raise AfrFillError(f"kpa_lean ({self.kpa_lean}) must be < kpa_rich ({self.kpa_rich})")
        if self.bias <= 0:
            raise AfrFillError(f"bias must be > 0, got {self.bias}")
        if self.step < 0:
            raise AfrFillError(f"step must be >= 0, got {self.step}")


def smootherstep(t: np.ndarray) -> np.ndarray:
    """``6t^5 - 15t^4 + 10t^3`` on [0, 1]: 0 -> 0, 1 -> 1, zero 1st and 2nd derivatives at the ends."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def afr_curve(kpa, fill: AfrFill, rounded: bool = True) -> np.ndarray:
    """Target AFR at load ``kpa`` (scalar or array); rounded to ``fill.step`` if ``rounded``."""
    k = np.asarray(kpa, dtype=np.float64)
    t = np.clip((k - fill.kpa_lean) / (fill.kpa_rich - fill.kpa_lean), 0.0, 1.0) ** fill.bias
    afr = fill.lean - (fill.lean - fill.rich) * smootherstep(t)
    if rounded and fill.step > 0:
        afr = np.round(afr / fill.step) * fill.step
        decimals = max(0, -math.floor(math.log10(fill.step)) + 1)
        afr = np.round(afr, decimals)
    return afr


def afr_target_table(rpm: RpmAxis, kpa: KpaAxis, fill: AfrFill,
                     caption: str = "AFR Target Table") -> Table:
    """AFR target table over ``rpm`` (columns) and ``kpa`` (rows), constant along RPM."""
    col = afr_curve(kpa.values, fill)
    values = np.repeat(col[:, None], len(rpm), axis=1)
    return Table(TableKind.AFR, rpm, kpa, values, caption=caption)
