"""Steady-state log filtering and aggregation of the correction factor into a table grid."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .axis import Axis
from .cell import _bounds
from .datalog import Log, LogError
from .table import Table, TableKind

CORR_STATS = ("median", "mean", "q25", "q75")
STATS = CORR_STATS + ("sum",)     # sum: count occurrences (e.g. sync losses) per cell
_STAT_FUNCS = {
    "median": np.median,
    "mean": np.mean,
    "q25": lambda a: np.percentile(a, 25),   # lower estimate of the cell
    "q75": lambda a: np.percentile(a, 75),   # upper estimate of the cell
    "sum": np.sum,
}
# VE actually applied by the ECU in each row; names differ between TunerStudio/firmware versions
VE_COLUMNS = ("Current VE", "VE (Current", "VE1")


@dataclass(frozen=True)
class LogFilter:
    """Which log rows are trusted for mixture correction. ``None`` disables a check.

    Checks whose column is missing from the log are skipped (see :meth:`report`).
    """

    rpm_min: float | None = 1.0              # engine running
    clt_min: float | None = 70.0             # warm engine, °C
    exclude_dfco: bool = True                # DFCO != 0 -> no fuel
    exclude_warmup: bool = True              # Gwarm > 100 -> warm-up enrichment active
    exclude_accel: bool = True               # Accel Enrich != 100 -> accel enrichment active ...
    accel_tail: int = 0                      # ... and this many rows after it
    afr_min: float | None = 10.0             # wideband out of range / rails
    afr_max: float | None = 19.5
    rpm_dot_min: float | None = None         # rpm/s >= value (0: no falling RPM / engine braking)
    tps_above: float | None = None           # TPS > value, % (0: throttle not closed)

    def _checks(self, log: Log) -> list[tuple[str, str, np.ndarray | None]]:
        """``(description, column, mask_or_None_if_column_missing)`` for every enabled check."""
        cols = set(log.columns)

        def col(name: str):
            return log.column(name) if name in cols else None

        out: list[tuple[str, str, np.ndarray | None]] = []
        if self.rpm_min is not None:
            v = col("RPM")
            out.append((f"RPM ≥ {self.rpm_min:g}", "RPM", None if v is None else v >= self.rpm_min))
        if self.clt_min is not None:
            v = col("CLT")
            out.append((f"CLT ≥ {self.clt_min:g}", "CLT", None if v is None else v >= self.clt_min))
        if self.exclude_dfco:
            v = col("DFCO")
            out.append(("без DFCO", "DFCO", None if v is None else v == 0))
        if self.exclude_warmup:
            v = col("Gwarm")
            out.append(("без прогревного обогащения", "Gwarm", None if v is None else v <= 100))
        if self.exclude_accel:
            v = col("Accel Enrich")
            if v is None:
                out.append(("без ускорительного обогащения", "Accel Enrich", None))
            else:
                hit = (v != 100).astype(np.int64)
                if self.accel_tail > 0:
                    # row is affected if any of the previous accel_tail rows (or itself) had enrichment
                    k = np.ones(self.accel_tail + 1, dtype=np.int64)
                    hit = np.convolve(hit, k)[: hit.size]
                out.append((f"без ускор. обогащения (+{self.accel_tail} строк)", "Accel Enrich", hit == 0))
        if self.afr_min is not None or self.afr_max is not None:
            v = col("AFR")
            lo = -np.inf if self.afr_min is None else self.afr_min
            hi = np.inf if self.afr_max is None else self.afr_max
            out.append((f"AFR в [{lo:g}, {hi:g}]", "AFR", None if v is None else (v >= lo) & (v <= hi)))
        if self.rpm_dot_min is not None:
            v = col("rpm/s")
            out.append((f"rpm/s ≥ {self.rpm_dot_min:g}", "rpm/s",
                        None if v is None else v >= self.rpm_dot_min))
        if self.tps_above is not None:
            v = col("TPS")
            out.append((f"TPS > {self.tps_above:g}", "TPS", None if v is None else v > self.tps_above))
        return out

    def mask(self, log: Log) -> np.ndarray:
        m = np.ones(len(log), dtype=bool)
        for _desc, _col, mk in self._checks(log):
            if mk is not None:
                m &= mk
        return m

    def apply(self, log: Log) -> Log:
        return log.where(self.mask(log))

    def report(self, log: Log) -> list[str]:
        """Human-readable lines: how many rows each check removes, and skipped checks."""
        lines = []
        for desc, column, mk in self._checks(log):
            if mk is None:
                lines.append(f"{desc}: нет колонки {column!r}, пропущено")
            else:
                lines.append(f"{desc}: −{int(np.sum(~mk))}")
        return lines


@dataclass(frozen=True)
class CorrGrid:
    """Per-cell correction statistics over ``x`` (columns) and ``y`` (rows).

    ``values`` is NaN where a cell has fewer than ``min_points`` points; ``iqr`` is the
    interquartile range (75th − 25th percentile) of the cell's points, NaN likewise.
    """

    x: Axis
    y: Axis
    values: np.ndarray
    counts: np.ndarray
    stat: str
    min_points: int
    overlap: float
    meta: dict = field(default_factory=dict)
    iqr: np.ndarray | None = None

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.values)

    def table(self, fill: float = 1.0, caption: str = "Коррекция") -> Table:
        """As a CORR :class:`Table`; cells without data get ``fill`` (1.0 = no change)."""
        return Table(TableKind.CORR, self.x, self.y,
                     np.where(self.valid, self.values, fill), caption=caption)

    def apply_to(self, table: Table, caption: str | None = None) -> Table:
        """``table * correction`` for cells with data (e.g. VE_new = VE * CORR); axes must match."""
        if table.x != self.x or table.y != self.y:
            raise LogError("table axes differ from the correction grid axes")
        vals = np.where(self.valid, table.values * np.where(self.valid, self.values, 1.0), table.values)
        if table.kind.digits == 0:
            vals = np.round(vals)
        return Table(table.kind, table.x, table.y, vals,
                     caption=table.caption if caption is None else caption)


def corr_grid(log: Log, x_axis: Axis, y_axis: Axis, *, column: str = "CORR",
              stat: str = "median", min_points: int = 10, overlap: float = 0.5,
              x: str = "RPM", y: str = "MAP", clamp: bool = False) -> CorrGrid:
    """Aggregate ``column`` of ``log`` per cell of the ``x_axis`` × ``y_axis`` grid.

    Cell bounds are the same as in :func:`cell.cell_points`: ``overlap`` is the capture
    radius in grid steps along each axis (0.5 tiles the plane without overlap, 1 reaches
    the neighbouring node, 2 the second one); all points in the window count equally.
    ``clamp`` folds rows beyond the axis ends into the end cells (a sync loss below the
    lowest RPM node still shows up on the map); off by default so sparse outliers never
    distort a cell's statistic.
    """
    if stat not in STATS:
        raise LogError(f"stat must be one of {STATS}, got {stat!r}")
    if min_points < 1:
        raise LogError(f"min_points must be >= 1, got {min_points}")
    if overlap <= 0:
        raise LogError(f"overlap must be > 0, got {overlap}")
    vals = log.column(column).astype(np.float64)
    xs, ys = log.column(x), log.column(y)
    if clamp:
        xs = np.clip(xs, x_axis.values[0], x_axis.values[-1])
        ys = np.clip(ys, y_axis.values[0], y_axis.values[-1])

    x_masks = []
    for v in x_axis:
        lo, hi = _bounds(x_axis, v, overlap)
        x_masks.append((xs >= lo) & (xs <= hi))
    y_masks = []
    for v in y_axis:
        lo, hi = _bounds(y_axis, v, overlap)
        y_masks.append((ys >= lo) & (ys <= hi))

    out = np.full((len(y_axis), len(x_axis)), np.nan)
    iqr = np.full((len(y_axis), len(x_axis)), np.nan)
    counts = np.zeros((len(y_axis), len(x_axis)), dtype=np.int64)
    func = _STAT_FUNCS[stat]
    for i, ym in enumerate(y_masks):
        if not ym.any():
            continue
        for j, xm in enumerate(x_masks):
            sel = vals[ym & xm]
            sel = sel[np.isfinite(sel)]
            counts[i, j] = sel.size
            if sel.size >= min_points:
                out[i, j] = float(func(sel))
                q25, q75 = np.percentile(sel, [25, 75])
                iqr[i, j] = float(q75 - q25)
    return CorrGrid(x_axis, y_axis, out, counts, stat, min_points, overlap, iqr=iqr)


def find_ve_column(log: Log) -> str | None:
    """First of :data:`VE_COLUMNS` present in ``log``."""
    return next((c for c in VE_COLUMNS if c in log.columns), None)


@dataclass(frozen=True)
class VeResult:
    """New VE table on the grid axes.

    ``table``: cells with data = aggregated ``VE_row * CORR_row``; other cells = ``base``
    (the tune VE resampled to the grid axes) or 0 when there is no tune.
    ``grid``: the per-cell aggregation (``valid``/``counts`` tell which cells were computed).
    """

    table: Table
    base: Table | None
    grid: CorrGrid
    ve_column: str


def ve_grid(log: Log, x_axis: Axis, y_axis: Axis, *, base: Table | None = None,
            ve_column: str | None = None, corr: str = "CORR", stat: str = "median",
            min_points: int = 10, overlap: float = 0.5, x: str = "RPM", y: str = "MAP",
            caption: str = "VE (новая)") -> VeResult:
    """Predict VE per cell from a log that already has ``corr`` (see :meth:`Log.with_corr`).

    Each row's VE (the value the ECU used, ``ve_column``, auto-detected) times its
    correction is the VE that would have hit the target; it is aggregated per cell like
    :func:`corr_grid`. Works for any axes, independent of the tune's own VE axes.
    """
    col = ve_column or find_ve_column(log)
    if col is None:
        raise LogError(f"no VE column in the log (looked for {', '.join(VE_COLUMNS)})")
    if base is not None and base.kind is not TableKind.VE:
        raise LogError(f"base must be a VE table, got {base.kind.label}")
    data = log.data.copy()
    data["VE_NEW"] = log.column(col).astype(np.float64) * log.column(corr)
    g = corr_grid(Log(data), x_axis, y_axis, column="VE_NEW", stat=stat,
                  min_points=min_points, overlap=overlap, x=x, y=y)
    base_rs = None if base is None else base.resample(x_axis, y_axis)
    fill = np.zeros(g.values.shape) if base_rs is None else base_rs.values
    digits = TableKind.VE.digits
    values = np.round(np.where(g.valid, g.values, fill), digits)
    if base_rs is not None:
        base_rs = Table(TableKind.VE, x_axis, y_axis, np.round(base_rs.values, digits),
                        caption=base_rs.caption)
    return VeResult(Table(TableKind.VE, x_axis, y_axis, values, caption=caption), base_rs, g, col)
