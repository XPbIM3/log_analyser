"""Estimate the mixture transport delay (the shift) from a log itself.

The mixture measured at row ``i + k`` is the engine's response to the parameters
of row ``i``; the right ``k`` maximizes the agreement between the commanded
target and the shifted mixture. Three independent estimators are computed for
every candidate shift: correlation of levels (insensitive to a constant rich/lean
bias), correlation of first differences (per segment, free of slow trends) and
the mean absolute target error after removing the per-section trend (a raw |error|
curve goes flat when the bias dominates the target wiggle, and a slow drift slides
its minimum along the scan). An estimate is "confident" when the level peak is
strong and at least one of the two other estimators corroborates it with a
meaningful peak of its own (both optima inside the scan); otherwise the caller
gets "weak" or
"none" with the reason — never a silent guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .datalog import Log

MIN_POINTS = 100       # filtered rows needed for a meaningful correlation
MIN_SEGMENT = 8        # rows a shifted segment must keep to contribute (shorter ones
                       # yield degenerate correlations: |r| = 1 on 2-3 points)
TARGET_STD = 0.05      # AFR: the target must vary by at least this much (std)
MIN_CORR = 0.10        # below this the mixture simply does not follow the target
CONFIDENT_CORR = 0.40  # peak height required for a confident estimate (on real logs
                       # the reliability boundary sits near 0.45: peaks below ~0.4
                       # scatter across thresholds and leave-one-run-out checks)
MIN_DIFF_CORR = 0.15   # the difference estimator may corroborate only when its own
                       # maximum is meaningful: on real logs its curve peaks at
                       # 0.02-0.13 (noise) while a working one reaches 0.5+
SCAN_SECONDS = 0.8     # scan shifts up to this delay
TRIM_SECONDS = 0.25    # throttle-transient rows trimmed at the edges of every section


@dataclass
class ShiftEstimate:
    """Result of scanning one log for the mixture delay."""

    rate_hz: float | None
    best_level: int = 0            # best k by level correlation
    best_diff: int = 0             # best k by correlation of differences
    best_err: int = 0              # best k by mean |mixture - target|
    delay_ms: float | None = None  # best_level converted to milliseconds
    confidence: str = "none"       # "confident" | "weak" | "none"
    reason: str = ""               # why not confident / none (human-readable)
    n: int = 0                     # filtered rows used
    section: str = ""              # which rows were scanned: "WOT" or "движение"
    at_edge: bool = False          # True when the best k sits at the scan edge (kmax):
                                   # the delay may be larger, or the curve may be a drift
    table: list[tuple[int, float, float, float]] = field(default_factory=list)
    # table: per k -> (level corr, difference corr, mean |mixture - target|)


def scan_file(path: str) -> ShiftEstimate:
    """Load the file raw (no shift applied) and scan it."""
    return scan_shift(Log.load(path, shift_ms=0))


def _detrend(v: np.ndarray) -> np.ndarray:
    """Remove a slow trend: within a section the mixture drifts (richer to the top end),
    and that drift flattens or slides the error minimum along the scan. The degree
    follows the length — a quadratic on a handful of points eats the signal whole
    (residual ~0 on 2-3 points), which zeroes short segments' error contributions.
    ``t`` is normalized to avoid the ill-conditioning of raw row indices."""
    n = len(v)
    deg = 2 if n >= 16 else (1 if n >= 8 else 0)
    if deg == 0:
        return v - v.mean()
    t = (np.arange(n, dtype=float) - n / 2) / n
    return v - np.polyval(np.polyfit(t, v, deg), t)


def _corroborated(bl: int, bd: int, be: int, dif: np.ndarray, kmax: int) -> bool:
    """True when a second estimator independently agrees with the level peak.

    A corroborating estimator must (a) have a meaningful response of its own —
    a difference curve peaking at noise level says nothing even when its argmax
    is close to bl — and (b) have its optimum inside the scan: an error minimum
    parked at the edge is unresolved drift, not agreement.
    """
    dif_meaningful = float(np.nanmax(dif)) >= MIN_DIFF_CORR
    return ((dif_meaningful and abs(bl - bd) <= 1 and bd < kmax)
            or (abs(bl - be) <= 1 and be < kmax))


def scan_shift(log: Log) -> ShiftEstimate:
    target_name = next((c for c in ("AFR Target", "Lambda Target") if c in log.columns), None)
    if target_name is None:
        return ShiftEstimate(log.rate_hz, reason="нет колонки цели (AFR Target / Lambda Target)")
    meas_name = "AFR" if target_name == "AFR Target" else "Lambda"
    if meas_name not in log.columns:
        return ShiftEstimate(log.rate_hz, reason=f"нет колонки смеси ({meas_name})")
    if not log.rate_hz:
        return ShiftEstimate(None, reason="нет герцовки (нет колонки Time)")

    # WOT rows first if there are enough of them: under load the mixture is the response
    # to the fueling itself, while cruise adds the EGO controller lag and transients add
    # wall-wetting. Note: no rpm/s > 0 here on purpose — the rpm-falling rows of a shift
    # (throttle still open, MAP high) sweep the target through its cells fastest and carry
    # the sharpest timing signal. The "движение" fallback keeps rpm/s > 0: at cruise decels
    # the fuel cut pegs the mixture lean and it is pure noise for the correlation.
    wot = log.column("RPM") > 1000
    if "MAP" in log.columns:
        wot &= log.column("MAP") > 90
    if "TPS" in log.columns:
        wot &= log.column("TPS") > 70
    moving = log.column("RPM") > 1000
    if "rpm/s" in log.columns:
        moving &= log.column("rpm/s") > 0
    if wot.sum() >= MIN_POINTS:
        sel, section = wot, "WOT"
    else:
        sel, section = moving, "движение"

    idx = np.flatnonzero(sel)
    n = int(idx.size)
    if n < MIN_POINTS:
        return ShiftEstimate(log.rate_hz, n=n, section=section,
                             reason=f"мало точек после фильтра ({n})")

    tgt_all, meas_all = log.column(target_name), log.column(meas_name)
    # std, not IQR: a step target that sits in one cell most of the time has a vanishing
    # IQR while its cell transitions carry the whole timing signal
    std = float(tgt_all[idx].std())
    if std < TARGET_STD:
        return ShiftEstimate(log.rate_hz, n=n, section=section,
                             reason=f"цель почти постоянна (std {std:.2f}) — не с чем сопоставлять")

    # contiguous runs of filtered rows are the segments; the shift never crosses a gap.
    # The edges of every run are throttle transients (wall wetting, accel enrichment):
    # they do not follow the delay structure and pollute all three estimators.
    trim = max(1, int(round(TRIM_SECONDS * log.rate_hz)))
    runs = [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            if len(r) >= 2 * trim + 8]
    runs = [r[trim:-trim] for r in runs]

    kmax = max(4, int(round(SCAN_SECONDS * log.rate_hz)))
    table: list[tuple[int, float, float, float]] = []
    for k in range(kmax + 1):
        xs, ys, dxs, dys, errs = [], [], [], [], []
        for r in runs:
            if len(r) - k < MIN_SEGMENT:
                continue                                        # degenerate at this k
            x, y = tgt_all[r[:len(r) - k]], meas_all[r[k:]]     # target(i) vs mixture(i+k)
            xs.append(x)
            ys.append(y)
            dxs.append(np.diff(x))
            dys.append(np.diff(y))
            errs.append(np.abs(_detrend(y) - _detrend(x)))
        if not xs or sum(map(len, xs)) < 20:
            table.append((k, np.nan, np.nan, np.nan))
            continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        dx, dy = np.concatenate(dxs), np.concatenate(dys)
        # detrended |error|: a constant bias or a slow drift within a section must not
        # flatten the curve or slide its minimum along the scan
        table.append((k, float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(dx, dy)[0, 1]),
                      float(np.mean(np.concatenate(errs)))))

    lvl = np.array([t[1] for t in table])
    if np.all(np.isnan(lvl)):
        return ShiftEstimate(log.rate_hz, n=n, section=section,
                             reason="слишком короткие куски после фильтра")
    dif = np.array([t[2] for t in table])
    err = np.array([t[3] for t in table])
    bl, bd, be = int(np.nanargmax(lvl)), int(np.nanargmax(dif)), int(np.nanargmin(err))
    at_edge = bl == kmax
    est = ShiftEstimate(log.rate_hz, bl, bd, be, delay_ms=bl * 1000.0 / log.rate_hz,
                        n=n, section=section, at_edge=at_edge, table=table)

    r_peak = float(np.nanmax(lvl))
    if r_peak < MIN_CORR:
        est.reason = f"слабая связь цели и смеси (r_max {r_peak:.2f})"
        return est
    corroborated = _corroborated(bl, bd, be, dif, kmax)
    if r_peak >= CONFIDENT_CORR and not at_edge and corroborated:
        est.confidence = "confident"
        return est
    est.confidence = "weak"
    if at_edge:
        est.reason = "пик на краю диапазона скана — возможно, задержка больше"
    elif not corroborated:
        est.reason = f"оценщики расходятся (уровень {bl}, разности {bd}, ошибка {be})"
    else:
        est.reason = f"слабый пик (r_max {r_peak:.2f})"
    return est


def recommend(estimates: Iterable[ShiftEstimate]) -> tuple[float | None, str]:
    """Median delay in ms over the files; confident files first, weak as a fallback.

    Weak estimates whose peak sits at the scan edge are skipped: their curve is
    a monotone drift (common for smooth cruise targets), not a resolved delay —
    the per-file reason already says so.
    """
    estimates = list(estimates)
    for conf, note in (("confident", "уверенная сходимость по файлам"),
                       ("weak", "только слабые оценки — ориентир, не гарантия")):
        # a zero-row delay is physically impossible for mixture transport — such a
        # weak level-0 peak is an artifact, not an estimate, and must not vote
        delays = [e.delay_ms for e in estimates
                  if e.confidence == conf and not e.at_edge
                  and e.delay_ms is not None and e.delay_ms > 0]
        if delays:
            text = f"{note} (файлов: {len(delays)})"
            all_delays = [e.delay_ms for e in estimates
                          if not e.at_edge and e.delay_ms is not None and e.delay_ms > 0]
            if max(all_delays) - min(all_delays) > 100.0:
                text += f"; файлы расходятся на {max(all_delays) - min(all_delays):.0f} мс"
            return float(np.median(delays)), text
    return None, "ни по одному файлу не удалось оценить задержку"
