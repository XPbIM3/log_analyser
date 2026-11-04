"""Tests for the mixture-delay scanner (shift auto-detection)."""
from pathlib import Path

import numpy as np
import pytest

from log_analyser.datalog import Log
from log_analyser.shift_scan import ShiftEstimate, recommend, scan_file, scan_shift

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def write_synth(path, dt=0.05, n=400, delay_rows=6, target_amp=0.45, noise=0.02, seed=1):
    """A 20 Hz pull-style log: the mixture is the target delayed by delay_rows (+ bias + noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    rpm = 1500 + 2500 * np.arange(n) / max(1, n - 1)
    rds = np.full(n, 2500.0 / (n * dt))                     # positive rpm/s: always accelerating
    tgt = 12.55 + target_amp * np.sin(2 * np.pi * t / 4.0)  # period 4 s
    afr = np.concatenate([np.full(delay_rows, tgt[0]), tgt[:n - delay_rows]]) + 0.3 \
        + rng.normal(0, noise, n)
    lines = ['"speeduino 202207: x"', '"Capture Date: Thu Sep 19 01:24:25 YEKT 2024"',
             "\t".join(("Time", "RPM", "MAP", "AFR", "Lambda", "AFR Target", "VE1", "Fan", "rpm/s")),
             "\t".join(("s", "rpm", "kpa", "O2", "", "O2", "%", "", "rpm/s"))]
    lines += ["\t".join(str(v) for v in row) for row in
              zip(t, rpm, np.full(n, 95.0), np.round(afr, 3), np.round(afr / 14.7, 3),
                  np.round(tgt, 3), np.full(n, 60.0), ["Inactive"] * n, rds)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def test_scan_finds_known_delay(tmp_path):
    # 6 rows at 20 Hz = 300 ms; the bias (+0.3) must not matter
    est = scan_file(write_synth(tmp_path / "s.msl", delay_rows=6))
    assert est.confidence == "confident"
    assert est.best_level == 6 and est.best_diff == 6 and est.best_err == 6
    assert est.delay_ms == pytest.approx(300.0, abs=1.0)


def test_detrend_short_segments_keep_signal():
    """Regression: the quadratic detrend used to return exact zeros on 2-3 point
    vectors, zeroing short pieces' error contributions and sliding the |error|
    minimum to the scan edge."""
    from log_analyser.shift_scan import _detrend
    base = np.array([0.0, 1.0, 0.5, -0.5, 1.0, 0.0, 0.5])
    for n in (2, 3, 5, 7):
        assert np.abs(_detrend(base[:n])).mean() > 1e-6


def write_pulses(path, dt=0.05, pulse=30, gap=20, n_pulses=15, delay_rows=6, seed=2):
    """A pull-style log of short WOT bursts: the mixture is the target delayed by
    delay_rows. Regression target: pieces so short that, unguarded, they yield
    2-3 point correlations (|r| = 1 garbage) at the far end of the scan."""
    rng = np.random.default_rng(seed)
    n = n_pulses * (pulse + gap)
    t = np.arange(n) * dt
    tgt = np.empty(n)
    wot = np.zeros(n, dtype=bool)
    for i in range(n_pulses):
        s = i * (pulse + gap)
        wot[s:s + gap] = False
        wot[s + gap:s + gap + pulse] = True
        tgt[s:s + gap] = 14.7
        phase = t[s:s + pulse]
        tgt[s + gap:s + gap + pulse] = 12.55 + 0.45 * np.sin(2 * np.pi * phase / 4.0)
    afr = np.concatenate([np.full(delay_rows, tgt[0]), tgt[:n - delay_rows]]) + 0.3 \
        + rng.normal(0, 0.02, n)
    rpm = np.where(wot, 4000.0, 2000.0)
    lines = ['"speeduino 202207: x"', '"Capture Date: Thu Sep 19 01:24:25 YEKT 2024"',
             "\t".join(("Time", "RPM", "MAP", "AFR", "Lambda", "AFR Target", "VE1", "Fan", "rpm/s")),
             "\t".join(("s", "rpm", "kpa", "O2", "", "O2", "%", "", "rpm/s"))]
    lines += ["\t".join(str(v) for v in row) for row in
              zip(t, rpm, np.where(wot, 95.0, 60.0), np.round(afr, 3), np.round(afr / 14.7, 3),
                  np.round(tgt, 3), np.full(n, 60.0), ["Inactive"] * n, np.full(n, 1000.0))]
    Path(path).write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def test_scan_on_short_bursts(tmp_path):
    est = scan_file(write_pulses(tmp_path / "p.msl"))
    assert est.section == "WOT"
    assert est.best_level == 6
    assert est.confidence == "confident"
    assert est.delay_ms == pytest.approx(300.0, abs=1.0)


def test_corroboration_requires_meaningful_diff():
    from log_analyser.shift_scan import _corroborated
    noise = np.array([0.02, 0.06, 0.01, 0.03])            # peaks at 0.06 < MIN_DIFF_CORR
    strong = np.array([0.02, 0.50, 0.01, 0.03])
    # bd=1 sits next to bl=2, but the dif curve itself is noise — no corroboration
    assert not _corroborated(2, 1, 9, noise, kmax=9)
    # a meaningful dif peak at bl corroborates
    assert _corroborated(1, 1, 9, strong, kmax=9)
    # err corroborates even with noisy dif, as long as its optimum is inside the scan
    assert _corroborated(4, 8, 4, noise, kmax=12)
    # an err optimum parked at the scan edge is drift, not agreement
    assert not _corroborated(4, 8, 12, noise, kmax=12)


def test_recommend_skips_edge_and_zero_estimates():
    weak_edge = ShiftEstimate(15.0, best_level=12, delay_ms=800.0, confidence="weak", at_edge=True)
    weak_zero = ShiftEstimate(15.0, best_level=0, delay_ms=0.0, confidence="weak")
    weak_ok = ShiftEstimate(15.0, best_level=4, delay_ms=268.0, confidence="weak")
    rec, note = recommend([weak_edge, weak_zero, weak_ok])
    assert rec == 268.0 and "слабые" in note


def test_scan_rejects_constant_target(tmp_path):
    est = scan_file(write_synth(tmp_path / "c.msl", target_amp=0.0))
    assert est.confidence == "none"
    assert "постоянна" in est.reason


def test_scan_rejects_short_log(tmp_path):
    est = scan_file(write_synth(tmp_path / "sh.msl", n=50))
    assert est.confidence == "none"
    assert "мало точек" in est.reason


def test_scan_rejects_missing_columns(tmp_path):
    p = tmp_path / "m.msl"
    p.write_text("Time\tRPM\ns\trpm\n0\t1500\n0.05\t1600\n")
    est = scan_shift(Log.load(p))
    assert est.confidence == "none" and "цели" in est.reason


@pytest.mark.skipif(not list(FIXTURES.glob("*.msl")), reason="msl fixtures not present")
def test_scan_on_real_fixture():
    est = scan_file(str(next(FIXTURES.glob("*.msl"))))
    assert est.rate_hz == pytest.approx(15.6, rel=0.02)
    if est.confidence != "none":
        assert est.delay_ms == pytest.approx(est.best_level * 1000.0 / est.rate_hz)
        assert 0 <= est.best_level <= max(4, int(round(0.8 * est.rate_hz)))


def test_recommend_median_and_fallback():
    conf = [ShiftEstimate(20.0, delay_ms=300.0, confidence="confident"),
            ShiftEstimate(15.0, delay_ms=260.0, confidence="confident")]
    rec, note = recommend(conf)
    assert rec == 280.0 and "уверенная" in note
    rec, note = recommend([ShiftEstimate(15.0, delay_ms=400.0, confidence="weak")])
    assert rec == 400.0 and "слабые" in note
    rec, note = recommend([ShiftEstimate(15.0, confidence="none")])
    assert rec is None and "не удалось" in note
