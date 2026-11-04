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
