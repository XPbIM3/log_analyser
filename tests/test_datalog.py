from pathlib import Path

import numpy as np
import pytest

from log_analyser import KpaAxis, Log, LogError, RpmAxis, Table, TableKind, Tune

ROOT = Path(__file__).resolve().parent.parent
REAL_LOGS = ROOT / "legacy" / "input"


def write_msl(path, rows, columns=("Time", "RPM", "MAP", "AFR", "Lambda", "AFR Target", "VE1", "Fan"),
              units=("s", "rpm", "kpa", "O2", "", "O2", "%", ""),
              date="Thu Sep 19 01:24:25 YEKT 2024", sig="speeduino 202207: Speeduino 202207.1"):
    lines = [f'"{sig}"', f'"Capture Date: {date}"', "\t".join(columns), "\t".join(units)]
    lines += ["\t".join(str(v) for v in r) for r in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def rows(n=5, afr0=14.0):
    return [(i * 0.067, 1000 + 100 * i, 50 + 2 * i, afr0 + i * 0.1, round((afr0 + i * 0.1) / 14.7, 3),
             14.0, 60 + i, "Inactive") for i in range(n)]


@pytest.fixture
def msl(tmp_path):
    return write_msl(tmp_path / "a.msl", rows())


def test_load_metadata(msl):
    log = Log.load(msl)
    assert log.signature == "speeduino 202207"
    assert log.firmware == "speeduino 202207: Speeduino 202207.1"
    assert log.capture_date == "Thu Sep 19 01:24:25 YEKT 2024"
    assert log.units["RPM"] == "rpm" and "Lambda" not in log.units
    assert log.data["source"].unique().tolist() == ["a.msl"]
    assert log.data["Fan"].tolist()[0] == "Inactive"
    assert log.data["RPM"].dtype.kind in "if"


def test_shift_next_row(msl):
    raw = Log.load(msl, shift=0)
    log = Log.load(msl)
    assert len(raw) == 5 and len(log) == 4
    np.testing.assert_allclose(log.column("AFR"), raw.column("AFR")[1:])
    np.testing.assert_allclose(log.column("Lambda"), raw.column("Lambda")[1:])
    # inputs stay on their own row
    np.testing.assert_array_equal(log.column("RPM"), raw.column("RPM")[:-1])
    np.testing.assert_array_equal(log.column("AFR Target"), raw.column("AFR Target")[:-1])


def test_shift_two_and_custom_columns(msl):
    raw = Log.load(msl, shift=0)
    log = Log.load(msl, shift=2, shift_columns=("AFR",))
    assert len(log) == 3
    np.testing.assert_allclose(log.column("AFR"), raw.column("AFR")[2:])
    np.testing.assert_allclose(log.column("Lambda"), raw.column("Lambda")[:3])
    with pytest.raises(LogError):
        Log.load(msl, shift=-1)


def test_shift_ignores_missing_columns(tmp_path):
    p = write_msl(tmp_path / "b.msl", [(0, 1000), (1, 1100)], columns=("Time", "RPM"), units=("s", "rpm"))
    assert len(Log.load(p)) == 2


def test_no_units_row(tmp_path):
    p = tmp_path / "c.msl"
    p.write_text("Time\tRPM\tAFR\n0\t1000\t14\n1\t1100\t15\n")
    log = Log.load(p)
    assert log.units == {} and len(log) == 1 and log.column("AFR")[0] == 15


@pytest.mark.parametrize("text, msg", [
    ('"sig"\nfoo\tbar\n1\t2\n', "no header"),
    ("Time\tRPM\ns\trpm\n", "no data"),
    ("Time\tRPM\ns\trpm\n0\t1\t2\t3\n", "cannot parse"),
])
def test_load_errors(tmp_path, text, msg):
    p = tmp_path / "bad.msl"
    p.write_text(text)
    with pytest.raises(LogError, match=msg):
        Log.load(p)


def test_missing_file(tmp_path):
    with pytest.raises(LogError, match="cannot read"):
        Log.load(tmp_path / "nope.msl")


def test_column_access(msl):
    log = Log.load(msl)
    with pytest.raises(LogError, match="no column 'Nope'"):
        log.column("Nope")


# ------------------------------------------------------------------ concat

def test_concat_same(tmp_path):
    a = Log.load(write_msl(tmp_path / "a.msl", rows(5)))
    b = Log.load(write_msl(tmp_path / "b.msl", rows(3, afr0=15)))
    c = Log.concat([a, b])
    assert len(c) == 4 + 2
    assert c.data.index.tolist() == list(range(6))
    assert c.data["source"].tolist() == ["a.msl"] * 4 + ["b.msl"] * 2
    # no leakage: last row of a keeps a's own shifted value
    assert c.column("AFR")[3] == pytest.approx(14.4)
    assert c.sources == a.sources + b.sources


def test_concat_different_columns(tmp_path):
    a = Log.load(write_msl(tmp_path / "a.msl", rows()))
    b = Log.load(write_msl(tmp_path / "b.msl", [r[:7] for r in rows()],
                           columns=("Time", "RPM", "MAP", "AFR", "Lambda", "AFR Target", "VE1"),
                           units=("s", "rpm", "kpa", "O2", "", "O2", "%")))
    with pytest.raises(LogError, match=r"b.msl lacks \['Fan'\]"):
        Log.concat([a, b])
    with pytest.warns(UserWarning, match="Fan"):
        c = Log.concat([a, b], columns="common")
    assert "Fan" not in c.columns and "VE1" in c.columns


def test_concat_errors(tmp_path):
    a = Log.load(write_msl(tmp_path / "a.msl", rows()))
    b = Log.load(write_msl(tmp_path / "b.msl", rows(), sig="rusefi: x"))
    with pytest.raises(LogError, match="different firmware"):
        Log.concat([a, b])
    with pytest.raises(LogError, match="nothing"):
        Log.concat([])
    with pytest.raises(LogError, match="columns must be"):
        Log.concat([a], columns="all")


def test_load_many_orders_by_date(tmp_path):
    write_msl(tmp_path / "1.msl", rows(3, afr0=15), date="Thu Sep 19 02:00:00 YEKT 2024")
    write_msl(tmp_path / "2.msl", rows(3, afr0=14), date="Thu Sep 19 01:00:00 YEKT 2024")
    log = Log.load_many(str(tmp_path / "*.msl"))
    assert log.data["source"].tolist()[0] == "2.msl"
    with pytest.raises(LogError, match="no log files"):
        Log.load_many(str(tmp_path / "*.nope"))


# ------------------------------------------------------------------ where

def test_where(msl):
    log = Log.load(msl)
    a = log.where(log.column("RPM") > 1100)
    b = log.where(lambda d: d.RPM > 1100)
    assert a.column("RPM").tolist() == b.column("RPM").tolist() == [1200, 1300]
    assert a.data.index.tolist() == [0, 1]
    assert log.where(lambda d: d.RPM > 1000).where(lambda d: d.MAP < 56).column("RPM").tolist() == [1100, 1200]
    assert a.signature == log.signature and a.units == log.units
    assert len(log) == 4  # original untouched


def test_where_errors(msl):
    log = Log.load(msl)
    with pytest.raises(LogError, match="shape"):
        log.where(np.array([True, False]))
    with pytest.raises(LogError, match="boolean"):
        log.where(log.column("RPM"))


# ------------------------------------------------------------------ CORR

def test_corr_from_log_column(msl):
    log = Log.load(msl).with_corr("AFR Target")
    np.testing.assert_allclose(log.column("CORR"), log.column("AFR") / 14.0)
    np.testing.assert_allclose(log.column("CORR_target"), 14.0)


def test_corr_from_table(msl):
    x = RpmAxis([1000, 2000])
    y = KpaAxis([40, 100])
    afr = Table(TableKind.AFR, x, y, [[15.0, 15.0], [12.0, 12.0]])
    log = Log.load(msl).with_corr(afr)
    expected_target = 15.0 - 3.0 * (log.column("MAP") - 40) / 60
    np.testing.assert_allclose(log.column("CORR_target"), expected_target)
    np.testing.assert_allclose(log.column("CORR"), log.column("AFR") / expected_target)


def test_corr_lambda_table(msl):
    lam = Table(TableKind.LAMBDA, RpmAxis([1000, 2000]), KpaAxis([40, 100]), np.ones((2, 2)))
    log = Log.load(msl).with_corr(lam, name="K")
    np.testing.assert_allclose(log.column("K"), log.column("Lambda"))


def test_corr_errors(msl):
    log = Log.load(msl)
    ve = Table(TableKind.VE, RpmAxis([1000, 2000]), KpaAxis([40, 100]), np.ones((2, 2)))
    with pytest.raises(LogError, match="AFR or LAMBDA"):
        log.with_corr(ve)
    with pytest.raises(LogError, match="infer"):
        log.with_corr("VE1")
    assert log.with_corr("VE1", measured="AFR").column("CORR").size == 4
    zero = Table(TableKind.AFR, RpmAxis([1000, 2000]), KpaAxis([40, 100]), np.zeros((2, 2)))
    with pytest.raises(LogError, match="zero"):
        log.with_corr(zero)


# ------------------------------------------------------------------ real data

@pytest.mark.skipif(not list(REAL_LOGS.glob("*.msl")), reason="legacy logs not present")
def test_real_logs():
    with pytest.raises(LogError, match="column sets differ"):
        Log.load_many(str(REAL_LOGS / "*.msl"))
    with pytest.warns(UserWarning, match="Error #"):
        log = Log.load_many(str(REAL_LOGS / "*.msl"), columns="common")
    assert len(log.sources) == 3
    assert len(log) == 54005 - 3
    assert not np.isnan(log.column("AFR")).any()
    tune = Tune.load(REAL_LOGS / "CurrentTune.msq")
    assert log.signature == tune.signature == "speeduino 202207"
