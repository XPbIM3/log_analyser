from pathlib import Path

import numpy as np
import pytest

from log_analyser import KpaAxis, RpmAxis, TableKind, TableSpec, Tune, TuneError

REAL_TUNE = Path(__file__).resolve().parent.parent / "legacy" / "input" / "CurrentTune.msq"


def _const(name, rows, cols, values, units="", digits=0):
    body = "\n".join(" ".join(f"{v:.1f}" for v in row) for row in np.asarray(values, float).reshape(rows, cols))
    return (f'<constant cols="{cols}" digits="{digits}" name="{name}" rows="{rows}" units="{units}">\n'
            f'{body}\n</constant>')


def _msq(*consts):
    return ('<?xml version="1.0" encoding="ISO-8859-1"?>\n'
            '<msq xmlns="http://www.msefi.com/:msq">\n'
            '<page number="1">\n' + "\n".join(consts) + '\n</page>\n</msq>').encode("iso-8859-1")


RPM = [500, 1000, 2000, 4000]
KPA = [30, 60, 100]
VE = np.arange(12).reshape(3, 4) + 50


@pytest.fixture
def tune():
    return Tune.from_string(_msq(
        _const("veTable", 3, 4, VE, "%"),
        _const("rpmBins", 4, 1, RPM, "RPM"),
        _const("fuelLoadBins", 3, 1, KPA, "kPa"),
        _const("afrTable", 3, 4, np.full((3, 4), 14.7), "AFR", digits=1),
        _const("rpmBinsAFR", 4, 1, RPM, "RPM"),
        _const("loadBinsAFR", 3, 1, KPA, "kPa"),
        _const("badBins", 3, 1, [30, 31, 40], "kPa"),
        '<constant name="nCylinders">"4 cylinder"</constant>',
    ))


def test_constants(tune):
    assert "veTable" in tune and "nope" not in tune
    c = tune.constant("veTable")
    assert (c.rows, c.cols, c.units, c.digits) == (3, 4, "%", 0)
    np.testing.assert_array_equal(c.array(), VE)


def test_missing_constant(tune):
    with pytest.raises(TuneError, match="no constant named 'nope'"):
        tune.constant("nope")


def test_non_numeric_constant(tune):
    with pytest.raises(TuneError, match="not numeric"):
        tune.constant("nCylinders").array()


def test_axis(tune):
    a = tune.axis("rpmBins", RpmAxis)
    assert isinstance(a, RpmAxis) and list(a) == RPM
    with pytest.raises(TuneError, match="not an axis"):
        tune.axis("veTable")
    with pytest.raises(TuneError, match="not a valid KpaAxis"):
        tune.axis("badBins", KpaAxis)


def test_table_by_name(tune):
    t = tune.table("veTable")
    assert t.kind is TableKind.VE
    assert t.caption == "Primary VE Table"
    assert list(t.x) == RPM and list(t.y) == KPA
    np.testing.assert_array_equal(t.values, VE)
    assert tune.table("veTable", caption="mine").caption == "mine"


def test_unknown_table(tune):
    with pytest.raises(TuneError, match="no axis mapping"):
        tune.table("someTable")


def test_table_from_and_bad_axis(tune):
    t = tune.table_from("veTable", "rpmBinsAFR", "loadBinsAFR", TableKind.CORR)
    assert t.kind is TableKind.CORR
    with pytest.raises(TuneError, match="veTable"):
        tune.table_from("veTable", "rpmBins", "rpmBins", TableKind.VE)  # 4 rows != 3


def test_tables_present_only(tune):
    assert set(tune.tables()) == {"veTable", "afrTable"}


def test_custom_specs():
    tune = Tune.from_string(_msq(
        _const("t", 3, 4, VE), _const("x", 4, 1, RPM), _const("y", 3, 1, KPA),
    ), table_specs={"t": TableSpec("x", "y", TableKind.IGN, "custom")})
    assert tune.table("t").kind is TableKind.IGN


def test_lambda_units_switch_kind():
    tune = Tune.from_string(_msq(
        _const("afrTable", 3, 4, np.ones((3, 4)), "Lambda"),
        _const("rpmBinsAFR", 4, 1, RPM), _const("loadBinsAFR", 3, 1, KPA),
    ))
    assert tune.table("afrTable").kind is TableKind.LAMBDA


@pytest.mark.parametrize("xml", [b"<notmsq/>", b"<msq>", b""])
def test_bad_xml(xml):
    with pytest.raises(TuneError):
        Tune.from_string(xml)


def test_size_mismatch():
    tune = Tune.from_string(_msq('<constant cols="4" name="t" rows="3">1 2 3</constant>'))
    with pytest.raises(TuneError, match="expected 3x4"):
        tune.constant("t").array()


@pytest.mark.skipif(not REAL_TUNE.exists(), reason="legacy tune not present")
def test_real_tune():
    tune = Tune.load(REAL_TUNE)
    # advTable2 is unused in this tune: its mapBins2 axis is all zeros
    with pytest.warns(UserWarning, match="advTable2"):
        tables = tune.tables()
    assert set(tables) == {"veTable", "veTable2", "advTable1", "afrTable"}
    with pytest.raises(TuneError, match="mapBins2"):
        tune.tables(strict=True)
    for t in tables.values():
        assert t.is_default_size
    ve = tables["veTable"]
    assert list(ve.x)[:3] == [400, 700, 1100]
    assert list(ve.y)[:3] == [28, 38, 48]
    assert ve.values[0, :4].tolist() == [56, 56, 60, 68]
    assert tables["afrTable"].values[0, 0] == pytest.approx(14.5)
