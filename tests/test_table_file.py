import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from log_analyser import (KpaAxis, RpmAxis, Table, TableFileError, TableKind,
                          export_table, read_table, table_to_string)

LEGACY_OUT = Path(__file__).resolve().parent.parent / "legacy" / "output"


@pytest.fixture
def ve():
    x = RpmAxis.gamma(400, 6000, 16, 1.5)
    y = KpaAxis.linspace(28, 160, 16)
    vals = np.add.outer(np.arange(16), np.arange(16)) + 50.4
    return Table(TableKind.VE, x, y, vals, caption="Primary VE Table")


def test_string_structure(ve):
    s = table_to_string(ve)
    root = ET.fromstring(s)
    ns = {"t": "http://www.EFIAnalytics.com/:table"}
    tab = root.find("t:table", ns)
    assert tab.get("rows") == "16" and tab.get("cols") == "16"
    assert tab.find("t:xAxis", ns).get("name") == "rpm"
    assert tab.find("t:yAxis", ns).get("name") == "fuelLoad"
    z_lines = tab.find("t:zValues", ns).text.strip().splitlines()
    assert len(z_lines) == 16
    assert z_lines[0].split()[:3] == ["50", "51", "52"]  # VE -> 0 digits, rounded


def test_digits(ve):
    afr = Table(TableKind.AFR, ve.x, ve.y, np.full((16, 16), 14.66))
    assert "14.7 14.7" in table_to_string(afr)
    assert "14.66 14.66" in table_to_string(afr, digits=2)
    assert "50.40 51.40" in table_to_string(ve, digits=2)
    with pytest.raises(ValueError):
        table_to_string(ve, digits=-1)


def test_round_trip(tmp_path, ve):
    p = tmp_path / "ve.table"
    export_table(ve, p, digits=1)
    back = read_table(p, TableKind.VE)
    assert back.x == ve.x and back.y == ve.y
    np.testing.assert_allclose(back.values, ve.values)


def test_non_square_round_trip(tmp_path):
    t = Table(TableKind.IGN, RpmAxis([500, 1000, 3000]), KpaAxis([30, 100]), [[10, 12, 20], [8, 9, 15]])
    p = tmp_path / "ign.table"
    export_table(t, p)
    back = read_table(p, TableKind.IGN)
    assert back.shape == (2, 3)
    np.testing.assert_array_equal(back.values, t.values)


def test_read_errors(tmp_path):
    p = tmp_path / "bad.table"
    p.write_text("<tableData/>")
    with pytest.raises(TableFileError, match="missing"):
        read_table(p, TableKind.VE)
    with pytest.raises(TableFileError, match="cannot read"):
        read_table(tmp_path / "nope.table", TableKind.VE)


@pytest.mark.skipif(not (LEGACY_OUT / "AFR_EXPORT.table").exists(), reason="legacy output not present")
@pytest.mark.parametrize("name, kind", [("AFR_EXPORT.table", TableKind.AFR), ("VE_EXPORT_w.table", TableKind.VE)])
def test_reads_legacy_exports(name, kind):
    t = read_table(LEGACY_OUT / name, kind)
    assert t.is_default_size
