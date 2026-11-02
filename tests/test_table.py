import numpy as np
import pytest

from log_analyser import DEFAULT_SIZE, KpaAxis, RpmAxis, Table, TableError, TableKind


@pytest.fixture
def rpm():
    return RpmAxis.linspace(400, 6000, 16)


@pytest.fixture
def kpa():
    return KpaAxis.linspace(28, 160, 16)


def test_default_ve_table(rpm, kpa):
    t = Table(TableKind.VE, rpm, kpa, np.full((16, 16), 70), caption="Primary VE Table")
    assert t.shape == (16, 16)
    assert t.is_default_size
    assert t.kind is TableKind.VE
    assert t.caption == "Primary VE Table"
    assert t.x is rpm and t.y is kpa
    assert t.values.dtype == np.float64
    assert repr(t) == "Table(VE, 'Primary VE Table', 16x16)"


def test_shape_mismatch(rpm, kpa):
    with pytest.raises(TableError, match="shape"):
        Table(TableKind.VE, rpm, kpa, np.zeros((16, 15)))


def test_transposed_shape_rejected():
    x = RpmAxis.linspace(500, 6000, 12)
    y = KpaAxis.linspace(30, 100, 8)
    Table(TableKind.AFR, x, y, np.zeros((8, 12)))
    with pytest.raises(TableError):
        Table(TableKind.AFR, x, y, np.zeros((12, 8)))


def test_non_default_size():
    t = Table(TableKind.IGN, RpmAxis([500, 1000, 1500]), KpaAxis([30, 60]), [[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert not t.is_default_size
    assert DEFAULT_SIZE == 16


@pytest.mark.parametrize("values", [np.zeros(16), np.zeros((16, 16, 1)), np.full((16, 16), np.nan)])
def test_bad_values(rpm, kpa, values):
    with pytest.raises(TableError):
        Table(TableKind.VE, rpm, kpa, values)


def test_bad_kind_or_axes(rpm, kpa):
    with pytest.raises(TableError):
        Table("VE", rpm, kpa, np.zeros((16, 16)))
    with pytest.raises(TableError):
        Table(TableKind.VE, list(rpm), kpa, np.zeros((16, 16)))


def test_values_read_only_and_copied(rpm, kpa):
    src = np.zeros((16, 16))
    t = Table(TableKind.CORR, rpm, kpa, src)
    src[0, 0] = 5
    assert t.values[0, 0] == 0
    with pytest.raises(ValueError):
        t.values[0, 0] = 1


@pytest.mark.parametrize("kind", list(TableKind))
def test_all_kinds(rpm, kpa, kind):
    t = Table(kind, rpm, kpa, np.ones((16, 16)))
    assert t.kind is kind
    assert kind.label and kind.unit


def test_lookup():
    t = Table(TableKind.AFR, RpmAxis([1000, 2000, 4000]), KpaAxis([30, 50]),
              [[10.0, 20.0, 40.0], [30.0, 40.0, 60.0]])
    # nodes
    assert t.lookup(1000, 30) == 10 and t.lookup(4000, 50) == 60
    # bilinear
    assert t.lookup(1500, 30) == pytest.approx(15)
    assert t.lookup(3000, 40) == pytest.approx((30 + 50) / 2)
    # clamped outside
    assert t.lookup(500, 20) == 10 and t.lookup(9000, 200) == 60
    out = t.lookup(np.array([1000, 1500]), np.array([30, 50]))
    np.testing.assert_allclose(out, [10, 35])
    with pytest.raises(TableError, match="same shape"):
        t.lookup([1000, 2000], [30])
