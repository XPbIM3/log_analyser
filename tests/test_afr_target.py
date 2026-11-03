import numpy as np
import pytest

from log_analyser import (AfrFill, AfrFillError, KpaAxis, RpmAxis, TableKind, afr_curve, afr_map,
                          afr_target_table, export_table, read_table, smootherstep)


@pytest.fixture
def fill():
    return AfrFill(lean=15.0, rich=12.5, kpa_lean=50, kpa_rich=90)


def test_smootherstep_ends_and_mid():
    assert smootherstep(np.array([0.0, 0.5, 1.0])).tolist() == [0.0, 0.5, 1.0]
    assert smootherstep(np.array([-1.0, 2.0])).tolist() == [0.0, 1.0]


def test_plateaus_exact(fill):
    k = np.array([0, 20, 50])
    assert np.all(afr_curve(k, fill) == 15.0)
    k = np.array([90, 150, 250])
    assert np.all(afr_curve(k, fill) == 12.5)


def test_midpoint(fill):
    assert afr_curve(70, fill, rounded=False) == pytest.approx(13.75)


def test_monotonic_non_increasing(fill):
    k = np.arange(0, 250, 1)
    for rounded in (True, False):
        assert np.all(np.diff(afr_curve(k, fill, rounded=rounded)) <= 1e-12)


def test_rounding_to_step(fill):
    v = afr_curve(np.arange(50, 91), fill)
    assert np.allclose(v * 10, np.round(v * 10))
    assert all(len(repr(float(x)).split(".")[1]) <= 1 for x in v)


def test_rpm_transition_is_or_with_load():
    f = AfrFill(lean=15.0, rich=12.5, kpa_lean=50, kpa_rich=90, rpm_lean=4000, rpm_rich=6000)
    assert afr_map(2000, 40, f) == 15.0            # low load, low rpm: lean
    assert afr_map(2000, 95, f) == 12.5            # high load: rich at any rpm
    assert afr_map(6500, 40, f) == 12.5            # high rpm: rich at any load
    assert afr_map(5000, 40, f, rounded=False) == pytest.approx(13.75)
    # richer of the two transitions
    assert afr_map(5000, 70, f, rounded=False) == pytest.approx(13.75)
    assert afr_map(5500, 70, f, rounded=False) < 13.75
    # without the rpm transition: same as the load curve
    k = np.arange(0, 120)
    assert np.array_equal(afr_map(7000, k, AfrFill(15.0, 12.5, 50, 90)), afr_curve(k, AfrFill(15.0, 12.5, 50, 90)))


def test_table_with_rpm_transition():
    rpm = RpmAxis.gamma(500, 7000, 16, 1.3)
    kpa = KpaAxis.linspace(20, 100, 12)
    t = afr_target_table(rpm, kpa, AfrFill(15.0, 12.5, 50, 90, rpm_lean=4000, rpm_rich=6000))
    assert np.all(t.values[:, -1] == 12.5)
    assert np.all(np.diff(t.values, axis=0) <= 0) and np.all(np.diff(t.values, axis=1) <= 0)


@pytest.mark.parametrize("kw", [
    dict(rich=15.5, lean=15.0),
    dict(kpa_lean=90, kpa_rich=90),
    dict(rpm_lean=5000, rpm_rich=None),
    dict(rpm_lean=5000, rpm_rich=5000),
    dict(rpm_lean=float("inf"), rpm_rich=float("inf")),
    dict(step=-0.1),
    dict(rich=0),
    dict(lean=float("nan")),
    dict(step=True),
])
def test_validation(kw):
    with pytest.raises(AfrFillError):
        AfrFill(**kw)


def test_table_constant_along_rpm(fill):
    rpm = RpmAxis.gamma(500, 7000, 16, 1.3)
    kpa = KpaAxis.linspace(20, 100, 12)
    t = afr_target_table(rpm, kpa, fill)
    assert t.kind is TableKind.AFR
    assert t.shape == (12, 16)
    assert np.all(t.values == t.values[:, :1])
    assert np.array_equal(t.values[:, 0], afr_curve(kpa.values, fill))


def test_export_roundtrip(fill, tmp_path):
    t = afr_target_table(RpmAxis.gamma(500, 7000, 16, 1.3), KpaAxis.linspace(20, 100, 16), fill)
    p = tmp_path / "afr.table"
    export_table(t, p)
    back = read_table(p, TableKind.AFR)
    assert back.x == t.x and back.y == t.y
    assert np.allclose(back.values, t.values)
