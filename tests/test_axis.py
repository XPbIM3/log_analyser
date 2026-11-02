import numpy as np
import pytest

from log_analyser import Axis, AxisError, KpaAxis, RpmAxis

LEGACY_RPM = [400, 700, 1100, 1500, 1800, 2200, 2600, 3000, 3300, 3700, 4100, 4500, 4800, 5200, 5600, 6000]
LEGACY_KPA = [28, 36, 46, 54, 62, 72, 80, 90, 98, 106, 116, 124, 134, 142, 150, 160]


def test_valid_axes():
    rpm = RpmAxis(LEGACY_RPM)
    kpa = KpaAxis(LEGACY_KPA)
    assert len(rpm) == 16 and len(kpa) == 16
    assert rpm.values.dtype == np.int64
    assert list(rpm) == LEGACY_RPM
    assert rpm.quantum == 50 and kpa.quantum == 2


def test_float_integers_accepted():
    assert list(KpaAxis([30.0, 40.0])) == [30, 40]


@pytest.mark.parametrize("cls, values, bad", [(RpmAxis, [400, 425, 500], 425), (KpaAxis, [30, 31, 40], 31)])
def test_not_multiple_of_quantum(cls, values, bad):
    with pytest.raises(AxisError, match=str(bad)):
        cls(values)


def test_non_integer():
    with pytest.raises(AxisError, match="integers"):
        KpaAxis([30, 30.5])


def test_duplicates():
    with pytest.raises(AxisError, match=r"duplicate values: \[500\]"):
        RpmAxis([400, 500, 500, 600])


def test_decreasing():
    with pytest.raises(AxisError, match="index 2: 600 > 550"):
        RpmAxis([400, 600, 550])


def test_negative():
    with pytest.raises(AxisError, match=">= 0"):
        KpaAxis([-2, 10])


@pytest.mark.parametrize("values", [[], [400], [[400, 500]], [400, np.nan], ["a"]])
def test_bad_shape_or_content(values):
    with pytest.raises(AxisError):
        RpmAxis(values)


def test_quantum_override():
    a = RpmAxis([425, 450], quantum=25)
    assert a.quantum == 25
    assert RpmAxis.quantum == 50  # class default untouched
    with pytest.raises(AxisError):
        RpmAxis([425, 450])


@pytest.mark.parametrize("q", [0, -2, 1.5, True])
def test_bad_quantum(q):
    with pytest.raises(AxisError, match="quantum"):
        KpaAxis([2, 4], quantum=q)
    with pytest.raises(AxisError, match="quantum"):
        KpaAxis.linspace(2, 40, 4, quantum=q)


def test_linspace_legacy_ranges():
    rpm = RpmAxis.linspace(400, 6000, 16)
    kpa = KpaAxis.linspace(28, 160, 16)
    assert isinstance(rpm, RpmAxis) and isinstance(kpa, KpaAxis)
    assert len(rpm) == 16 and len(kpa) == 16
    assert rpm[0] == 400 and rpm[-1] == 6000
    assert kpa[0] == 28 and kpa[-1] == 160
    assert np.all(rpm.values % 50 == 0) and np.all(kpa.values % 2 == 0)


def test_linspace_does_not_fit():
    with pytest.raises(AxisError, match="do not fit"):
        KpaAxis.linspace(30, 34, 16)


def test_values_read_only():
    a = KpaAxis(LEGACY_KPA)
    with pytest.raises(ValueError):
        a.values[0] = 0


def test_input_not_aliased():
    src = np.array([30, 40], dtype=np.int64)
    a = KpaAxis(src)
    src[0] = 32
    assert a[0] == 30


def test_equality():
    assert KpaAxis([2, 4]) == KpaAxis([2, 4])
    assert KpaAxis([2, 4]) != KpaAxis([2, 6])
    assert KpaAxis([100, 200]) != RpmAxis([100, 200])
    assert Axis([2, 4]) != KpaAxis([2, 4])
    assert len({KpaAxis([2, 4]), KpaAxis([2, 4])}) == 1


# ---------------------------------------------------------------- gamma / from_curve

def _check_grid(a, start, stop, n):
    assert len(a) == n
    assert a[0] == start and a[-1] == stop
    assert np.all(a.values % a.quantum == 0)
    assert np.all(a.steps >= a.quantum)


def test_steps():
    assert KpaAxis([20, 30, 50]).steps.tolist() == [10, 20]


def test_gamma_one_equals_linspace():
    assert RpmAxis.gamma(400, 6000, 16, 1.0) == RpmAxis.linspace(400, 6000, 16)
    assert KpaAxis.gamma(28, 160, 16, 1.0) == KpaAxis.linspace(28, 160, 16)


@pytest.mark.parametrize("cls, start, stop", [(RpmAxis, 400, 6000), (KpaAxis, 28, 160)])
def test_gamma_dense_at_start(cls, start, stop):
    a = cls.gamma(start, stop, 16, 2.0)
    assert isinstance(a, cls)
    _check_grid(a, start, stop, 16)
    assert a.steps[0] < a.steps[-1]
    assert a.steps[-1] == a.steps.max()


def test_gamma_dense_at_end():
    a = RpmAxis.gamma(400, 6000, 16, 0.5)
    _check_grid(a, 400, 6000, 16)
    assert a.steps[0] > a.steps[-1]


def test_gamma_collapse_is_repaired():
    # raw first steps are < 2 kPa and would round onto each other
    a = KpaAxis.gamma(28, 160, 16, 3.0)
    _check_grid(a, 28, 160, 16)
    assert a.steps.min() == 2
    assert a.values[:4].tolist() == [28, 30, 32, 34]


def test_tight_fit_gives_minimal_grid():
    a = KpaAxis.gamma(30, 60, 16, 2.0)
    assert a.values.tolist() == list(range(30, 61, 2))


@pytest.mark.parametrize("g", [0, -1, np.nan])
def test_bad_gamma(g):
    with pytest.raises(AxisError, match="gamma"):
        RpmAxis.gamma(400, 6000, 16, g)


@pytest.mark.parametrize("start, stop", [(425, 6000), (400, 6010), (400.5, 6000)])
def test_ends_must_be_multiples(start, stop):
    with pytest.raises(AxisError, match="multiple of 50"):
        RpmAxis.linspace(start, stop, 16)


@pytest.mark.parametrize("start, stop", [(6000, 400), (400, 400)])
def test_start_before_stop(start, stop):
    with pytest.raises(AxisError, match="start"):
        RpmAxis.linspace(start, stop, 4)


@pytest.mark.parametrize("n", [1, 0, 2.5])
def test_bad_n(n):
    with pytest.raises(AxisError, match="n must be"):
        RpmAxis.linspace(400, 6000, n)


def test_custom_curve():
    a = RpmAxis.from_curve(400, 6000, 16, lambda t: np.sin(t * np.pi / 2))
    _check_grid(a, 400, 6000, 16)
    assert a.steps[0] > a.steps[-1]


@pytest.mark.parametrize("curve, msg", [
    (lambda t: 1 - t, "0->0"),
    (lambda t: t * 0.5, "0->0"),
    (lambda t: np.where(t < 0.5, t * 1.5, t), "non-decreasing"),
    (lambda t: t[:-1], "finite values"),
    (lambda t: t + np.nan, "finite values"),
])
def test_bad_curve(curve, msg):
    with pytest.raises(AxisError, match=msg):
        RpmAxis.from_curve(400, 6000, 16, curve)


def test_quantum_override_in_constructor():
    a = RpmAxis.gamma(500, 7000, 16, 1.5, quantum=100)
    assert a.quantum == 100
    _check_grid(a, 500, 7000, 16)
