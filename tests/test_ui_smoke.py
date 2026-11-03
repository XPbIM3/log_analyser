import os

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

if os.name != "nt" and not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)

from log_analyser import afr_curve  # noqa: E402
from log_analyser.ui.afr_target_app import AfrTargetApp  # noqa: E402


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"Tk unavailable: {e}")
    root.geometry("1100x700")
    a = AfrTargetApp(root)
    a.pack(fill=tk.BOTH, expand=True)
    root.update()
    yield a
    root.destroy()


def test_builds_default_table(app):
    assert app.table.shape == (16, 16)
    assert app.table.values[0, 0] == app.fill.fill.lean or app.table.values[0, 0] <= app.fill.fill.lean


def test_param_change_updates_table(app):
    app.fill._rows["rich"].set(11.0)
    app.kpa.stop.set(200)
    app.root.update()
    app.update_now()
    assert app.fill.fill.rich == 11.0
    assert app.kpa.axis.values[-1] == 200
    assert app.table.values[-1, 0] == 11.0
    assert np.array_equal(app.table.values[:, 0], afr_curve(app.kpa.axis.values, app.fill.fill))


def test_invalid_input_keeps_last_valid(app):
    before = app.rpm.axis
    app.rpm.start.var.set("abc")
    app.root.update()
    assert app.rpm.axis == before
    app.rpm.start.set(8000)  # min > max
    app.root.update()
    assert app.rpm.axis == before
    assert app.rpm._error.cget("text")


def test_range_sliders(app):
    app.update_now()
    col0 = app.table.values[:, 0].copy()
    # load slider moves the transition
    app.kpa_slider.set(60, 80)
    app.root.update()
    app.update_now()
    assert (app.fill.fill.kpa_lean, app.fill.fill.kpa_rich) == (60, 80)
    assert not np.array_equal(app.table.values[:, 0], col0)
    # rpm off: constant along rpm
    app.rpm_on.set(False)
    app._ranges_changed()
    app.update_now()
    assert np.all(app.table.values == app.table.values[:, :1])
    # rpm on: the transition reaches rich at the last rpm column
    app.rpm_on.set(True)
    app._ranges_changed()
    app.rpm_slider.set(4000, 5000)
    app.root.update()
    app.update_now()
    assert (app.fill.fill.rpm_lean, app.fill.fill.rpm_rich) == (4000, 5000)
    assert np.all(app.table.values[:, -1] == app.fill.fill.rich)
    app.rpm_on.set(False)
    app._ranges_changed()
    app.update_now()
    assert app.fill.fill.rpm_lean is None


def test_slider_drag_maps_to_axis(app):
    s = app.kpa_slider
    s.canvas.update()
    kpa = app.kpa.axis
    # pixel of a node centre maps back to that node
    for k in (0, 7, len(kpa) - 1):
        v = float(kpa.values[k])
        assert abs(s._value(s._pix(v)) - v) <= 1
    # drag the rich handle to the top end
    top = s._pix(float(kpa.values[-1]))
    s._press(type("E", (), {"x": 10, "y": s._pix(s.hi)})())
    s._move(type("E", (), {"x": 10, "y": top})())
    assert s.hi == kpa.values[-1]


def test_slider_limits_zero_to_double_max(app):
    for s, axis in ((app.kpa_slider, app.kpa.axis), (app.rpm_slider, app.rpm.axis)):
        s.canvas.update()
        top = 2 * float(axis.values[-1])
        assert s.limits == (0.0, top)
        # track ends map to the limits
        length = s._length()
        lo_end, hi_end = (length, 0) if s.vertical else (0, length)
        assert s._value(lo_end) == 0 and s._value(hi_end) == top
        # values beyond the axis round-trip through the overrun zones
        for v in (0.0, top):
            assert abs(s._value(s._pix(v)) - v) <= s._q
        s.set(-50, 10 * top)
        assert (s.lo, s.hi) == (0.0, top)


def test_sliders_line_up_with_table_cells(app):
    for _ in range(5):
        app.root.update()
    lw, th, cw, ch, rows, cols = app.grid_view._geom
    gv, ks, rs = app.grid_view.canvas, app.kpa_slider, app.rpm_slider
    for i in (0, rows // 2, rows - 1):
        y_tab = gv.winfo_rooty() + th + (rows - 1 - i + 0.5) * ch
        y_sl = ks.canvas.winfo_rooty() + ks._pix(float(app.kpa.axis.values[i]))
        assert abs(y_tab - y_sl) < 1
    for j in (0, cols // 2, cols - 1):
        x_tab = gv.winfo_rootx() + lw + (j + 0.5) * cw
        x_sl = rs.canvas.winfo_rootx() + rs._pix(float(app.rpm.axis.values[j]))
        assert abs(x_tab - x_sl) < 1
