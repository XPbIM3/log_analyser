import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

# Tk on Windows (nt) and macOS (Cocoa) does not need X11; elsewhere require DISPLAY
if os.name != "nt" and sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)

from log_analyser.ui.app import MainWindow  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def win():
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"Tk unavailable: {e}")
    root.geometry("1280x820")
    w = MainWindow(root)
    w.pack(fill=tk.BOTH, expand=True)
    root.update()
    yield w
    root.destroy()


def wait_loaded(w, timeout=30):
    end = time.time() + timeout
    while w.logs._pending and time.time() < end:
        w.update()
        time.sleep(0.05)
    assert not w.logs._pending


def test_sync_loss_map(win):
    msl = sorted(str(p) for p in FIXTURES.glob("*.msl"))
    if not msl:
        pytest.skip("no sample logs")
    win.logs.load_tune(str(FIXTURES / "CurrentTune.msq"))
    win.logs.load_logs(msl)
    wait_loaded(win)
    win.sync.refresh()
    g = win.sync.grid_result
    assert g is not None
    assert int(np.nansum(g.values)) == 1       # the fixture log has exactly one sync loss
    i, j = (int(v) for v in np.argwhere(g.values > 0)[0])
    win.sync._cell_clicked(i, j)
    assert f"потерь 1" in win.sync.cell_info.cget("text")
    assert f"точек {g.counts[i, j]}" in win.sync.cell_info.cget("text")
    # same on the tune VE axes
    win.sync.axes_src.set("ve")
    win.sync.refresh()
    assert int(np.nansum(win.sync.grid_result.values)) == 1


def test_shift_ms_control(win):
    msl = sorted(str(p) for p in FIXTURES.glob("*.msl"))
    if not msl:
        pytest.skip("no sample logs")
    win.logs.load_logs(msl)
    wait_loaded(win)
    assert float(win.logs.shift_ms.get()) == 270.0            # recommended default
    items = win.logs.tree.get_children()
    assert all(win.logs.tree.item(i, "values")[2] for i in items)          # Гц detected
    assert all(l.shift_rows == 4 and l.rate_hz == pytest.approx(15.6, rel=0.02)
               for l in win.session.logs.values())
    rows_default = {i: int(win.logs.tree.item(i, "values")[3]) for i in items}

    win.logs.shift_ms.set(0)
    win.logs.reload_all()
    wait_loaded(win)
    rows_off = {i: int(win.logs.tree.item(i, "values")[3]) for i in items}
    assert all(rows_off[i] > rows_default[i] for i in items)   # no rows dropped
    assert all(l.shift_rows == 0 for l in win.session.logs.values())

    win.logs.shift_ms.set(540)
    win.logs.schedule_reload()                                 # debounced path
    end = time.time() + 5
    while time.time() < end and (win.logs._reload_job is not None or win.logs._pending):
        win.update()
        time.sleep(0.02)
    assert not win.logs._pending and win.logs._reload_job is None
    assert all(l.shift_rows == 8 for l in win.session.logs.values())
    rows_big = {i: int(win.logs.tree.item(i, "values")[3]) for i in items}
    assert all(rows_big[i] < rows_default[i] for i in items)


def test_detect_shift_button(win, monkeypatch):
    import log_analyser.ui.logs_tab as logs_tab
    from log_analyser.shift_scan import ShiftEstimate
    msl = sorted(str(p) for p in FIXTURES.glob("*.msl"))
    if not msl:
        pytest.skip("no sample logs")
    win.logs.load_logs(msl)
    wait_loaded(win)
    fake = ShiftEstimate(rate_hz=15.6, best_level=4, best_diff=4, best_err=4,
                         delay_ms=500.0, confidence="confident")
    monkeypatch.setattr(logs_tab, "scan_file", lambda p: fake)
    win.logs.detect_shift()
    end = time.time() + 5
    while time.time() < end and win.logs._shift_dialog is None:
        win.update()
        time.sleep(0.02)
    assert win.logs._shift_dialog is not None
    assert "500" in win.logs._shift_note.cget("text")
    win.logs._apply_shift()
    assert float(win.logs.shift_ms.get()) == 500.0           # rounded to the spinbox step
    wait_loaded(win)
    assert all(log.shift_rows == 8 for log in win.session.logs.values())   # 500 ms @ 15.6 Hz


def test_canvas_texts_have_explicit_fill(win):
    # macOS Dark Mode: canvas text without an explicit fill defaults to systemTextColor
    # (white) -> invisible on the white canvases (axis labels, slider labels)
    def walk(w):
        yield w
        for ch in w.winfo_children():
            yield from walk(ch)

    for w in walk(win):
        if isinstance(w, tk.Canvas):
            for item in w.find_all():
                if w.type(item) == "text":
                    fill = w.itemcget(item, "fill")
                    assert fill and fill != "systemTextColor", f"text without fill: {w}"


def test_no_logs_message(win):
    win.corr.refresh()
    assert win.corr.grid_result is None


def test_corr_colors_red_is_lean(win):
    # red = leaner than target (add fuel), blue = richer (remove fuel) — user-facing
    # legend must not drift away from the cmap direction again
    msl = sorted(str(p) for p in FIXTURES.glob("*.msl"))
    if not msl:
        pytest.skip("no sample logs")
    win.logs.load_logs(msl)
    wait_loaded(win)
    win.corr.refresh()
    assert win.corr.view._data[5] == "RdBu_r"                  # cmap: high CORR -> red
    assert "красный — беднее" in win.corr.view._data[4]        # legend says the same


def test_logs_tune_and_correction(win):
    msl = sorted(str(p) for p in FIXTURES.glob("*.msl"))
    if not msl:
        pytest.skip("no sample logs")
    win.logs.load_tune(str(FIXTURES / "CurrentTune.msq"))
    win.logs.load_logs(msl)
    wait_loaded(win)
    assert len(win.session.logs) == len(msl)
    assert len(win.logs.tree.get_children()) == len(msl)

    # default: target axes; VE new is on those axes, empty cells from the tune VE
    win.corr.refresh()
    g = win.corr.grid_result
    assert g is not None and win.corr.ve_new is not None and win.corr.ve_base is not None
    assert win.corr.ve_new.x == g.x and win.corr.ve_new.y == g.y
    assert np.array_equal(win.corr.ve_new.values[~g.valid], win.corr.ve_base.values[~g.valid])
    win.corr.axes_src.set("ve")
    for src in ("generated", "tune", "log"):
        win.corr.target_src.set(src)
        win.corr.refresh()
        g = win.corr.grid_result
        assert g is not None, src
        assert g.valid.any()
        assert win.corr.ve_new is not None
        assert win.corr.ve_new.x == win.session.tables["veTable"].x
    for mode in ("corr", "counts", "ve"):
        win.corr.mode.set(mode)
        win.corr.refresh()
    i, j = np.argwhere(g.valid)[0]
    win.corr._cell_clicked(int(i), int(j))
    assert "точек" in win.corr.cell_info.cget("text")

    win.logs.tree.selection_set(win.logs.tree.get_children()[0])
    win.logs.remove_selected()
    assert len(win.session.logs) == len(msl) - 1
    win.logs.clear()
    win.corr.refresh()
    assert win.corr.grid_result is None


def test_duration_text():
    import pandas as pd
    from log_analyser import Log
    from log_analyser.ui.logs_tab import duration_text
    t = list(np.arange(0, 60, 0.1)) + list(np.arange(660, 720, 0.1))   # 2 min of data, 10 min pause
    log = Log(pd.DataFrame({"Time": t}))
    assert duration_text(log) == "12.0 (паузы 10.0)"
    assert duration_text(Log(pd.DataFrame({"Time": [np.nan, 0.0, 120.0]}))) == "2.0 (паузы 2.0)"
    assert duration_text(Log(pd.DataFrame({"RPM": [1, 2]}))) == ""


def test_ve_new_without_tune_fills_zero(win):
    msl = sorted(str(p) for p in FIXTURES.glob("*.msl"))
    if not msl:
        pytest.skip("no sample logs")
    win.logs.load_logs(msl)
    wait_loaded(win)
    win.corr.mode.set("ve")
    win.corr.refresh()
    g, ve = win.corr.grid_result, win.corr.ve_new
    assert ve is not None and win.corr.ve_base is None
    assert np.all(ve.values[~g.valid] == 0) and np.all(ve.values[g.valid] > 0)
    assert "instate" not in str(win.corr.export_btn.state())
    assert "disabled" not in win.corr.export_btn.state()


def test_cell_panel_and_spread(win):
    msl = sorted(str(p) for p in FIXTURES.glob("*.msl"))
    if not msl:
        pytest.skip("no sample logs")
    win.logs.load_logs(msl)
    wait_loaded(win)
    c = win.corr
    c.refresh()
    g = c.grid_result
    i, j = (int(v) for v in np.unravel_index(np.argmax(g.counts), g.counts.shape))
    c.view.selected = (i, j)
    c._cell_clicked(i, j)
    p = c.cell_panel
    assert len(p.points) == g.counts[i, j]       # same cell bounds as the aggregation
    assert "медиана" in p.title.cget("text")
    assert p.x_col.get() == "rpm/s"
    assert p.y_col.get() == "CORR"
    p.x_col.set("TPS")
    p._draw()
    p.y_col.set("Lambda")
    p.x_col.set("PW")
    p._draw()
    assert p.ax_scat.get_ylabel().startswith("Lambda") and p.ax_scat.get_xlabel().startswith("PW")
    assert any("r =" in t.get_text() for t in p.ax_scat.get_legend().get_texts())
    p._swap()
    assert (p.x_col.get(), p.y_col.get()) == ("Lambda", "PW")

    c.mode.set("spread")
    c.refresh()                                  # selection kept -> panel refreshed
    assert c.cell_panel.points is not None
    assert np.isfinite(g.iqr[g.valid]).all()

    c.f_rpmdot_on.set(False)
    c.f_tps_on.set(False)
    c.refresh()
    n_before = int(c.grid_result.counts.sum())
    c.f_rpmdot_on.set(True)
    c.f_tps_on.set(True)
    c.f_tps.set("2")
    c.refresh()
    assert int(c.grid_result.counts.sum()) < n_before
    assert "после фильтра" in c.status.cget("text")

    c.view.selected = None
    c.min_points.set("100000")                    # nothing valid any more
    c.refresh()
    assert c.cell_panel.points is None
