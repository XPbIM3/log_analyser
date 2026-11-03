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
