import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from log_analyser import KpaAxis, Log, LogError, RpmAxis, Table, TableKind, cell_bounds, cell_points, plot_cell

X = RpmAxis([1000, 2000, 4000])
Y = KpaAxis([30, 50, 100])


def test_bounds_middle():
    assert cell_bounds(X, Y, 2000, 50) == (1500, 3000, 40, 75)
    assert cell_bounds(X, Y, 2000, 50, overlap=1.0) == (1000, 4000, 30, 100)


def test_bounds_edges():
    assert cell_bounds(X, Y, 1000, 30) == (500, 1500, 20, 40)
    assert cell_bounds(X, Y, 4000, 100) == (3000, 5000, 75, 125)


def test_bounds_errors():
    with pytest.raises(LogError, match="not a node"):
        cell_bounds(X, Y, 1500, 50)
    with pytest.raises(LogError, match="overlap"):
        cell_bounds(X, Y, 2000, 50, overlap=0)


@pytest.fixture
def log():
    rpm = np.array([1400, 1500, 2000, 3000, 3100, 2500, 2500])
    kpa = np.array([50, 50, 50, 50, 50, 39, 76])
    df = pd.DataFrame({"RPM": rpm, "MAP": kpa, "AFR": np.linspace(13, 15, 7), "VE1": np.arange(7) + 60})
    return Log(df, sources=["synthetic"], units={"RPM": "rpm", "AFR": "O2"})


def test_cell_points(log):
    pts = cell_points(log, X, Y, 2000, 50)
    assert pts.column("RPM").tolist() == [1500, 2000, 3000]
    assert len(cell_points(log, X, Y, 2000, 50, overlap=1.0)) == 7


def test_plot_cell(tmp_path, log):
    afr = Table(TableKind.AFR, X, Y, np.full((3, 3), 14.0))
    p = tmp_path / "cell.png"
    fig = plot_cell(log.with_corr(afr), X, Y, 2000, 50, what="CORR", vs="RPM", path=p)
    assert p.exists() and p.stat().st_size > 0
    ax = fig.axes[0]
    assert "3 points" in ax.get_title()
    assert ax.get_xlabel() == "RPM [rpm]"
    assert ax.get_ylabel() == "CORR"
