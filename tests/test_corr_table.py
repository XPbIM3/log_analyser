import numpy as np
import pandas as pd
import pytest

from log_analyser import (AfrFill, KpaAxis, Log, LogError, LogFilter, RpmAxis, Table, TableKind,
                          afr_target_table, corr_grid)

X = RpmAxis([1000, 2000, 3000])
Y = KpaAxis([30, 60, 90])


def make_log(**cols):
    n = len(next(iter(cols.values())))
    base = {"RPM": [2000] * n, "MAP": [60] * n, "CLT": [90] * n, "DFCO": [0] * n,
            "Gwarm": [100] * n, "Accel Enrich": [100] * n, "AFR": [14.7] * n}
    base.update(cols)
    return Log(pd.DataFrame(base))


def test_filter_checks():
    log = make_log(RPM=[0, 2000, 2000, 2000, 2000, 2000],
                   CLT=[90, 50, 90, 90, 90, 90],
                   DFCO=[0, 0, 1, 0, 0, 0],
                   Gwarm=[100, 100, 100, 110, 100, 100],
                   AFR=[14.7, 14.7, 14.7, 14.7, 25.0, 14.7])
    assert LogFilter().mask(log).tolist() == [False, False, False, False, False, True]
    assert LogFilter(rpm_min=None, clt_min=None, exclude_dfco=False, exclude_warmup=False,
                     afr_min=None, afr_max=None).mask(log).all()


def test_accel_tail():
    ae = [100, 120, 100, 100, 100, 100]
    log = make_log(**{"Accel Enrich": ae})
    assert LogFilter(accel_tail=2).mask(log).tolist() == [True, False, False, False, True, True]
    assert LogFilter(accel_tail=0).mask(log).tolist() == [True, False, True, True, True, True]


def test_missing_columns_skipped():
    log = Log(pd.DataFrame({"RPM": [1000, 2000], "AFR": [14.0, 14.0]}))
    assert LogFilter().mask(log).all()
    rep = LogFilter().report(log)
    assert any("CLT" in r and "пропущено" in r for r in rep)


def test_sum_stat_and_clamp():
    # 2 losses at (800, 40) and one below every axis end (100, 40) -> folded into (1000, 30)
    log = Log(pd.DataFrame({"RPM": [800, 800, 100], "MAP": [40, 40, 40],
                            "SyncLoss+": [1, 1, 1]}))
    g = corr_grid(log, X, Y, column="SyncLoss+", stat="sum", min_points=1, clamp=True)
    assert g.counts[0, 0] == 3 and g.values[0, 0] == 3
    assert np.isnan(g.values[1, 1])       # no points there at all
    without_clamp = corr_grid(log, X, Y, column="SyncLoss+", stat="sum", min_points=1)
    assert without_clamp.values[0, 0] == 2 and without_clamp.counts[0, 0] == 2


def test_corr_grid_cells_and_min_points():
    # 12 points at node (2000, 60) with CORR 1.1, 3 points at (1000, 30) with CORR 0.9
    rpm = [2000] * 12 + [1050] * 3
    kpa = [60] * 12 + [32] * 3
    log = Log(pd.DataFrame({"RPM": rpm, "MAP": kpa, "CORR": [1.1] * 12 + [0.9] * 3}))
    g = corr_grid(log, X, Y, min_points=3)
    assert g.counts[1, 1] == 12 and g.counts[0, 0] == 3
    assert g.values[1, 1] == pytest.approx(1.1)
    assert g.values[0, 0] == pytest.approx(0.9)
    assert np.isnan(g.values[2, 2])
    g10 = corr_grid(log, X, Y, min_points=10)
    assert np.isnan(g10.values[0, 0]) and g10.counts[0, 0] == 3
    t = g10.table()
    assert t.kind is TableKind.CORR and t.values[0, 0] == 1.0


def test_corr_grid_stats_and_errors():
    log = Log(pd.DataFrame({"RPM": [2000] * 3, "MAP": [60] * 3, "CORR": [1.0, 1.0, 1.3]}))
    assert corr_grid(log, X, Y, min_points=1, stat="median").values[1, 1] == pytest.approx(1.0)
    assert corr_grid(log, X, Y, min_points=1, stat="mean").values[1, 1] == pytest.approx(1.1)
    assert corr_grid(log, X, Y, min_points=1, stat="q25").values[1, 1] == pytest.approx(1.0)
    assert corr_grid(log, X, Y, min_points=1, stat="q75").values[1, 1] == pytest.approx(1.15)
    with pytest.raises(LogError):
        corr_grid(log, X, Y, stat="max")
    with pytest.raises(LogError):
        corr_grid(log, X, Y, min_points=0)


def test_apply_to_ve():
    log = Log(pd.DataFrame({"RPM": [2000] * 5, "MAP": [60] * 5, "CORR": [1.1] * 5}))
    g = corr_grid(log, X, Y, min_points=5)
    ve = Table(TableKind.VE, X, Y, np.full((3, 3), 50.0))
    new = g.apply_to(ve)
    assert new.values[1, 1] == 55 and new.values[0, 0] == 50
    with pytest.raises(LogError):
        g.apply_to(Table(TableKind.VE, RpmAxis([1000, 2000, 3500]), Y, np.ones((3, 3))))


def test_end_to_end_with_generated_target():
    rng = np.random.default_rng(0)
    n = 2000
    rpm = rng.uniform(800, 3200, n)
    kpa = rng.uniform(25, 95, n)
    target = afr_target_table(X, Y, AfrFill())
    afr = target.lookup(rpm, kpa) * 1.05  # 5 % lean everywhere
    log = Log(pd.DataFrame({"RPM": rpm, "MAP": kpa, "AFR": afr})).with_corr(target)
    g = corr_grid(log, X, Y, min_points=5)
    assert np.allclose(g.values[g.valid], 1.05)


def _ve_log(n=40, ve=50.0, corr=1.1, rpm=2000, kpa=60, col="Current VE"):
    return Log(pd.DataFrame({"RPM": [rpm] * n, "MAP": [kpa] * n, col: [ve] * n, "CORR": [corr] * n}))


def test_ve_grid_without_tune_zero_fill():
    from log_analyser import ve_grid
    r = ve_grid(_ve_log(), X, Y, min_points=10)
    assert r.base is None and r.ve_column == "Current VE"
    assert r.table.kind is TableKind.VE
    assert r.table.values[1, 1] == 55
    assert r.table.values[0, 0] == 0 and r.table.values[2, 2] == 0
    assert r.grid.valid.sum() == 1


def test_ve_grid_with_tune_base_on_new_axes():
    from log_analyser import ve_grid
    base = Table(TableKind.VE, RpmAxis([1000, 3000]), KpaAxis([30, 90]), [[40, 60], [80, 100]])
    r = ve_grid(_ve_log(col="VE (Current"), X, Y, base=base, min_points=10)
    assert r.ve_column == "VE (Current"
    assert r.base.x == X and r.base.y == Y
    # base resampled: (2000, 30) -> 50, (1000, 60) -> 60, (2000, 60) computed from log
    assert r.table.values[0, 1] == 50 and r.table.values[1, 0] == 60
    assert r.table.values[1, 1] == 55


def test_ve_grid_per_row_ve():
    from log_analyser import ve_grid
    log = Log(pd.DataFrame({"RPM": [2000] * 4, "MAP": [60] * 4, "VE1": [40.0, 40.0, 60.0, 60.0],
                            "CORR": [1.0, 1.0, 1.1, 1.1]}))
    assert ve_grid(log, X, Y, min_points=1, stat="mean").table.values[1, 1] == 53  # (40+40+66+66)/4


def test_ve_grid_errors():
    from log_analyser import ve_grid
    with pytest.raises(LogError, match="no VE column"):
        ve_grid(Log(pd.DataFrame({"RPM": [1], "MAP": [1], "CORR": [1.0]})), X, Y)
    afr = Table(TableKind.AFR, X, Y, np.full((3, 3), 14.7))
    with pytest.raises(LogError, match="VE table"):
        ve_grid(_ve_log(), X, Y, base=afr)


def test_corr_grid_iqr():
    vals = [0.9, 1.0, 1.0, 1.1, 1.3]
    log = Log(pd.DataFrame({"RPM": [2000] * 5, "MAP": [60] * 5, "CORR": vals}))
    g = corr_grid(log, X, Y, min_points=5)
    q25, q75 = np.percentile(vals, [25, 75])
    assert g.iqr[1, 1] == pytest.approx(q75 - q25)
    assert np.isnan(g.iqr[0, 0])
    assert np.isnan(corr_grid(log, X, Y, min_points=6).iqr[1, 1])


def test_filter_rpm_dot_and_tps():
    log = make_log(**{"rpm/s": [-50, 0, 120, 0], "TPS": [10, 0, 5, 3.5]})
    assert LogFilter().mask(log).all()                                  # both off by default
    assert LogFilter(rpm_dot_min=0).mask(log).tolist() == [False, True, True, True]
    assert LogFilter(tps_above=0).mask(log).tolist() == [True, False, True, True]
    assert LogFilter(tps_above=4).mask(log).tolist() == [True, False, True, False]
    rep = LogFilter(rpm_dot_min=0, tps_above=0).report(make_log(RPM=[2000]))
    assert any("rpm/s" in r and "пропущено" in r for r in rep)
