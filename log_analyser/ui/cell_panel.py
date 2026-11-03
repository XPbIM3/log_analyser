"""Diagnostics of one table cell: CORR histogram and a scatter of any column vs any column."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..cell import cell_points
from ..corr_table import CorrGrid
from ..datalog import SOURCE_COLUMN, Log

PREFERRED_X = ("rpm/s", "TPS DOT", "TPS", "MAP", "RPM", "CLT", "Time")
DEFAULT_Y = "CORR"
MAX_LEGEND_FILES = 10


class CellPanel(ttk.Frame):
    """Shows the points of the selected cell. Call :meth:`show` / :meth:`clear`."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self._log: Log | None = None
        self._grid: CorrGrid | None = None
        self._cell: tuple[int, int] | None = None
        self.points: Log | None = None

        bar = ttk.Frame(self)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Y:").pack(side=tk.LEFT)
        self.y_col = tk.StringVar(value=DEFAULT_Y)
        self.y_box = ttk.Combobox(bar, textvariable=self.y_col, state="readonly", width=18)
        self.y_box.pack(side=tk.LEFT, padx=4)
        self.y_box.bind("<<ComboboxSelected>>", lambda _e: self._draw())
        ttk.Button(bar, text="⇄", width=3, command=self._swap).pack(side=tk.LEFT)
        ttk.Label(bar, text="X:").pack(side=tk.LEFT, padx=(4, 0))
        self.x_col = tk.StringVar(value="rpm/s")
        self.x_box = ttk.Combobox(bar, textvariable=self.x_col, state="readonly", width=18)
        self.x_box.pack(side=tk.LEFT, padx=4)
        self.x_box.bind("<<ComboboxSelected>>", lambda _e: self._draw())
        self.trend = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="тренд", variable=self.trend,
                        command=self._draw).pack(side=tk.LEFT, padx=(8, 0))
        self.title = ttk.Label(bar, font=("TkDefaultFont", 10, "bold"))
        self.title.pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(7, 2.6), dpi=96, layout="constrained")
        self.ax_hist = self.figure.add_subplot(1, 2, 1)
        self.ax_scat = self.figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.clear()

    # ------------------------------------------------------------------ public

    def clear(self, message: str = "Кликните по ячейке таблицы, чтобы увидеть её точки") -> None:
        self._cell = None
        self.points = None
        self.title.configure(text="")
        for ax in (self.ax_hist, self.ax_scat):
            ax.clear()
            ax.set_axis_off()
        self.ax_hist.text(0.02, 0.5, message, transform=self.ax_hist.transAxes, color="#666666")
        self.canvas.draw_idle()

    def show(self, log: Log, grid: CorrGrid, i: int, j: int) -> None:
        """Points of cell (row ``i`` = ``grid.y[i]``, column ``j`` = ``grid.x[j]``) of ``log`` (with CORR)."""
        self._log, self._grid, self._cell = log, grid, (i, j)
        cols = [c for c in log.columns if c != SOURCE_COLUMN and log.data[c].dtype.kind in "if"]
        self.x_box.configure(values=cols)
        self.y_box.configure(values=cols)
        if self.x_col.get() not in cols:
            self.x_col.set(next((c for c in PREFERRED_X if c in cols), cols[0] if cols else ""))
        if self.y_col.get() not in cols:
            self.y_col.set(DEFAULT_Y if DEFAULT_Y in cols else (cols[0] if cols else ""))
        self.points = cell_points(log, grid.x, grid.y, int(grid.x[j]), int(grid.y[i]),
                                  overlap=grid.overlap)
        self._draw()

    # ------------------------------------------------------------------ drawing

    def _swap(self) -> None:
        x, y = self.x_col.get(), self.y_col.get()
        self.x_col.set(y)
        self.y_col.set(x)
        self._draw()

    def _label(self, pts: Log, col: str) -> str:
        unit = pts.units.get(col, "")
        return f"{col} [{unit}]" if unit else col

    def _draw(self) -> None:
        if self._cell is None or self.points is None:
            return
        g, (i, j), pts = self._grid, self._cell, self.points
        self.title.configure(text=f"RPM {g.x[j]}, kPa {g.y[i]}: {len(pts)} точек")
        for ax in (self.ax_hist, self.ax_scat):
            ax.clear()
            ax.set_axis_on()
        if not len(pts):
            self.ax_hist.set_axis_off()
            self.ax_scat.set_axis_off()
            self.ax_hist.text(0.02, 0.5, "в ячейке нет точек", transform=self.ax_hist.transAxes,
                              color="#666666")
            self.canvas.draw_idle()
            return

        corr = pts.column("CORR").astype(float)
        src = pts.data[SOURCE_COLUMN].to_numpy() if SOURCE_COLUMN in pts.columns else np.array([""] * len(pts))
        files = list(dict.fromkeys(src))  # in log order
        cmap = colormaps["tab10"]
        colors = {f: cmap(k % 10) for k, f in enumerate(files)}
        med, mean = float(np.median(corr)), float(np.mean(corr))
        q25, q75 = (float(v) for v in np.percentile(corr, [25, 75]))
        v = g.values[i, j]
        in_table = (f"в таблице {100 * (v - 1):+.1f} %" if np.isfinite(v)
                    else f"в таблице — (< {g.min_points} точек)")
        self.title.configure(text=f"RPM {g.x[j]}, kPa {g.y[i]}: {len(pts)} точек · "
                                  f"медиана {100 * (med - 1):+.1f} % · "
                                  f"IQR {100 * (q75 - q25):.1f} п.п. · {in_table}")

        # histogram
        ax = self.ax_hist
        ax.hist(corr, bins=min(40, max(5, int(np.sqrt(len(corr))))), color="#7a9cc6", edgecolor="white")
        ax.axvspan(q25, q75, color="#f0c040", alpha=0.25, label=f"IQR {100 * (q75 - q25):.1f} п.п.")
        ax.axvline(med, color="black", lw=1.5, label=f"медиана {100 * (med - 1):+.1f} %")
        ax.axvline(mean, color="black", lw=1.0, ls="--", label=f"среднее {100 * (mean - 1):+.1f} %")
        ax.axvline(1.0, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("CORR (AFR изм. / target)")
        ax.set_ylabel("точек")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        # scatter Y vs X, coloured by file
        ax = self.ax_scat
        xc, yc = self.x_col.get(), self.y_col.get()
        if xc in pts.columns and yc in pts.columns:
            xv = pts.column(xc).astype(float)
            yv = pts.column(yc).astype(float)
            for f in files:
                m = src == f
                ax.scatter(xv[m], yv[m], s=6, color=colors[f], label=f, alpha=0.7)
            ok = np.isfinite(xv) & np.isfinite(yv)
            if self.trend.get() and ok.sum() >= 3 and np.ptp(xv[ok]) > 0:
                k, b = np.polyfit(xv[ok], yv[ok], 1)
                r = float(np.corrcoef(xv[ok], yv[ok])[0, 1]) if np.ptp(yv[ok]) > 0 else 0.0
                xs = np.array([xv[ok].min(), xv[ok].max()])
                ax.plot(xs, k * xs + b, color="black", lw=1.2, ls="--",
                        label=f"y = {k:.4g}·x + {b:.4g}, r = {r:+.2f}")
            elif ok.any():
                ax.axhline(float(np.median(yv[ok])), color="black", lw=1.0)
            if yc == "CORR":
                ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
            ax.set_xlabel(self._label(pts, xc))
            ax.set_ylabel(self._label(pts, yc))
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        keep = [k for k, l in enumerate(labels)
                if not (l in colors and not 1 < len(files) <= MAX_LEGEND_FILES)]
        if keep:
            ax.legend([handles[k] for k in keep], [labels[k] for k in keep],
                      fontsize=6, loc="best", markerscale=2)

        self.canvas.draw_idle()
