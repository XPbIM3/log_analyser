"""Diagnostics of one table cell: CORR histogram, CORR vs a chosen column, stats per file."""
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
        ttk.Label(bar, text="CORR против:").pack(side=tk.LEFT)
        self.x_col = tk.StringVar(value="rpm/s")
        self.x_box = ttk.Combobox(bar, textvariable=self.x_col, state="readonly", width=18)
        self.x_box.pack(side=tk.LEFT, padx=4)
        self.x_box.bind("<<ComboboxSelected>>", lambda _e: self._draw())
        self.title = ttk.Label(bar, font=("TkDefaultFont", 10, "bold"))
        self.title.pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(7, 2.6), dpi=96, layout="constrained")
        self.ax_hist = self.figure.add_subplot(1, 2, 1)
        self.ax_scat = self.figure.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.stats = tk.Text(body, width=34, height=10, font="TkFixedFont", relief=tk.FLAT,
                             background=self.winfo_toplevel().cget("background"))
        self.stats.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0))
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
        self._set_stats("")
        self.canvas.draw_idle()

    def show(self, log: Log, grid: CorrGrid, i: int, j: int) -> None:
        """Points of cell (row ``i`` = ``grid.y[i]``, column ``j`` = ``grid.x[j]``) of ``log`` (with CORR)."""
        self._log, self._grid, self._cell = log, grid, (i, j)
        cols = [c for c in log.columns if c != SOURCE_COLUMN and log.data[c].dtype.kind in "if"]
        self.x_box.configure(values=cols)
        if self.x_col.get() not in cols:
            self.x_col.set(next((c for c in PREFERRED_X if c in cols), cols[0] if cols else ""))
        self.points = cell_points(log, grid.x, grid.y, int(grid.x[j]), int(grid.y[i]),
                                  overlap=grid.overlap)
        self._draw()

    # ------------------------------------------------------------------ drawing

    def _set_stats(self, text: str) -> None:
        self.stats.configure(state=tk.NORMAL)
        self.stats.delete("1.0", tk.END)
        self.stats.insert("1.0", text)
        self.stats.configure(state=tk.DISABLED)

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
            self._set_stats("")
            self.canvas.draw_idle()
            return

        corr = pts.column("CORR").astype(float)
        src = pts.data[SOURCE_COLUMN].to_numpy() if SOURCE_COLUMN in pts.columns else np.array([""] * len(pts))
        files = list(dict.fromkeys(src))  # in log order
        cmap = colormaps["tab10"]
        colors = {f: cmap(k % 10) for k, f in enumerate(files)}
        med, mean = float(np.median(corr)), float(np.mean(corr))
        q25, q75 = (float(v) for v in np.percentile(corr, [25, 75]))

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

        # scatter vs chosen column, coloured by file
        ax = self.ax_scat
        xc = self.x_col.get()
        if xc in pts.columns:
            xv = pts.column(xc).astype(float)
            for f in files:
                m = src == f
                ax.scatter(xv[m], corr[m], s=6, color=colors[f], label=f, alpha=0.7)
            unit = pts.units.get(xc, "")
            ax.set_xlabel(f"{xc} [{unit}]" if unit else xc)
        ax.axhline(med, color="black", lw=1.0)
        ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
        ax.set_ylabel("CORR")
        ax.grid(True, alpha=0.3)
        if 1 < len(files) <= MAX_LEGEND_FILES:
            ax.legend(fontsize=6, loc="best", markerscale=2)

        # stats text
        lines = [
            f"точек    {len(corr)}",
            f"медиана  {100 * (med - 1):+6.1f} %",
            f"среднее  {100 * (mean - 1):+6.1f} %",
            f"IQR      {100 * (q75 - q25):6.1f} п.п.",
            f"5…95 %   {100 * (np.percentile(corr, 5) - 1):+.1f} … {100 * (np.percentile(corr, 95) - 1):+.1f}",
        ]
        v = g.values[i, j]
        lines.append(f"в таблице {100 * (v - 1):+.1f} % ({g.stat})" if np.isfinite(v)
                     else f"в таблице — (< {g.min_points} точек)")
        lines.append("")
        lines.append("по файлам (точек, медиана):")
        by_file = sorted(files, key=lambda f: -int(np.sum(src == f)))
        for f in by_file[:8]:
            c = corr[src == f]
            name = f[:-4] if f.lower().endswith(".msl") else f
            lines.append(f" {name[:19]:19} {c.size:6} {100 * (np.median(c) - 1):+6.1f}")
        if len(by_file) > 8:
            lines.append(f" … ещё {len(by_file) - 8}")
        self._set_stats("\n".join(lines))
        self.canvas.draw_idle()
