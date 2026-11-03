"""Colour-coded view of a :class:`Table` on a tk Canvas (TunerStudio layout)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex

from ..table import Table


class TableGrid(ttk.Frame):
    """Rows = y axis with the highest value on top, columns = x axis, x labels at the bottom.

    Colours come from ``cmap`` over ``[vmin, vmax]`` (fixed, so colours stay comparable
    while the table is edited); when not given, the table's own range is used.
    """

    def __init__(self, parent: tk.Misc, *, cmap: str = "RdYlBu",
                 vmin: float | None = None, vmax: float | None = None) -> None:
        super().__init__(parent)
        self._cmap = colormaps[cmap]
        self._vmin, self._vmax = vmin, vmax
        self._table: Table | None = None
        self.canvas = tk.Canvas(self, background="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw())

    @property
    def table(self) -> Table | None:
        return self._table

    def show(self, table: Table) -> None:
        self._table = table
        self._draw()

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        t = self._table
        w, h = c.winfo_width(), c.winfo_height()
        if t is None or w < 50 or h < 50:
            return

        rows, cols = t.shape
        label_w, label_h, title_h = 44, 20, 20
        cw = (w - label_w - 2) / cols
        ch = (h - label_h - title_h - 2) / rows
        size = max(6, min(11, int(min(cw / 3.6, ch / 1.6))))
        font = ("TkDefaultFont", size)
        digits = t.kind.digits

        vals = t.values
        lo = float(vals.min()) if self._vmin is None else self._vmin
        hi = float(vals.max()) if self._vmax is None else self._vmax
        norm = Normalize(lo, hi if hi > lo else lo + 1e-9, clip=True)

        c.create_text(label_w, title_h / 2, anchor="w", font=("TkDefaultFont", 10, "bold"),
                      text=f"{t.caption or t.kind.label} [{t.kind.unit}]   "
                           f"строки: {t.y.unit}, столбцы: {t.x.unit}   "
                           f"{vals.min():.{digits}f} … {vals.max():.{digits}f}")
        for r in range(rows):
            i = rows - 1 - r                      # highest y on top
            y0 = title_h + r * ch
            c.create_text(label_w - 4, y0 + ch / 2, anchor="e", font=font, text=str(t.y[i]))
            for j in range(cols):
                x0 = label_w + j * cw
                v = float(vals[i, j])
                rgba = self._cmap(norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                c.create_rectangle(x0, y0, x0 + cw, y0 + ch, fill=to_hex(rgba), outline="#ffffff")
                c.create_text(x0 + cw / 2, y0 + ch / 2, font=font,
                              fill="black" if lum > 0.5 else "white", text=f"{v:.{digits}f}")
        y_lab = title_h + rows * ch + label_h / 2
        for j in range(cols):
            c.create_text(label_w + (j + 0.5) * cw, y_lab, font=font, text=str(t.x[j]))
