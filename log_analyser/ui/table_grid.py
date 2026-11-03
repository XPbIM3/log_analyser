"""Colour-coded view of a 2-D grid on a tk Canvas (TunerStudio layout)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_hex

from ..axis import Axis
from ..table import Table

MISSING_FILL = "#e6e6e6"
LABEL_W = 44      # y labels on the left
LABEL_H = 20      # x labels at the bottom
TITLE_H = 20      # title on top


class TableGrid(ttk.Frame):
    """Rows = y axis with the highest value on top, columns = x axis, x labels at the bottom.

    :meth:`show` displays a :class:`Table`; :meth:`show_grid` displays arbitrary texts
    over colour values (NaN colour -> grey cell). Colours come from ``cmap`` over a
    fixed ``[vmin, vmax]`` when given (stable while editing), else the data range.
    ``on_click(i, j)`` gets the row/column index in axis order (row 0 = ``y[0]``).
    """

    def __init__(self, parent: tk.Misc, *, cmap: str = "RdYlBu",
                 vmin: float | None = None, vmax: float | None = None,
                 on_click: Callable[[int, int], None] | None = None) -> None:
        super().__init__(parent)
        self._cmap_name, self._vmin, self._vmax = cmap, vmin, vmax
        self._on_click = on_click
        self._data: tuple | None = None
        self._geom: tuple[float, float, float, float, int, int] | None = None
        self.selected: tuple[int, int] | None = None
        self.canvas = tk.Canvas(self, background="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw())
        self.canvas.bind("<Button-1>", self._click)

    def show(self, table: Table) -> None:
        d = table.kind.digits
        texts = [[f"{v:.{d}f}" for v in row] for row in table.values]
        title = (f"{table.caption or table.kind.label} [{table.kind.unit}]   "
                 f"строки: {table.y.unit}, столбцы: {table.x.unit}   "
                 f"{table.values.min():.{d}f} … {table.values.max():.{d}f}")
        self.show_grid(table.x, table.y, table.values, texts, title=title)

    def show_grid(self, x: Axis, y: Axis, colors: np.ndarray, texts: Sequence[Sequence[str]], *,
                  title: str = "", cmap: str | None = None,
                  vmin: float | None = None, vmax: float | None = None) -> None:
        if self._data is not None and (self._data[0] != x or self._data[1] != y):
            self.selected = None
        self._data = (x, y, np.asarray(colors, dtype=float), texts, title,
                      cmap or self._cmap_name,
                      self._vmin if vmin is None else vmin, self._vmax if vmax is None else vmax)
        self._draw()

    def clear(self, message: str = "") -> None:
        self._data = None
        self._geom = None
        self.canvas.delete("all")
        if message:
            self.canvas.create_text(20, 20, anchor="nw", text=message, fill="#666666",
                                    font=("TkDefaultFont", 11), width=max(200, self.canvas.winfo_width() - 40))

    def _draw(self) -> None:
        c = self.canvas
        if self._data is None:
            return
        c.delete("all")
        x, y, colors, texts, title, cmap_name, vmin, vmax = self._data
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or h < 50:
            return

        rows, cols = len(y), len(x)
        label_w, label_h, title_h = LABEL_W, LABEL_H, TITLE_H
        cw = (w - label_w - 2) / cols
        ch = (h - label_h - title_h - 2) / rows
        self._geom = (label_w, title_h, cw, ch, rows, cols)
        size = max(6, min(11, int(min(cw / 3.6, ch / 1.6))))
        font = ("TkDefaultFont", size)

        finite = colors[np.isfinite(colors)]
        lo = vmin if vmin is not None else (float(finite.min()) if finite.size else 0.0)
        hi = vmax if vmax is not None else (float(finite.max()) if finite.size else 1.0)
        norm = Normalize(lo, hi if hi > lo else lo + 1e-9, clip=True)
        cmap = colormaps[cmap_name]

        c.create_text(label_w, title_h / 2, anchor="w", font=("TkDefaultFont", 10, "bold"), text=title,
                      fill="black")
        for r in range(rows):
            i = rows - 1 - r                      # highest y on top
            y0 = title_h + r * ch
            c.create_text(label_w - 4, y0 + ch / 2, anchor="e", font=font, text=str(y[i]),
                          fill="black")
            for j in range(cols):
                x0 = label_w + j * cw
                v = float(colors[i, j])
                if np.isfinite(v):
                    rgba = cmap(norm(v))
                    fill = to_hex(rgba)
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    fg = "black" if lum > 0.5 else "white"
                else:
                    fill, fg = MISSING_FILL, "#888888"
                c.create_rectangle(x0, y0, x0 + cw, y0 + ch, fill=fill, outline="#ffffff")
                c.create_text(x0 + cw / 2, y0 + ch / 2, font=font, fill=fg, text=texts[i][j])
        y_lab = title_h + rows * ch + label_h / 2
        for j in range(cols):
            c.create_text(label_w + (j + 0.5) * cw, y_lab, font=font, text=str(x[j]), fill="black")
        if self.selected is not None:
            i, j = self.selected
            x0, y0 = label_w + j * cw, title_h + (rows - 1 - i) * ch
            c.create_rectangle(x0 + 1, y0 + 1, x0 + cw - 1, y0 + ch - 1, outline="black", width=3)

    def _click(self, event: tk.Event) -> None:
        if self._geom is None or self._on_click is None:
            return
        label_w, title_h, cw, ch, rows, cols = self._geom
        j = int((event.x - label_w) // cw)
        r = int((event.y - title_h) // ch)
        if 0 <= j < cols and 0 <= r < rows:
            self.selected = (rows - 1 - r, j)
            self._draw()
            self._on_click(*self.selected)
