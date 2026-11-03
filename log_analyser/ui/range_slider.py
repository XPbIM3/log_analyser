"""Two-handle slider aligned with the rows or columns of a :class:`TableGrid`."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..axis import Axis

HANDLE = 7            # handle half-size, px
MARGIN = HANDLE + 1   # track starts/ends this far from the canvas edge
BAND = "#f0a040"      # transition zone
TRACK = "#bbbbbb"
OUTSIDE = "#dddddd"   # track beyond the table
LEAN, RICH = "#3070c0", "#c03030"


class AxisRangeSlider(ttk.Frame):
    """Range ``lo < hi`` in axis units, from 0 to ``limit_factor`` × the axis maximum.

    The track has three zones along its length (counted from the low-value end:
    left for horizontal, bottom for vertical):

    * ``pad_lo`` px before the table cells — values from 0 up to the edge of the
      first cell, linearly;
    * the table cells — node ``k`` of the axis sits at the centre of cell ``k``,
      between nodes the value is interpolated by index (follows non-linear axes),
      half a cell beyond the end nodes the end step is continued;
    * ``pad_hi`` px after the cells — from the edge of the last cell up to the limit.

    ``pad_lo``/``pad_hi`` must equal the canvas space next to the table's cells so
    the track lines up with them. Values snap to ``quantum``.
    """

    def __init__(self, parent: tk.Misc, axis: Axis, lo: float, hi: float, *, orient: str,
                 pad_lo: int, pad_hi: int, quantum: int = 1, thickness: int = 46,
                 limit_factor: float = 2.0, on_change: Callable[[], None] = lambda: None) -> None:
        super().__init__(parent)
        self._q = quantum
        self._factor = limit_factor
        self.axis, self.lo, self.hi = axis, self._snap(lo), self._snap(hi)
        self.vertical = orient == "vertical"
        self._pads = (pad_lo, pad_hi)
        self._on_change = on_change
        self._drag: str | None = None
        self.enabled = True
        size = dict(width=thickness) if self.vertical else dict(height=thickness)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="white", **size)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw())
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._move)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))

    # ------------------------------------------------------------------ public

    @property
    def limits(self) -> tuple[float, float]:
        return 0.0, self._snap(self._factor * float(self.axis.values[-1]))

    def set_axis(self, axis: Axis) -> None:
        """New axis; values are kept (drawn clamped at the track ends if outside)."""
        if axis != self.axis:
            self.axis = axis
            self._draw()

    def set(self, lo: float, hi: float, notify: bool = True) -> None:
        vmin, vmax = self.limits
        lo = min(max(self._snap(lo), vmin), vmax - self._q)
        hi = min(max(self._snap(hi), lo + self._q), vmax)
        if (lo, hi) != (self.lo, self.hi):
            self.lo, self.hi = lo, hi
            self._draw()
            if notify:
                self._on_change()

    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self._draw()

    # ------------------------------------------------------------------ geometry
    # ``u`` is the track coordinate in px from the low-value end of the canvas.

    def _snap(self, v: float) -> float:
        return float(round(v / self._q) * self._q)

    def _length(self) -> float:
        c = self.canvas
        return c.winfo_height() if self.vertical else c.winfo_width()

    def _zones(self) -> tuple[float, float, float, float, float]:
        """(u of track start, u of first cell edge, u of last cell edge, u of track end, cell size)."""
        length = self._length()
        pad_lo, pad_hi = self._pads
        cells_lo, cells_hi = pad_lo, length - pad_hi
        n = len(self.axis)
        return MARGIN, cells_lo, cells_hi, length - MARGIN, max(1.0, (cells_hi - cells_lo) / n)

    def _u(self, value: float) -> float:
        start, c_lo, c_hi, end, cell = self._zones()
        n = len(self.axis)
        vmin, vmax = self.limits
        e_lo, e_hi = self.axis.at(-0.5), self.axis.at(n - 0.5)
        value = min(max(value, vmin), vmax)
        if value < e_lo:
            return start + (c_lo - start) * (value - vmin) / max(e_lo - vmin, 1e-9)
        if value > e_hi:
            return c_hi + (end - c_hi) * (value - e_hi) / max(vmax - e_hi, 1e-9)
        return c_lo + (self.axis.position(value) + 0.5) * cell

    def _value_at(self, u: float) -> float:
        start, c_lo, c_hi, end, cell = self._zones()
        n = len(self.axis)
        vmin, vmax = self.limits
        e_lo, e_hi = self.axis.at(-0.5), self.axis.at(n - 0.5)
        if u <= c_lo:
            v = vmin + (e_lo - vmin) * (u - start) / max(c_lo - start, 1e-9)
        elif u >= c_hi:
            v = e_hi + (vmax - e_hi) * (u - c_hi) / max(end - c_hi, 1e-9)
        else:
            v = self.axis.at((u - c_lo) / cell - 0.5)
        return min(max(self._snap(v), vmin), vmax)

    def _pix(self, value: float) -> float:
        u = self._u(value)
        return self._length() - u if self.vertical else u

    def _value(self, pix: float) -> float:
        return self._value_at(self._length() - pix if self.vertical else pix)

    # ------------------------------------------------------------------ drawing / input

    def _draw(self) -> None:
        c = self.canvas
        c.delete("all")
        if self._length() < 30:
            return
        start, c_lo, c_hi, end, _cell = self._zones()
        across = c.winfo_width() if self.vertical else c.winfo_height()
        mid = across / 2 if self.vertical else across * 0.35
        flip = (lambda u: self._length() - u) if self.vertical else (lambda u: u)

        def seg(a, b, **kw):
            a, b = flip(a), flip(b)
            if self.vertical:
                c.create_line(mid, a, mid, b, **kw)
            else:
                c.create_line(a, mid, b, mid, **kw)

        def tick(u, half, **kw):
            p = flip(u)
            if self.vertical:
                c.create_line(mid - half, p, mid + half, p, **kw)
            else:
                c.create_line(p, mid - half, p, mid + half, **kw)

        seg(start, c_lo, fill=OUTSIDE, width=2, dash=(3, 2))
        seg(c_lo, c_hi, fill=TRACK, width=2)
        seg(c_hi, end, fill=OUTSIDE, width=2, dash=(3, 2))
        for u in (c_lo, c_hi):                       # table edges
            tick(u, 6, fill=TRACK, width=2)
        for v in self.axis.values:                   # cell centres
            tick(self._u(float(v)), 3, fill=TRACK)
        if not self.enabled:
            return
        u_lo, u_hi = self._u(self.lo), self._u(self.hi)
        seg(u_lo, u_hi, fill=BAND, width=6)
        for value, u, colour, label_side in ((self.lo, u_lo, LEAN, -1), (self.hi, u_hi, RICH, 1)):
            p = flip(u)
            if self.vertical:
                c.create_rectangle(mid - HANDLE, p - HANDLE / 2, mid + HANDLE, p + HANDLE / 2,
                                   fill=colour, outline="black")
                c.create_text(mid, p - label_side * (HANDLE + 6), text=f"{value:g}",
                              font=("TkDefaultFont", 8))
            else:
                c.create_rectangle(p - HANDLE / 2, mid - HANDLE, p + HANDLE / 2, mid + HANDLE,
                                   fill=colour, outline="black")
                c.create_text(p, mid + HANDLE + 8, text=f"{value:g}", font=("TkDefaultFont", 8))

    def _press(self, event: tk.Event) -> None:
        if not self.enabled:
            return
        pos = event.y if self.vertical else event.x
        d_lo, d_hi = abs(pos - self._pix(self.lo)), abs(pos - self._pix(self.hi))
        self._drag = "lo" if d_lo < d_hi or (d_lo == d_hi and self._value(pos) < self.lo) else "hi"
        self._move(event)

    def _move(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        v = self._value(event.y if self.vertical else event.x)
        if self._drag == "lo":
            self.set(min(v, self.hi - self._q), self.hi)
        else:
            self.set(self.lo, max(v, self.lo + self._q))
