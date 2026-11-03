"""Macro-control that builds a table axis from min / max / points / gamma."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..axis import Axis
from .widgets import ParamRow


class AxisControl(ttk.LabelFrame):
    """Edits an axis of class ``axis_cls`` via :meth:`Axis.gamma` and previews it.

    ``axis`` always holds the last valid axis; ``on_change`` is called whenever it changes.
    """

    def __init__(self, parent: tk.Misc, axis_cls: type[Axis], title: str, *,
                 start: int, stop: int, n: int = 16, gamma: float = 1.0,
                 range_max: int, on_change: Callable[[], None] = lambda: None) -> None:
        super().__init__(parent, text=title, padding=8)
        self._cls = axis_cls
        self._on_change = on_change
        q = axis_cls.quantum

        self.axis: Axis = axis_cls.gamma(start, stop, n, gamma)
        self.start = ParamRow(self, 0, "min", start, axis_cls.min_value, range_max, q, 0, self._recompute)
        self.stop = ParamRow(self, 1, "max", stop, axis_cls.min_value, range_max, q, 0, self._recompute)
        self.n = ParamRow(self, 2, "points", n, 2, 32, 1, 0, self._recompute)
        self.gamma = ParamRow(self, 3, "gamma", gamma, 0.3, 3.0, 0.05, 2, self._recompute)

        self._strip = tk.Canvas(self, height=28, highlightthickness=0, background="white")
        self._strip.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self._strip.bind("<Configure>", lambda _e: self._draw_strip())
        self._values = ttk.Label(self, wraplength=320, font="TkFixedFont")
        self._values.grid(row=5, column=0, columnspan=3, sticky="w")
        self._error = ttk.Label(self, foreground="#b00000", wraplength=320)
        self._error.grid(row=6, column=0, columnspan=3, sticky="w")
        self._show_values()

    def _recompute(self) -> None:
        rows = {"start": self.start, "stop": self.stop, "n": self.n, "gamma": self.gamma}
        parsed: dict[str, float] = {}
        for name, row in rows.items():
            try:
                parsed[name] = row.get()
                row.mark_error(False)
            except ValueError:
                row.mark_error(True)
        if len(parsed) != len(rows):
            self._error.configure(text="Введите число")
            return
        try:
            n = parsed["n"]
            if n != int(n):
                raise ValueError("points must be an integer")
            axis = self._cls.gamma(parsed["start"], parsed["stop"], int(n), parsed["gamma"])
        except ValueError as e:  # AxisError is a ValueError
            self._error.configure(text=str(e))
            return
        self._error.configure(text="")
        if axis != self.axis:
            self.axis = axis
            self._show_values()
            self._on_change()

    def _show_values(self) -> None:
        steps = self.axis.steps
        self._values.configure(
            text=" ".join(str(v) for v in self.axis)
            + f"\nшаг: {steps.min()}…{steps.max()} {self._cls.unit}")
        self._draw_strip()

    def _draw_strip(self) -> None:
        c = self._strip
        c.delete("all")
        w, h = c.winfo_width(), int(c["height"])
        if w < 20:
            return
        pad = 6
        vals = self.axis.values
        lo, hi = int(vals[0]), int(vals[-1])
        y = h // 2
        c.create_line(pad, y, w - pad, y, fill="#999999")
        for v in vals:
            x = pad + (w - 2 * pad) * (int(v) - lo) / (hi - lo)
            c.create_line(x, y - 8, x, y + 8, fill="#1f5fa8", width=2)
