"""Small reusable widgets: a labelled numeric parameter with a slider and a spinbox."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

ERROR_STYLE = "Error.TSpinbox"


def init_styles(root: tk.Misc) -> None:
    style = ttk.Style(root)
    style.configure(ERROR_STYLE, fieldbackground="#ffd6d6", foreground="#a00000")


class ParamRow:
    """``label  [====slider====]  [spinbox]`` on one grid row of ``parent``.

    Slider and spinbox are kept in sync; the slider snaps to ``inc``. ``on_change``
    is called after every edit (the value may be unparsable: callers use :meth:`get`
    inside ``try`` and call :meth:`mark_error`).
    """

    def __init__(self, parent: tk.Misc, row: int, label: str, value: float,
                 lo: float, hi: float, inc: float, digits: int,
                 on_change: Callable[[], None]) -> None:
        self._inc = inc
        self._digits = digits
        self._on_change = on_change
        self.var = tk.StringVar(value=self._fmt(value))

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=1)
        self.scale = ttk.Scale(parent, from_=lo, to=hi, orient=tk.HORIZONTAL,
                               command=self._on_scale)
        self.scale.grid(row=row, column=1, sticky="ew", pady=1)
        self.spin = ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, width=7,
                                textvariable=self.var, format=f"%.{digits}f")
        self.spin.grid(row=row, column=2, sticky="e", padx=(6, 0), pady=1)
        parent.columnconfigure(1, weight=1)

        self.scale.set(value)
        self.var.trace_add("write", self._on_var)

    def _fmt(self, v: float) -> str:
        return f"{v:.{self._digits}f}"

    def _on_scale(self, raw: str) -> None:
        v = round(float(raw) / self._inc) * self._inc
        text = self._fmt(v)
        if text != self.var.get():
            self.var.set(text)  # triggers _on_var -> on_change

    def _on_var(self, *_args) -> None:
        try:
            v = float(self.var.get())
        except ValueError:
            pass
        else:
            if abs(float(self.scale.get()) - v) > self._inc / 2:
                self.scale.set(v)  # may re-enter _on_scale; text is equal, so no loop
        self._on_change()

    def get(self) -> float:
        """Current value; raises ``ValueError`` if the text is not a number."""
        try:
            return float(self.var.get())
        except ValueError:
            raise ValueError(f"not a number: {self.var.get()!r}") from None

    def set(self, value: float) -> None:
        self.var.set(self._fmt(value))

    def mark_error(self, bad: bool) -> None:
        self.spin.configure(style=ERROR_STYLE if bad else "TSpinbox")
