"""Macro-control for the AFR(kPa) fill curve."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..afr_target import AfrFill
from .widgets import ParamRow


class AfrFillControl(ttk.LabelFrame):
    """Edits :class:`AfrFill`; ``fill`` always holds the last valid value."""

    def __init__(self, parent: tk.Misc, fill: AfrFill = AfrFill(), *, kpa_max: int = 250,
                 on_change: Callable[[], None] = lambda: None) -> None:
        super().__init__(parent, text="AFR вдоль нагрузки", padding=8)
        self._on_change = on_change
        self.fill = fill
        self._rows = {
            "lean": ParamRow(self, 0, "AFR lean (ХХ/круиз)", fill.lean, 10.0, 18.0, 0.1, 1, self._recompute),
            "kpa_lean": ParamRow(self, 1, "lean до, kPa", fill.kpa_lean, 0, kpa_max, 1, 0, self._recompute),
            "rich": ParamRow(self, 2, "AFR rich (нагрузка)", fill.rich, 9.0, 16.0, 0.1, 1, self._recompute),
            "kpa_rich": ParamRow(self, 3, "rich от, kPa", fill.kpa_rich, 0, kpa_max, 1, 0, self._recompute),
            "bias": ParamRow(self, 4, "сдвиг перехода", fill.bias, 0.3, 3.0, 0.05, 2, self._recompute),
        }
        ttk.Label(self, foreground="#666666", wraplength=320,
                  text="сдвиг >1: обогащение позже, <1: раньше").grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._error = ttk.Label(self, foreground="#b00000", wraplength=320)
        self._error.grid(row=6, column=0, columnspan=3, sticky="w")

    def _recompute(self) -> None:
        parsed: dict[str, float] = {}
        for name, row in self._rows.items():
            try:
                parsed[name] = row.get()
                row.mark_error(False)
            except ValueError:
                row.mark_error(True)
        if len(parsed) != len(self._rows):
            self._error.configure(text="Введите число")
            return
        try:
            fill = AfrFill(step=self.fill.step, **parsed)
        except ValueError as e:  # AfrFillError
            self._error.configure(text=str(e))
            return
        self._error.configure(text="")
        if fill != self.fill:
            self.fill = fill
            self._on_change()
