"""Macro-control for the AFR levels; the transition ranges come from the sliders by the table."""
from __future__ import annotations

import dataclasses
import tkinter as tk
from tkinter import ttk
from typing import Callable

from ..afr_target import AfrFill
from .widgets import ParamRow


class AfrFillControl(ttk.LabelFrame):
    """Edits ``lean``/``rich`` of :class:`AfrFill`; ranges are set by :meth:`set_ranges`.

    ``fill`` always holds the last valid value.
    """

    def __init__(self, parent: tk.Misc, fill: AfrFill = AfrFill(), *,
                 on_change: Callable[[], None] = lambda: None) -> None:
        super().__init__(parent, text="AFR", padding=8)
        self._on_change = on_change
        self.fill = fill
        self._ranges = dict(kpa_lean=fill.kpa_lean, kpa_rich=fill.kpa_rich,
                            rpm_lean=fill.rpm_lean, rpm_rich=fill.rpm_rich)
        self._rows = {
            "lean": ParamRow(self, 0, "AFR lean (ХХ/круиз)", fill.lean, 10.0, 18.0, 0.1, 1, self._recompute),
            "rich": ParamRow(self, 1, "AFR rich (нагрузка)", fill.rich, 9.0, 16.0, 0.1, 1, self._recompute),
        }
        self._error = ttk.Label(self, foreground="#b00000", wraplength=320)
        self._error.grid(row=2, column=0, columnspan=3, sticky="w")

    def set_ranges(self, *, kpa_lean: float, kpa_rich: float,
                   rpm_lean: float | None, rpm_rich: float | None) -> None:
        self._ranges = dict(kpa_lean=kpa_lean, kpa_rich=kpa_rich, rpm_lean=rpm_lean, rpm_rich=rpm_rich)
        self._recompute()

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
            fill = dataclasses.replace(self.fill, **parsed, **self._ranges)
        except ValueError as e:  # AfrFillError
            self._error.configure(text=str(e))
            return
        self._error.configure(text="")
        if fill != self.fill:
            self.fill = fill
            self._on_change()
