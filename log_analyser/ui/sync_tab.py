"""Tab showing in which cells of the RPM x kPa grid sync losses occurred."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np

from ..corr_table import corr_grid
from ..datalog import SYNC_LOSS_COLUMN, SYNC_LOSS_SOURCE, LogError
from .session import Session
from .table_grid import TableGrid

DEBOUNCE_MS = 150
DEFAULT_OVERLAP = 0.5     # windows tile the plane: every loss is counted exactly once

AXES = (("ve", "оси VE из тюна"), ("target", "оси таблицы AFR target"))


class SyncTab(ttk.Frame):
    """Per-cell map of the ``SyncLoss+`` increments (see :meth:`Log.load`)."""

    def __init__(self, parent: tk.Misc, session: Session,
                 generated_target: Callable[[], object]) -> None:
        super().__init__(parent, padding=8)
        self.session = session
        self._generated = generated_target
        self._pending: str | None = None
        self.grid_result = None

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        box = ttk.LabelFrame(left, text="Оси", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        self.axes_src = self._var("ve")
        for key, text in AXES:
            ttk.Radiobutton(box, text=text, value=key, variable=self.axes_src).pack(anchor="w")
        box = ttk.LabelFrame(left, text="Радиус захвата", padding=8)
        box.pack(fill=tk.X)
        self.overlap = self._var(f"{DEFAULT_OVERLAP:.1f}")
        ttk.Spinbox(box, from_=0.1, to=4.0, increment=0.1, width=6, format="%.1f",
                    textvariable=self.overlap).pack(anchor="w")
        ttk.Label(box, foreground="#666666", wraplength=280,
                  text="в шагах сетки: 0.5 — ячейки без перекрытия, каждая потеря "
                       "учитывается один раз; точки за пределами осей попадают в крайние ячейки"
                       ).pack(anchor="w", pady=(4, 0))

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        self.view = TableGrid(right, cmap="Reds", on_click=self._cell_clicked)
        self.view.pack(fill=tk.BOTH, expand=True)
        self.cell_info = ttk.Label(right, font=("TkDefaultFont", 10))
        self.cell_info.pack(fill=tk.X, pady=(4, 0))
        self.status = ttk.Label(right, foreground="#555555", wraplength=800, justify=tk.LEFT)
        self.status.pack(fill=tk.X, pady=(4, 0))

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        session.subscribe(self.schedule)

    def _var(self, value, kind=tk.StringVar):
        v = kind(value=value)
        v.trace_add("write", lambda *_: self.schedule())
        return v

    # ------------------------------------------------------------------ compute

    def schedule(self) -> None:
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(DEBOUNCE_MS, self.refresh)

    def _axes(self) -> tuple:
        if self.axes_src.get() == "ve":
            ve = self.session.tables.get("veTable")
            if ve is None:
                raise LogError("в тюне нет veTable — загрузите тюн на вкладке «Данные» "
                               "или выберите «оси таблицы AFR target»")
            return ve.x, ve.y
        gen = self._generated()
        if gen is None:
            raise LogError("нет сгенерированной таблицы AFR target — откройте вкладку «AFR Target»")
        return gen.x, gen.y

    def refresh(self) -> None:
        self._pending = None
        self.grid_result = None
        try:
            self._compute()
        except (LogError, ValueError) as e:
            self.view.clear(str(e))
            self.cell_info.configure(text="")
            self.status.configure(text="")
            return
        self._render()
        sel = self.view.selected
        if sel is not None:
            self._cell_clicked(*sel)
        else:
            self.cell_info.configure(text="")

    def _compute(self) -> None:
        if not self.session.logs:
            raise LogError("Загрузите логи на вкладке «Данные».")
        log, notes = self.session.combined()
        if SYNC_LOSS_COLUMN not in log.columns:
            raise LogError(f"в логах нет колонки «{SYNC_LOSS_SOURCE}» "
                           f"(или она есть не во всех файлах)")
        try:
            overlap = float(self.overlap.get())
        except ValueError:
            raise LogError(f"радиус захвата: не число ({self.overlap.get()!r})") from None
        x_axis, y_axis = self._axes()
        g = corr_grid(log, x_axis, y_axis, column=SYNC_LOSS_COLUMN, stat="sum",
                      min_points=1, overlap=overlap, clamp=True)
        self.grid_result = g
        losses = int(g.values[np.isfinite(g.values)].sum())
        cells = int((g.values > 0).sum())
        lines = [f"Потерь {losses} · в {cells} ячейках из {int(g.valid.sum())} с данными · "
                 f"строк {int(g.counts.sum())}"]
        if notes:
            lines.append("Объединение логов: " + "; ".join(notes))
        self.status.configure(text="\n".join(lines))

    def _render(self) -> None:
        g = self.grid_result
        losses = g.values
        texts = [[str(int(v)) if np.isfinite(v) else "" for v in row] for row in losses]
        self.view.show_grid(g.x, g.y, losses, texts, cmap="Reds", vmin=0,
                            vmax=float(max(1, np.nanmax(losses))),
                            title=f"Потери синхронизации по ячейкам; серые — нет точек, "
                                  f"0 — точки есть, потерь нет")

    def _cell_clicked(self, i: int, j: int) -> None:
        g = self.grid_result
        if g is None:
            return
        text = f"RPM {g.x[j]}, kPa {g.y[i]}: точек {g.counts[i, j]}"
        v = g.values[i, j]
        if np.isfinite(v):
            text += f", потерь {int(v)}"
        else:
            text += " (нет точек)"
        self.cell_info.configure(text=text)
