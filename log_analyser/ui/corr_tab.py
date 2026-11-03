"""Tab showing the mixture correction table computed from the loaded logs."""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np

from ..corr_table import CORR_STATS, CorrGrid, LogFilter, corr_grid, ve_grid
from ..datalog import LogError
from ..table import Table, TableKind
from ..table_file import export_table, read_table
from .session import Session
from .cell_panel import CellPanel
from .table_grid import TableGrid

DEBOUNCE_MS = 150
CORR_SPAN = 0.15          # colour scale: 1 ± 15 %

# Defaults of the correction tab (filter defaults live in LogFilter)
DEFAULT_TARGET = "generated"
DEFAULT_AXES = "target"
DEFAULT_STAT = "median"
DEFAULT_MIN_POINTS = 10
DEFAULT_OVERLAP = 1.5
DEFAULT_MODE = "ve"
DEFAULT_RPM_DOT_ON = True          # rpm/s >= 0
DEFAULT_TPS_ON = True              # TPS > DEFAULT_TPS_ABOVE
DEFAULT_TPS_ABOVE = 0

TARGETS = (("tune", "из тюна (afrTable)"),
           ("generated", "сгенерированная (вкладка AFR Target)"),
           ("log", "из лога (колонка «AFR Target»)"),
           ("file", "из файла .table"))
AXES = (("ve", "оси VE из тюна"), ("target", "оси таблицы target"))
MODES = (("corr", "коррекция, %"), ("spread", "разброс (IQR)"), ("counts", "точек"), ("ve", "VE новый"))
IQR_MAX = 0.20           # colour scale of the spread view: 0 … 20 п.п.


class CorrTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, session: Session,
                 generated_target: Callable[[], Table]) -> None:
        super().__init__(parent, padding=8)
        self.session = session
        self._generated = generated_target
        self._file_table: Table | None = None
        self._pending: str | None = None
        self.grid_result: CorrGrid | None = None
        self.ve_base: Table | None = None     # tune VE resampled to the grid axes
        self.ve_new: Table | None = None
        self.ve_note = ""

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self._build_target(left)
        self._build_filter(left)
        self._build_aggregation(left)
        self._build_view(left)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        paned = ttk.PanedWindow(right, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)
        self.view = TableGrid(paned, cmap="RdBu", on_click=self._cell_clicked)
        self.cell_panel = CellPanel(paned)
        paned.add(self.view, weight=3)
        paned.add(self.cell_panel, weight=2)
        self.cell_info = ttk.Label(right, font=("TkDefaultFont", 10))
        self.cell_info.pack(fill=tk.X, pady=(4, 0))
        self.status = ttk.Label(right, foreground="#555555", wraplength=800, justify=tk.LEFT)
        self.status.pack(fill=tk.X, pady=(4, 0))

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        session.subscribe(self.schedule)

    # ------------------------------------------------------------------ controls

    def _var(self, value, kind=tk.StringVar):
        v = kind(value=value)
        v.trace_add("write", lambda *_: self.schedule())
        return v

    def _build_target(self, parent: tk.Misc) -> None:
        box = ttk.LabelFrame(parent, text="Target AFR", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        self.target_src = self._var(DEFAULT_TARGET)
        self._target_buttons = {}
        for key, text in TARGETS:
            b = ttk.Radiobutton(box, text=text, value=key, variable=self.target_src)
            b.pack(anchor="w")
            self._target_buttons[key] = b
        row = ttk.Frame(box)
        row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(row, text="Выбрать .table…", command=self.choose_file).pack(side=tk.LEFT)
        self.file_label = ttk.Label(row, text="", foreground="#555555")
        self.file_label.pack(side=tk.LEFT, padx=6)

        ttk.Separator(box).pack(fill=tk.X, pady=6)
        self.axes_src = self._var(DEFAULT_AXES)
        for key, text in AXES:
            ttk.Radiobutton(box, text=text, value=key, variable=self.axes_src).pack(anchor="w")

    def _build_filter(self, parent: tk.Misc) -> None:
        box = ttk.LabelFrame(parent, text="Фильтр установившегося режима", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        d = LogFilter()
        self.f_clt_on = self._var(True, tk.BooleanVar)
        self.f_clt = self._var(f"{d.clt_min:g}")
        self.f_dfco = self._var(d.exclude_dfco, tk.BooleanVar)
        self.f_warm = self._var(d.exclude_warmup, tk.BooleanVar)
        self.f_accel = self._var(d.exclude_accel, tk.BooleanVar)
        self.f_tail = self._var(str(d.accel_tail))
        self.f_afr_on = self._var(True, tk.BooleanVar)
        self.f_afr_lo = self._var(f"{d.afr_min:g}")
        self.f_afr_hi = self._var(f"{d.afr_max:g}")
        self.f_rpmdot_on = self._var(DEFAULT_RPM_DOT_ON, tk.BooleanVar)
        self.f_tps_on = self._var(DEFAULT_TPS_ON, tk.BooleanVar)
        self.f_tps = self._var(f"{DEFAULT_TPS_ABOVE:g}")

        r = 0
        ttk.Checkbutton(box, text="CLT ≥, °C", variable=self.f_clt_on).grid(row=r, column=0, sticky="w")
        ttk.Spinbox(box, from_=0, to=120, increment=5, width=6, textvariable=self.f_clt).grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Checkbutton(box, text="без DFCO", variable=self.f_dfco).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        ttk.Checkbutton(box, text="без прогревного обогащения", variable=self.f_warm).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        ttk.Checkbutton(box, text="без ускор. обогащ., +строк", variable=self.f_accel).grid(row=r, column=0, sticky="w")
        ttk.Spinbox(box, from_=0, to=100, increment=1, width=6, textvariable=self.f_tail).grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Checkbutton(box, text="AFR в диапазоне", variable=self.f_afr_on).grid(row=r, column=0, sticky="w")
        rng = ttk.Frame(box)
        rng.grid(row=r, column=1, columnspan=2, sticky="w")
        ttk.Spinbox(rng, from_=5, to=25, increment=0.5, width=5, textvariable=self.f_afr_lo).pack(side=tk.LEFT)
        ttk.Label(rng, text="…").pack(side=tk.LEFT)
        ttk.Spinbox(rng, from_=5, to=25, increment=0.5, width=5, textvariable=self.f_afr_hi).pack(side=tk.LEFT)
        r += 1
        ttk.Checkbutton(box, text="rpm/s ≥ 0 (без сброса оборотов)",
                        variable=self.f_rpmdot_on).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        ttk.Checkbutton(box, text="TPS >, %", variable=self.f_tps_on).grid(row=r, column=0, sticky="w")
        ttk.Spinbox(box, from_=0, to=100, increment=1, width=6, textvariable=self.f_tps).grid(row=r, column=1, sticky="w")

    def _build_aggregation(self, parent: tk.Misc) -> None:
        box = ttk.LabelFrame(parent, text="Агрегация по ячейкам", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        self.stat = self._var(DEFAULT_STAT)
        self.min_points = self._var(str(DEFAULT_MIN_POINTS))
        self.overlap = self._var(f"{DEFAULT_OVERLAP:.1f}")
        ttk.Label(box, text="статистика").grid(row=0, column=0, sticky="w")
        ttk.Combobox(box, values=CORR_STATS, state="readonly", width=8, textvariable=self.stat).grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="мин. точек").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(box, from_=1, to=1000, increment=5, width=6, textvariable=self.min_points).grid(row=1, column=1, sticky="w")
        ttk.Label(box, text="радиус захвата").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(box, from_=0.1, to=4.0, increment=0.1, width=6, format="%.1f",
                    textvariable=self.overlap).grid(row=2, column=1, sticky="w")
        ttk.Label(box, foreground="#666666", wraplength=300,
                  text="в шагах сетки: 0.5 — до середины к соседу, 1 — до соседа, 2 — до второго соседа").grid(row=3, column=0, columnspan=2, sticky="w")

    def _build_view(self, parent: tk.Misc) -> None:
        box = ttk.LabelFrame(parent, text="Показать", padding=8)
        box.pack(fill=tk.X)
        self.mode = self._var(DEFAULT_MODE)
        for key, text in MODES:
            ttk.Radiobutton(box, text=text, value=key, variable=self.mode).pack(anchor="w")
        self.export_btn = ttk.Button(box, text="Экспорт VE новый .table…", command=self.export_ve)
        self.export_btn.pack(anchor="w", pady=(6, 0))

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Target AFR .table",
                                          filetypes=[("TunerStudio table", "*.table"), ("Все", "*")])
        if not path:
            return
        try:
            self._file_table = read_table(path, TableKind.AFR, caption=os.path.basename(path))
        except ValueError as e:
            messagebox.showerror("Target AFR", str(e), parent=self)
            return
        self.file_label.configure(text=os.path.basename(path))
        self.target_src.set("file")
        self.schedule()

    # ------------------------------------------------------------------ compute

    def schedule(self) -> None:
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(DEBOUNCE_MS, self.refresh)

    def _filter(self) -> LogFilter:
        def num(var: tk.StringVar, what: str) -> float:
            try:
                return float(var.get())
            except ValueError:
                raise LogError(f"{what}: не число ({var.get()!r})") from None

        on_afr = self.f_afr_on.get()
        return LogFilter(
            clt_min=num(self.f_clt, "CLT") if self.f_clt_on.get() else None,
            exclude_dfco=self.f_dfco.get(),
            exclude_warmup=self.f_warm.get(),
            exclude_accel=self.f_accel.get(),
            accel_tail=int(num(self.f_tail, "строк после ускорения")),
            afr_min=num(self.f_afr_lo, "AFR от") if on_afr else None,
            afr_max=num(self.f_afr_hi, "AFR до") if on_afr else None,
            rpm_dot_min=0.0 if self.f_rpmdot_on.get() else None,
            tps_above=num(self.f_tps, "TPS") if self.f_tps_on.get() else None,
        )

    def _target(self) -> tuple[Table | str, Table | None]:
        """(target for Log.with_corr, table whose axes can be used for the grid)."""
        src = self.target_src.get()
        if src == "tune":
            t = self.session.tables.get("afrTable")
            if t is None:
                raise LogError("в тюне нет afrTable (загрузите тюн на вкладке «Данные»)")
            return t, t
        if src == "file":
            if self._file_table is None:
                raise LogError("файл target не выбран")
            return self._file_table, self._file_table
        gen = self._generated()
        if src == "log":
            return "AFR Target", gen
        return gen, gen

    def refresh(self) -> None:
        self._pending = None
        has_afr = "afrTable" in self.session.tables
        self._target_buttons["tune"].state(["!disabled"] if has_afr else ["disabled"])
        self.grid_result = self.ve_base = self.ve_new = self.corr_log = None
        self.ve_note = ""
        try:
            self._compute()
        except (LogError, ValueError) as e:
            self.view.clear(str(e))
            self.cell_panel.clear()
            self.cell_info.configure(text="")
            self.status.configure(text="", foreground="#555555")
            self._update_export()
            return
        self._render()
        self._update_export()
        sel = self.view.selected
        if sel is not None:
            self._cell_clicked(*sel)     # same axes: refresh the panel with the new data
        else:
            self.cell_panel.clear()
            self.cell_info.configure(text="")

    def _compute(self) -> None:
        if not self.session.logs:
            raise LogError("Загрузите логи на вкладке «Данные».")
        log, notes = self.session.combined()
        flt = self._filter()
        filtered = flt.apply(log)
        target, axes_table = self._target()

        ve = self.session.tables.get("veTable")
        if self.axes_src.get() == "ve":
            if ve is None:
                raise LogError("в тюне нет veTable — выберите «оси таблицы target» или загрузите тюн")
            x_axis, y_axis = ve.x, ve.y
        else:
            x_axis, y_axis = axes_table.x, axes_table.y

        try:
            min_points = int(self.min_points.get())
            overlap = float(self.overlap.get())
        except ValueError:
            raise LogError("мин. точек / радиус захвата: не число") from None
        if not len(filtered):
            raise LogError("после фильтра не осталось строк:\n" + "\n".join(flt.report(log)))
        corr = filtered.with_corr(target)
        self.corr_log = corr
        g = corr_grid(corr, x_axis, y_axis, stat=self.stat.get(),
                      min_points=min_points, overlap=overlap)
        self.grid_result = g
        try:
            vr = ve_grid(corr, x_axis, y_axis, base=ve, stat=self.stat.get(),
                         min_points=min_points, overlap=overlap)
        except LogError as e:
            self.ve_note = f"VE новый не посчитан: {e}"
        else:
            self.ve_new, self.ve_base = vr.table, vr.base
            self.ve_note = (f"VE новый = {g.stat}(«{vr.ve_column}» × CORR) по ячейке; без данных — "
                            + ("VE из тюна в узлах новых осей." if ve is not None else "0 (тюн не загружен)."))

        valid = g.valid
        lines = [f"Строк {len(log)} → после фильтра {len(filtered)} "
                 f"({100 * len(filtered) / len(log):.0f} %) · ячеек с данными {int(valid.sum())} из {valid.size}"]
        if self.ve_new is None and self.ve_note:
            lines.append(self.ve_note)
        if notes:
            lines.append("Объединение логов: " + "; ".join(notes))
        self.status.configure(text="\n".join(lines))

    def _render(self) -> None:
        g = self.grid_result
        mode = self.mode.get()
        valid = g.valid
        corr_colors = np.where(valid, g.values, np.nan)
        if mode == "counts":
            texts = [[str(c) if c else "" for c in row] for row in g.counts]
            colors = np.where(g.counts > 0, g.counts.astype(float), np.nan)
            self.view.show_grid(g.x, g.y, colors, texts, cmap="Greens", vmin=0,
                                vmax=float(max(1, g.counts.max())),
                                title=f"Точек в ячейке (мин. для расчёта {g.min_points})")
            return
        if mode == "spread":
            iqr = g.iqr
            texts = [[f"{100 * v:.1f}" if np.isfinite(v) else "" for v in row] for row in iqr]
            self.view.show_grid(g.x, g.y, iqr, texts, cmap="Oranges", vmin=0, vmax=IQR_MAX,
                                title="Разброс коррекции в ячейке: IQR (75 % − 25 %), п.п.; "
                                      "тёмные — подозрительные, смотрите по клику")
            return
        if mode == "ve":
            if self.ve_new is None:
                self.view.clear(self.ve_note or "VE новый не посчитан.")
                return
            new = self.ve_new.values
            texts = [[f"{new[i, j]:.0f}" for j in range(new.shape[1])] for i in range(new.shape[0])]
            rest = "из тюна" if self.ve_base is not None else "0, тюн не загружен"
            self.view.show_grid(g.x, g.y, corr_colors, texts, cmap="RdBu",
                                vmin=1 - CORR_SPAN, vmax=1 + CORR_SPAN,
                                title=f"VE новый: посчитано {int(valid.sum())} из {valid.size} ячеек "
                                      f"(цвет — коррекция); серые — {rest}")
            return
        texts = [[f"{(v - 1) * 100:+.1f}" if np.isfinite(v) else "" for v in row] for row in g.values]
        self.view.show_grid(g.x, g.y, corr_colors, texts, cmap="RdBu",
                            vmin=1 - CORR_SPAN, vmax=1 + CORR_SPAN,
                            title=f"Коррекция топлива, % ({g.stat}); красный — богаче цели, синий — беднее")

    def _cell_clicked(self, i: int, j: int) -> None:
        g = self.grid_result
        if g is None:
            return
        text = f"RPM {g.x[j]}, kPa {g.y[i]}: точек {g.counts[i, j]}"
        v = g.values[i, j]
        if np.isfinite(v):
            text += f", коррекция {(v - 1) * 100:+.1f} %"
            if self.ve_new is not None:
                if self.ve_base is not None:
                    text += f", VE {self.ve_base.values[i, j]:.0f} → {self.ve_new.values[i, j]:.0f}"
                else:
                    text += f", VE новый {self.ve_new.values[i, j]:.0f}"
        else:
            text += f" (меньше {g.min_points}, не считается)"
            if self.ve_base is not None:
                text += f", VE из тюна {self.ve_base.values[i, j]:.0f}"
        self.cell_info.configure(text=text)
        if self.corr_log is not None:
            self.cell_panel.show(self.corr_log, g, i, j)

    def _update_export(self) -> None:
        self.export_btn.state(["!disabled"] if self.ve_new is not None else ["disabled"])

    def export_ve(self) -> None:
        if self.ve_new is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Экспорт VE", defaultextension=".table", initialfile="VE_NEW.table",
            filetypes=[("TunerStudio table", "*.table"), ("Все", "*")])
        if not path:
            return
        try:
            export_table(self.ve_new, path)
        except OSError as e:
            messagebox.showerror("Экспорт", str(e), parent=self)
            return
        self.cell_info.configure(text=f"сохранено: {os.path.basename(path)}")
