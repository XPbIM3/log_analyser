"""Window for creating an AFR target table: two axis controls + AFR levels, live preview.

The lean → rich transition is set by two-handle sliders next to the table: on the
left along load (kPa), at the bottom along RPM (optional, combined with max).
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..afr_target import AfrFill, afr_target_table
from ..axis import KpaAxis, RpmAxis
from ..table import Table, TableKind
from ..table_file import export_table, read_table
from .afr_fill_control import AfrFillControl
from .axis_control import AxisControl
from .range_slider import AxisRangeSlider
from .table_grid import LABEL_H, LABEL_W, TITLE_H, TableGrid
from .widgets import init_styles

DEBOUNCE_MS = 50

# Defaults of the AFR Target tab
RPM_DEFAULT = dict(start=600, stop=6000, n=16, gamma=1.5)
KPA_DEFAULT = dict(start=30, stop=100, n=16, gamma=1.5)
# transition handles default to 0 … axis max (the sliders' own range is 0 … 2× axis max):
# e.g. for the 30…100 kPa axis the transition runs 0…100, adjustable up to 0…200
KPA_SLIDER_DEFAULT = (0, KPA_DEFAULT["stop"])
RPM_SLIDER_DEFAULT = (0, RPM_DEFAULT["stop"])
FILL_DEFAULT = AfrFill(lean=15.0, rich=12.5,
                       kpa_lean=KPA_SLIDER_DEFAULT[0], kpa_rich=KPA_SLIDER_DEFAULT[1],
                       rpm_lean=RPM_SLIDER_DEFAULT[0], rpm_rich=RPM_SLIDER_DEFAULT[1])
KPA_SLIDER_Q = 1
RPM_SLIDER_Q = 50
SLIDER_OVERRUN = 48                # px of slider track beyond the table on each side


class AfrTargetApp(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=8)
        self.root = self.winfo_toplevel()
        init_styles(self.root)
        self._pending: str | None = None

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.rpm = AxisControl(left, RpmAxis, "Ось RPM", **RPM_DEFAULT,
                               range_max=12000, on_change=self.schedule_update)
        self.kpa = AxisControl(left, KpaAxis, "Ось нагрузки, kPa", **KPA_DEFAULT,
                               range_max=300, on_change=self.schedule_update)
        self.fill = AfrFillControl(left, FILL_DEFAULT, on_change=self.schedule_update)
        for w in (self.rpm, self.kpa, self.fill):
            w.pack(fill=tk.X, pady=(0, 8))

        # right: table with a load slider on the left and an RPM slider below; the empty
        # strips around the table give the sliders room beyond the axis range (0 … 2×max)
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Frame(right, height=SLIDER_OVERRUN).grid(row=0, column=1)
        self.grid_view = TableGrid(right, cmap="RdYlBu", vmin=10.5, vmax=16.0)
        self.grid_view.grid(row=1, column=1, sticky="nsew")
        ttk.Frame(right, height=SLIDER_OVERRUN).grid(row=2, column=1)
        ttk.Frame(right, width=SLIDER_OVERRUN).grid(row=1, column=2)
        self.kpa_slider = AxisRangeSlider(
            right, self.kpa.axis, FILL_DEFAULT.kpa_lean, FILL_DEFAULT.kpa_rich, orient="vertical",
            pad_lo=SLIDER_OVERRUN + LABEL_H + 2, pad_hi=SLIDER_OVERRUN + TITLE_H,
            quantum=KPA_SLIDER_Q, on_change=self._ranges_changed)
        self.kpa_slider.grid(row=0, column=0, rowspan=3, sticky="ns")
        rpm = self.rpm.axis
        self.rpm_slider = AxisRangeSlider(
            right, rpm, *RPM_SLIDER_DEFAULT, orient="horizontal",
            pad_lo=LABEL_W, pad_hi=SLIDER_OVERRUN + 2,
            quantum=RPM_SLIDER_Q, on_change=self._ranges_changed)
        self.rpm_slider.grid(row=3, column=1, columnspan=2, sticky="ew")
        self.rpm_on = tk.BooleanVar(value=FILL_DEFAULT.rpm_lean is not None)
        ttk.Checkbutton(right, text="RPM", variable=self.rpm_on,
                        command=self._ranges_changed).grid(row=3, column=0)
        self.rpm_slider.set_enabled(self.rpm_on.get())
        right.columnconfigure(1, weight=1)
        right.rowconfigure(1, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(bar, text="Экспорт .table…", command=self.export).pack(side=tk.LEFT)
        ttk.Button(bar, text="Открыть .table…", command=self.open).pack(side=tk.LEFT, padx=6)
        self.status = ttk.Label(bar, foreground="#555555")
        self.status.pack(side=tk.LEFT, padx=12)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.table: Table = self._build()
        self.update_now()

    def _ranges_changed(self) -> None:
        on = self.rpm_on.get()
        self.rpm_slider.set_enabled(on)
        self.fill.set_ranges(kpa_lean=self.kpa_slider.lo, kpa_rich=self.kpa_slider.hi,
                             rpm_lean=self.rpm_slider.lo if on else None,
                             rpm_rich=self.rpm_slider.hi if on else None)

    def _build(self) -> Table:
        return afr_target_table(self.rpm.axis, self.kpa.axis, self.fill.fill)

    def schedule_update(self) -> None:
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(DEBOUNCE_MS, self.update_now)

    def update_now(self) -> None:
        self._pending = None
        self.table = self._build()
        self.grid_view.show(self.table)
        self.kpa_slider.set_axis(self.kpa.axis)
        self.rpm_slider.set_axis(self.rpm.axis)
        self.status.configure(text=f"{self.table.shape[0]}×{self.table.shape[1]}, сгенерирована")

    def export(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Экспорт AFR target", defaultextension=".table",
            initialfile="AFR_TARGET.table", filetypes=[("TunerStudio table", "*.table"), ("Все", "*")])
        if not path:
            return
        try:
            export_table(self.table, path)
        except OSError as e:
            messagebox.showerror("Экспорт", str(e), parent=self.root)
            return
        self.status.configure(text=f"сохранено: {Path(path).name}")

    def open(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root, title="Открыть .table",
            filetypes=[("TunerStudio table", "*.table"), ("Все", "*")])
        if not path:
            return
        try:
            table = read_table(path, TableKind.AFR, caption=Path(path).name)
        except ValueError as e:
            messagebox.showerror("Открыть", str(e), parent=self.root)
            return
        self.grid_view.show(table)
        self.status.configure(
            text=f"просмотр: {Path(path).name} (любое изменение параметров вернёт генерацию)")


def main() -> None:
    from .app import main as app_main
    app_main()


if __name__ == "__main__":
    main()
