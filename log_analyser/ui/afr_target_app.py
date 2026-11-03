"""Window for creating an AFR target table: two axis controls + AFR fill, live preview."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..afr_target import AfrFill, afr_target_table
from ..axis import KpaAxis, RpmAxis
from ..table import Table, TableKind
from ..table_file import export_table, read_table
from .afr_curve_plot import AfrCurvePlot
from .afr_fill_control import AfrFillControl
from .axis_control import AxisControl
from .table_grid import TableGrid
from .widgets import init_styles

DEBOUNCE_MS = 50

# Defaults of the AFR Target tab
RPM_DEFAULT = dict(start=600, stop=6000, n=16, gamma=1.5)
KPA_DEFAULT = dict(start=30, stop=100, n=16, gamma=1.5)
FILL_DEFAULT = AfrFill(lean=15.0, rich=12.5, kpa_lean=0, kpa_rich=95, bias=1.0)


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
        self.fill = AfrFillControl(left, FILL_DEFAULT, kpa_max=300, on_change=self.schedule_update)
        for w in (self.rpm, self.kpa, self.fill):
            w.pack(fill=tk.X, pady=(0, 8))

        right = ttk.PanedWindow(self, orient=tk.VERTICAL)
        right.grid(row=0, column=1, sticky="nsew")
        self.grid_view = TableGrid(right, cmap="RdYlBu", vmin=10.5, vmax=16.0)
        self.curve = AfrCurvePlot(right)
        right.add(self.grid_view, weight=3)
        right.add(self.curve, weight=2)

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
        self.curve.show(self.kpa.axis, self.fill.fill)
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
