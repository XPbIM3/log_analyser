"""Main window: tabs «Данные» (logs + tune), «AFR Target», «Коррекция»."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .afr_target_app import AfrTargetApp
from .corr_tab import CorrTab
from .logs_tab import LogsTab
from .session import Session
from .widgets import init_styles


class MainWindow(ttk.Frame):
    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root)
        init_styles(root)
        self.session = Session()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.logs = LogsTab(self.notebook, self.session)
        self.afr = AfrTargetApp(self.notebook)
        self.corr = CorrTab(self.notebook, self.session, generated_target=lambda: self.afr.table)
        self.notebook.add(self.logs, text="Данные")
        self.notebook.add(self.afr, text="AFR Target")
        self.notebook.add(self.corr, text="Коррекция")
        self.notebook.bind("<<NotebookTabChanged>>", self._tab_changed)

    def _tab_changed(self, _event=None) -> None:
        if self.notebook.select() == str(self.corr):
            self.corr.schedule()   # the generated target may have changed meanwhile


def main() -> None:
    root = tk.Tk()
    root.title("log_analyser")
    root.geometry("1280x820")
    root.minsize(1000, 650)
    MainWindow(root).pack(fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
