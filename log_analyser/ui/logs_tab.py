"""Tab for importing logs (.msl list) and the tune (.msq)."""
from __future__ import annotations

import os
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ..datalog import Log
from ..tune import Tune
from .session import Session

POLL_MS = 100
PAUSE_S = 5.0          # a gap between rows longer than this counts as a pause (offline, ECU off)


def duration_text(log: Log) -> str:
    """Elapsed minutes by the Time column, plus pauses longer than PAUSE_S: "81.4 (паузы 34.8)"."""
    if "Time" not in log.columns or len(log) < 2:
        return ""
    t = log.column("Time").astype(float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return ""
    d = np.diff(t)
    pauses = d[d > PAUSE_S].sum()
    text = f"{(t.max() - t.min()) / 60:.1f}"
    if pauses >= 6:
        text += f" (паузы {pauses / 60:.1f})"
    return text


class LogsTab(ttk.Frame):
    def __init__(self, parent: tk.Misc, session: Session) -> None:
        super().__init__(parent, padding=8)
        self.session = session
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._pending: dict[str, Future] = {}

        # --- tune
        tune_box = ttk.LabelFrame(self, text="Тюн (.msq)", padding=8)
        tune_box.pack(fill=tk.X)
        ttk.Button(tune_box, text="Открыть тюн…", command=self.open_tune).pack(side=tk.LEFT)
        self.tune_label = ttk.Label(tune_box, text="не загружен", foreground="#555555")
        self.tune_label.pack(side=tk.LEFT, padx=12)

        # --- logs
        logs_box = ttk.LabelFrame(self, text="Логи (.msl)", padding=8)
        logs_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        bar = ttk.Frame(logs_box)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Добавить логи…", command=self.add_logs).pack(side=tk.LEFT)
        ttk.Button(bar, text="Удалить выбранные", command=self.remove_selected).pack(side=tk.LEFT, padx=6)
        ttk.Button(bar, text="Очистить", command=self.clear).pack(side=tk.LEFT)

        cols = ("file", "date", "rows", "duration", "fw", "status")
        self.tree = ttk.Treeview(logs_box, columns=cols, show="headings", selectmode="extended")
        for cid, text, width, anchor in (
            ("file", "Файл", 260, "w"), ("date", "Дата записи", 200, "w"),
            ("rows", "Строк", 80, "e"), ("duration", "Длит., мин", 130, "e"),
            ("fw", "Прошивка", 200, "w"), ("status", "Статус", 260, "w"),
        ):
            self.tree.heading(cid, text=text)
            self.tree.column(cid, width=width, anchor=anchor, stretch=cid in ("file", "status"))
        scroll = ttk.Scrollbar(logs_box, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(6, 0))
        scroll.pack(side=tk.LEFT, fill=tk.Y, pady=(6, 0))
        self.tree.bind("<Delete>", lambda _e: self.remove_selected())

        self.summary = ttk.Label(self, foreground="#555555")
        self.summary.pack(fill=tk.X, pady=(6, 0))
        self._update_summary()

    # ------------------------------------------------------------------ tune

    def open_tune(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Открыть тюн",
                                          filetypes=[("TunerStudio tune", "*.msq"), ("Все", "*")])
        if path:
            self.load_tune(path)

    def load_tune(self, path: str) -> None:
        try:
            tune = Tune.load(path)
        except ValueError as e:
            messagebox.showerror("Тюн", str(e), parent=self)
            return
        self.session.set_tune(tune)
        names = ", ".join(self.session.tables) or "нет известных таблиц"
        text = f"{os.path.basename(path)}  ({tune.signature or 'без сигнатуры'})  таблицы: {names}"
        if self.session.tune_warnings:
            text += f"  — пропущено: {len(self.session.tune_warnings)}"
        self.tune_label.configure(text=text)

    # ------------------------------------------------------------------ logs

    def add_logs(self) -> None:
        paths = filedialog.askopenfilenames(parent=self, title="Добавить логи",
                                            filetypes=[("TunerStudio log", "*.msl"), ("Все", "*")])
        self.load_logs(paths)

    def load_logs(self, paths) -> None:
        for p in paths:
            p = os.path.normpath(p)
            if p in self.session.logs or p in self._pending:
                continue
            self.tree.insert("", tk.END, iid=p, values=(os.path.basename(p), "", "", "", "", "загрузка…"))
            self._pending[p] = self._pool.submit(Log.load, p)
        if self._pending:
            self.after(POLL_MS, self._poll)

    def _poll(self) -> None:
        for p, fut in list(self._pending.items()):
            if not fut.done():
                continue
            del self._pending[p]
            if not self.tree.exists(p):        # removed while loading
                continue
            try:
                log = fut.result()
            except Exception as e:  # noqa: BLE001 - show any load error in the list
                self.tree.item(p, values=(os.path.basename(p), "", "", "", "", f"ошибка: {e}"))
                continue
            dur = duration_text(log)
            self.tree.item(p, values=(os.path.basename(p), log.capture_date, len(log), dur,
                                      log.firmware, "ок"))
            self.session.add_log(p, log)
        self._update_summary()
        if self._pending:
            self.after(POLL_MS, self._poll)

    def remove_selected(self) -> None:
        sel = self.tree.selection()
        self.tree.delete(*sel)
        self.session.remove_logs(sel)
        self._update_summary()

    def clear(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.session.clear_logs()
        self._update_summary()

    def _update_summary(self) -> None:
        logs = self.session.logs
        if not logs:
            self.summary.configure(text="Логи не загружены")
            return
        rows = sum(len(l) for l in logs.values())
        sigs = {l.signature for l in logs.values() if l.signature}
        text = f"Файлов: {len(logs)}, строк: {rows}"
        if len(sigs) > 1:
            text += f"  — ВНИМАНИЕ: разные прошивки {sorted(sigs)}, объединить нельзя"
        if self._pending:
            text += f"  (загружается ещё {len(self._pending)})"
        self.summary.configure(text=text)
