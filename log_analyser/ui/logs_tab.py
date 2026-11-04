"""Tab for importing logs (.msl list) and the tune (.msq)."""
from __future__ import annotations

import os
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ..datalog import DEFAULT_DELAY_MS, Log
from ..shift_scan import ShiftEstimate, recommend, scan_file
from ..tune import Tune
from .session import Session

POLL_MS = 100
PAUSE_S = 5.0          # a gap between rows longer than this counts as a pause (offline, ECU off)
RELOAD_DELAY_MS = 500  # debounce for applying a new shift value


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

        shift_bar = ttk.Frame(logs_box)
        shift_bar.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(shift_bar, text="Сдвиг смеси, мс:").pack(side=tk.LEFT)
        self.shift_ms = tk.DoubleVar(value=DEFAULT_DELAY_MS)
        self._applied_shift = DEFAULT_DELAY_MS
        self._reload_job: str | None = None
        spin = ttk.Spinbox(shift_bar, from_=0, to=2000, increment=10, width=7,
                           textvariable=self.shift_ms, command=self.schedule_reload)
        spin.bind("<Return>", self.schedule_reload)
        spin.bind("<FocusOut>", self.schedule_reload)
        spin.pack(side=tk.LEFT, padx=(6, 12))
        self._scan_btn = ttk.Button(shift_bar, text="Подобрать сдвиг", command=self.detect_shift)
        self._scan_btn.pack(side=tk.LEFT)
        self._scans: dict[str, Future] = {}
        self._shift_dialog: tk.Toplevel | None = None
        self._shift_note: ttk.Label | None = None
        self._shift_rec: float | None = None
        ttk.Label(shift_bar, foreground="#555555",
                  text=f"рекомендация: {DEFAULT_DELAY_MS:.0f} мс (замер на протягах ~15 Гц) — "
                       "ряды = сдвиг ÷ герцовка файла").pack(side=tk.LEFT)

        cols = ("file", "date", "hz", "rows", "duration", "fw", "status")
        self.tree = ttk.Treeview(logs_box, columns=cols, show="headings", selectmode="extended")
        for cid, text, width, anchor in (
            ("file", "Файл", 260, "w"), ("date", "Дата записи", 200, "w"), ("hz", "Гц", 55, "e"),
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
            self.tree.insert("", tk.END, iid=p,
                             values=(os.path.basename(p), "", "", "", "", "", "загрузка…"))
            self._pending[p] = self._pool.submit(Log.load, p, shift_ms=self._applied_shift)
        if self._pending:
            self.after(POLL_MS, self._poll)

    def schedule_reload(self, _event=None) -> None:
        """The shift value changed: re-read all files (debounced)."""
        try:
            value = float(self.shift_ms.get())
        except (ValueError, tk.TclError):
            return                     # half-typed value: keep what is applied
        if value < 0 or value == self._applied_shift:
            return
        if self._reload_job is not None:
            self.after_cancel(self._reload_job)
        self._reload_job = self.after(RELOAD_DELAY_MS, self.reload_all)

    def reload_all(self) -> None:
        """Re-read every loaded file with the current shift value."""
        if self._reload_job is not None:
            self.after_cancel(self._reload_job)
            self._reload_job = None
        try:
            value = float(self.shift_ms.get())
        except (ValueError, tk.TclError):
            return
        value = max(0.0, value)
        self._applied_shift = value
        for p in set(self.session.logs) | set(self._pending):
            if not self.tree.exists(p):
                continue
            self.tree.item(p, values=(os.path.basename(p), "", "", "", "", "", "перечитывается…"))
            self._pending[p] = self._pool.submit(Log.load, p, shift_ms=value)
        if self._pending:
            self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------- shift detect

    def detect_shift(self) -> None:
        """Scan every loaded log for the mixture delay (in a background pool)."""
        if not self.session.logs:
            messagebox.showinfo("Подобрать сдвиг", "Сначала загрузите логи.", parent=self)
            return
        self._scan_btn.configure(state="disabled")
        self._scans = {p: self._pool.submit(scan_file, p) for p in self.session.logs}
        self.after(POLL_MS, self._poll_scans)

    def _poll_scans(self) -> None:
        if not all(f.done() for f in self._scans.values()):
            self.after(POLL_MS, self._poll_scans)
            return
        results, errors = [], []
        for p, fut in self._scans.items():
            try:
                results.append((os.path.basename(p), fut.result()))
            except Exception as e:  # noqa: BLE001 - show any scan error in the dialog
                errors.append((os.path.basename(p), str(e)))
        self._scans.clear()
        self._scan_btn.configure(state="normal")
        self._show_shift_dialog(results, errors)

    def _show_shift_dialog(self, results: list[tuple[str, ShiftEstimate]],
                           errors: list[tuple[str, str]]) -> None:
        rec, note = recommend([e for _, e in results])
        self._shift_rec = rec
        win = tk.Toplevel(self)
        win.title("Подбор сдвига смеси")
        win.transient(self)
        win.grab_set()
        self._shift_dialog = win
        cols = ("file", "hz", "k", "delay", "conf")
        tree = ttk.Treeview(win, columns=cols, show="headings",
                            height=min(8, max(3, len(results) + len(errors))))
        for cid, text, width, anchor in (("file", "Файл", 230, "w"), ("hz", "Гц", 55, "e"),
                                         ("k", "Сдвиг, рядов", 100, "e"),
                                         ("delay", "Задержка, мс", 105, "e"),
                                         ("conf", "Оценка", 330, "w")):
            tree.heading(cid, text=text)
            tree.column(cid, width=width, anchor=anchor)
        tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        for name, e in results:
            if e.confidence == "none":
                tree.insert("", tk.END, values=(name, "—", "—", "—", e.reason))
                continue
            conf = "уверенно" if e.confidence == "confident" else "слабо"
            text = conf + (f"; {e.reason}" if e.reason else "") + f" · {e.section}"
            tree.insert("", tk.END, values=(name, f"{e.rate_hz:.1f}", e.best_level,
                                            f"{e.delay_ms:.0f}", text))
        for name, err in errors:
            tree.insert("", tk.END, values=(name, "", "", "", f"ошибка: {err}"))
        self._shift_note = ttk.Label(win, wraplength=800, justify=tk.LEFT, foreground="#555555",
                                     text=(f"Рекомендация: сдвиг {rec:.0f} мс ({note})." if rec is not None
                                           else note))
        self._shift_note.pack(fill=tk.X, padx=8)
        bar = ttk.Frame(win)
        bar.pack(fill=tk.X, padx=8, pady=(4, 8))
        apply_btn = ttk.Button(bar, text="Применить", command=self._apply_shift,
                               state="normal" if rec is not None else "disabled")
        apply_btn.pack(side=tk.LEFT)
        ttk.Button(bar, text="Закрыть", command=win.destroy).pack(side=tk.LEFT, padx=6)

    def _apply_shift(self) -> None:
        if self._shift_rec is not None:
            self.shift_ms.set(round(self._shift_rec / 10.0) * 10.0)   # keep the spinbox step
            self.reload_all()
        if self._shift_dialog is not None:
            self._shift_dialog.destroy()
            self._shift_dialog = None

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
                self.tree.item(p, values=(os.path.basename(p), "", "", "", "", "", f"ошибка: {e}"))
                continue
            dur = duration_text(log)
            hz = f"{log.rate_hz:.1f}" if log.rate_hz else ""
            status = f"ок · сдвиг {log.shift_rows} ряд." if log.shift_rows else "ок"
            self.tree.item(p, values=(os.path.basename(p), log.capture_date, hz, len(log), dur,
                                      log.firmware, status))
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
