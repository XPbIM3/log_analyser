"""Shared, UI-independent state of the application: loaded logs and tune."""
from __future__ import annotations

import warnings
from typing import Callable

from ..datalog import Log
from ..table import Table
from ..tune import Tune


class Session:
    """Loaded logs (by path, in insertion order) and the tune; notifies listeners on change."""

    def __init__(self) -> None:
        self.logs: dict[str, Log] = {}
        self.tune: Tune | None = None
        self.tune_warnings: list[str] = []
        self.tables: dict[str, Table] = {}
        self._combined: Log | None = None
        self._combine_notes: list[str] = []
        self._listeners: list[Callable[[], None]] = []

    def subscribe(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _changed(self) -> None:
        self._combined = None
        for fn in self._listeners:
            fn()

    # ------------------------------------------------------------------ logs

    def add_log(self, path: str, log: Log) -> None:
        self.logs[path] = log
        self._changed()

    def remove_logs(self, paths) -> None:
        for p in paths:
            self.logs.pop(p, None)
        self._changed()

    def clear_logs(self) -> None:
        self.logs.clear()
        self._changed()

    def combined(self) -> tuple[Log, list[str]]:
        """All logs concatenated (common columns only); raises ``LogError`` if none/incompatible."""
        if self._combined is None:
            logs = list(self.logs.values())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._combined = Log.concat(logs, columns="common")
            dropped = sorted(set().union(*(l.columns for l in logs)) - set(self._combined.columns))
            self._combine_notes = (
                [f"колонки есть не во всех файлах, отброшены: {', '.join(dropped)}"] if dropped else [])
        return self._combined, self._combine_notes

    # ------------------------------------------------------------------ tune

    def set_tune(self, tune: Tune | None) -> None:
        self.tune = tune
        self.tune_warnings = []
        if tune is not None:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.tables = tune.tables()
            self.tune_warnings = [str(w.message) for w in caught]
        else:
            self.tables = {}
        self._changed()
