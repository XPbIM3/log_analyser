"""TunerStudio data logs (.msl): import, concatenation, filtering, correction factor."""
from __future__ import annotations

import glob
import io
import os
import warnings
from datetime import datetime
from typing import Callable, Iterable, Sequence, Union

import numpy as np
import pandas as pd

from .table import Table, TableKind

Condition = Union[np.ndarray, pd.Series, Callable[[pd.DataFrame], Union[np.ndarray, pd.Series]]]

DEFAULT_SHIFT_COLUMNS = ("AFR", "Lambda")
SOURCE_COLUMN = "source"
MARK_PREFIX = "MARK "

# target column -> measured column
_MEASURED_FOR_TARGET_COLUMN = {"AFR Target": "AFR", "Lambda Target": "Lambda"}
_MEASURED_FOR_KIND = {TableKind.AFR: "AFR", TableKind.LAMBDA: "Lambda"}


class LogError(ValueError):
    """Raised when a log cannot be read or an operation on it is invalid."""


def _parse_capture_date(text: str) -> datetime | None:
    # "Thu Sep 19 01:24:25 YEKT 2024" — the time zone abbreviation is not parseable, drop it
    parts = text.split()
    if len(parts) == 6:
        parts = parts[:4] + parts[5:]
    try:
        return datetime.strptime(" ".join(parts), "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return None


class Log:
    """A data log as a ``pd.DataFrame`` plus metadata from the .msl header."""

    def __init__(self, data: pd.DataFrame, sources: Sequence[str] = (), signature: str = "",
                 firmware: str = "", capture_date: str = "", units: dict[str, str] | None = None,
                 marks: Sequence[tuple[float, str]] = ()) -> None:
        self.data = data
        self.sources = tuple(sources)
        self.signature = signature
        self.firmware = firmware
        self.capture_date = capture_date
        self.units = dict(units or {})
        # TunerStudio "MARK nnn - ..." lines: (Time of the last data row before the mark, text)
        self.marks = tuple(marks)

    # ------------------------------------------------------------------ import

    @classmethod
    def load(cls, path: str | os.PathLike, shift: int = 1,
             shift_columns: Iterable[str] = DEFAULT_SHIFT_COLUMNS) -> "Log":
        """Read a TunerStudio .msl file.

        By default the mixture columns (``AFR``, ``Lambda``) are shifted one row up:
        row ``i`` gets the mixture measured in row ``i + shift`` — the engine's
        response to the parameters (VE, PW, ...) of row ``i``. Rows left without a
        value after the shift are dropped. ``shift=0`` disables it.

        ``MARK ...`` lines that TunerStudio writes into the data (going offline/online,
        ECU reset, manual marks) are skipped and kept in :attr:`marks`. They split the
        log into segments; the shift is applied within each segment, so a mixture
        value never crosses a reconnect/reset gap.
        """
        path = os.fspath(path)
        if shift < 0:
            raise LogError(f"{path}: shift must be >= 0, got {shift}")
        try:
            with open(path, "r", encoding="latin-1") as f:
                lines = f.read().splitlines()
        except OSError as e:
            raise LogError(f"{path}: cannot read: {e}") from None

        header = next((i for i, l in enumerate(lines) if l.startswith("Time\t")), None)
        if header is None:
            raise LogError(f"{path}: no header line starting with 'Time'")

        firmware, capture_date = "", ""
        for l in lines[:header]:
            l = l.strip().strip('"')
            if l.startswith("Capture Date:"):
                # "Capture Date: Tue Aug 11 15:17:18 YEKT 2026, File author: TunerStudio ..."
                capture_date = l.split(":", 1)[1].split(",", 1)[0].strip()
            elif l and not firmware:
                firmware = l
        signature = firmware.split(":", 1)[0].strip()

        names = lines[header].split("\t")
        skip = list(range(header))
        units: dict[str, str] = {}
        first = header + 1
        if first < len(lines):
            head = lines[first].split("\t")[0].strip()
            try:
                float(head)
            except ValueError:  # not a number -> units row
                units = {n: u for n, u in zip(names, lines[first].split("\t")) if u}
                skip.append(first)

        data_start = skip[-1] + 1 if skip and skip[-1] > header else header + 1
        data_lines: list[str] = []
        segments: list[int] = []
        marks: list[tuple[int, str]] = []          # (number of data rows before the mark, text)
        seg = 0
        for n, l in enumerate(lines[data_start:], start=data_start + 1):
            if not l:
                continue
            if l.startswith(MARK_PREFIX):
                marks.append((len(data_lines), l.strip()))
                if data_lines and (not segments or segments[-1] == seg):
                    seg += 1
                continue
            if l.count("\t") + 1 != len(names):
                raise LogError(f"{path}: cannot parse line {n}: {l.count(chr(9)) + 1} fields, "
                               f"header has {len(names)}")
            data_lines.append(l)
            segments.append(seg)
        if not data_lines:
            raise LogError(f"{path}: no data rows")

        try:
            df = pd.read_csv(io.StringIO("\n".join([lines[header]] + data_lines)), sep="\t",
                             header=0, index_col=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
            raise LogError(f"{path}: cannot parse data: {e}") from None
        if df.empty:
            raise LogError(f"{path}: no data rows")

        # TunerStudio writes a stale first row: Time is NaN or left over from the previous
        # recording (e.g. 4884.3 followed by 0.000); the rest of it duplicates the next row.
        if "Time" in df.columns and len(df) > 1:
            t0, t1 = df["Time"].iloc[0], df["Time"].iloc[1]
            if pd.isna(t0) or t0 > t1:
                df = df.iloc[1:].reset_index(drop=True)
                segments = segments[1:]
                marks = [(max(0, i - 1), text) for i, text in marks]
                if df.empty:
                    raise LogError(f"{path}: no data rows")

        time = df["Time"].to_numpy() if "Time" in df.columns else None
        mark_list = [(float(time[i - 1]) if time is not None and i > 0 else float("nan"), text)
                     for i, text in marks]

        if shift:
            cols = [c for c in shift_columns if c in df.columns]
            if cols:
                grouped = df[cols].groupby(np.asarray(segments), sort=False)
                df[cols] = grouped.shift(-shift)
                df = df.dropna(subset=cols)
        df = df.reset_index(drop=True)
        df[SOURCE_COLUMN] = os.path.basename(path)

        return cls(df, sources=(path,), signature=signature, firmware=firmware,
                   capture_date=capture_date, units=units, marks=mark_list)

    @classmethod
    def load_many(cls, pattern: str | Sequence[str | os.PathLike], columns: str = "same", **load_kwargs) -> "Log":
        """Load files by glob pattern (or a list of paths), order by capture date, concatenate."""
        paths = sorted(glob.glob(pattern)) if isinstance(pattern, str) else [os.fspath(p) for p in pattern]
        if not paths:
            raise LogError(f"no log files match {pattern!r}")
        logs = [cls.load(p, **load_kwargs) for p in paths]
        logs.sort(key=lambda l: (_parse_capture_date(l.capture_date) or datetime.max, l.sources))
        return cls.concat(logs, columns=columns)

    # ------------------------------------------------------------------ combine / filter

    @classmethod
    def concat(cls, logs: Sequence["Log"], columns: str = "same") -> "Log":
        """Concatenate logs.

        ``columns="same"``: column sets must match exactly (error otherwise);
        ``columns="common"``: keep only columns present in every log, warn about the rest.
        """
        if columns not in ("same", "common"):
            raise LogError(f"columns must be 'same' or 'common', got {columns!r}")
        logs = list(logs)
        if not logs:
            raise LogError("nothing to concatenate")

        signatures = {l.signature for l in logs if l.signature}
        if len(signatures) > 1:
            raise LogError(f"logs come from different firmware: {sorted(signatures)}")

        all_cols = set().union(*(l.columns for l in logs))
        common = set.intersection(*(set(l.columns) for l in logs))
        if all_cols != common:
            report = "; ".join(
                f"{', '.join(l.sources) or '<log>'} lacks {sorted(all_cols - set(l.columns))}"
                for l in logs if all_cols - set(l.columns)
            )
            if columns == "same":
                raise LogError(f"column sets differ: {report}")
            warnings.warn(f"dropping columns not present in every log: {sorted(all_cols - common)} ({report})",
                          stacklevel=2)
        order = [c for c in logs[0].columns if c in common]
        data = pd.concat([l.data[order] for l in logs], ignore_index=True)

        units: dict[str, str] = {}
        for l in logs:
            for k, v in l.units.items():
                if k in common:
                    units.setdefault(k, v)
        first = logs[0]
        return cls(data, sources=[s for l in logs for s in l.sources], signature=first.signature,
                   firmware=first.firmware, capture_date=first.capture_date, units=units,
                   marks=[m for l in logs for m in l.marks])

    def where(self, cond: Condition) -> "Log":
        """Rows where ``cond`` is true: a boolean mask or ``callable(df) -> mask``."""
        mask = cond(self.data) if callable(cond) else cond
        mask = np.asarray(mask)
        if mask.shape != (len(self.data),):
            raise LogError(f"mask shape {mask.shape} does not match log length {len(self.data)}")
        if mask.dtype != bool:
            raise LogError(f"mask must be boolean, got {mask.dtype}")
        return self._replace(self.data[mask].reset_index(drop=True))

    # ------------------------------------------------------------------ correction

    def with_corr(self, target: str | Table, measured: str | None = None,
                  x: str = "RPM", y: str = "MAP", name: str = "CORR") -> "Log":
        """Add ``name = measured / target`` and ``name + "_target"`` columns.

        ``target`` is a log column (``"AFR Target"``/``"Lambda Target"``) or a user
        ``Table`` of kind AFR/LAMBDA looked up at ``(x, y)`` (RPM, MAP by default).
        ``measured`` defaults to ``"AFR"`` or ``"Lambda"`` to match the target.
        CORR > 1 means leaner than target (VE should go up: ``VE_new = VE * CORR``).
        """
        if isinstance(target, Table):
            if target.kind not in _MEASURED_FOR_KIND:
                raise LogError(f"target table must be AFR or LAMBDA, got {target.kind.label}")
            measured = measured or _MEASURED_FOR_KIND[target.kind]
            tgt = target.lookup(self.column(x), self.column(y))
        else:
            if measured is None:
                if target not in _MEASURED_FOR_TARGET_COLUMN:
                    raise LogError(f"cannot infer measured column for target {target!r}; pass measured=")
                measured = _MEASURED_FOR_TARGET_COLUMN[target]
            tgt = self.column(target)
        meas = self.column(measured)
        if np.any(tgt == 0):
            raise LogError(f"target has zero values ({int(np.sum(tgt == 0))} rows)")
        data = self.data.copy()
        data[name + "_target"] = tgt
        data[name] = meas / tgt
        return self._replace(data)

    # ------------------------------------------------------------------ access

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    @property
    def source(self) -> str:
        return ", ".join(self.sources)

    def column(self, name: str) -> np.ndarray:
        if name not in self.data.columns:
            raise LogError(f"{self.source or '<log>'}: no column {name!r}")
        return self.data[name].to_numpy()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Log({len(self)} rows, {len(self.sources)} file(s), {self.signature!r})"

    def _replace(self, data: pd.DataFrame) -> "Log":
        return Log(data, sources=self.sources, signature=self.signature, firmware=self.firmware,
                   capture_date=self.capture_date, units=self.units, marks=self.marks)
