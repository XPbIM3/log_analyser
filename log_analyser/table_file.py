"""TunerStudio ``.table`` file export/import (the format legacy wrote via VE.table.template)."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from .axis import Axis, KpaAxis, RpmAxis
from .table import Table, TableKind

NAMESPACE = "http://www.EFIAnalytics.com/:table"


class TableFileError(ValueError):
    """Raised when a .table file cannot be read."""


def _fmt(v: float, digits: int) -> str:
    return f"{v:.{digits}f}" if digits > 0 else str(int(round(v)))


def table_to_string(table: Table, digits: int | None = None,
                    x_name: str = "rpm", y_name: str = "fuelLoad") -> str:
    """Serialize ``table`` into TunerStudio .table XML.

    ``digits`` defaults to ``table.kind.digits``. Rows are written in axis order
    (row 0 = ``y[0]``), as TunerStudio and the legacy exporter do.
    """
    d = table.kind.digits if digits is None else digits
    if d < 0:
        raise ValueError(f"digits must be >= 0, got {d}")
    rows, cols = table.shape
    xs = "\n".join(str(v) for v in table.x)
    ys = "\n".join(str(v) for v in table.y)
    zs = "\n".join(" ".join(_fmt(v, d) for v in row) for row in table.values)
    date = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<tableData xmlns="{NAMESPACE}">\n'
        f'<bibliography author="log_analyser" company="" writeDate="{date}"/>\n'
        '<versionInfo fileFormat="1.0"/>\n'
        f'<table cols="{cols}" rows="{rows}">\n'
        f'<xAxis cols="{cols}" name="{x_name}">\n{xs}\n</xAxis>\n'
        f'<yAxis name="{y_name}" rows="{rows}">\n{ys}\n</yAxis>\n'
        f'<zValues cols="{cols}" rows="{rows}">\n{zs}\n</zValues>\n'
        '</table>\n'
        '</tableData>\n'
    )


def export_table(table: Table, path: str | os.PathLike, digits: int | None = None,
                 x_name: str = "rpm", y_name: str = "fuelLoad") -> None:
    """Write ``table`` to ``path`` in TunerStudio .table format."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(table_to_string(table, digits, x_name, y_name))


def read_table(path: str | os.PathLike, kind: TableKind, caption: str = "",
               x_cls: type[Axis] = RpmAxis, y_cls: type[Axis] = KpaAxis,
               x_quantum: int | None = None, y_quantum: int | None = None) -> Table:
    """Read a TunerStudio .table file back into a :class:`Table`."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as e:
        raise TableFileError(f"{path}: cannot read: {e}") from None
    ns = {"t": NAMESPACE}
    parts = {k: root.find(f"t:table/t:{k}", ns) for k in ("xAxis", "yAxis", "zValues")}
    missing = [k for k, el in parts.items() if el is None]
    if missing:
        raise TableFileError(f"{path}: missing elements {missing}")

    def nums(el):
        try:
            return np.array([float(t) for t in (el.text or "").split()])
        except ValueError as e:
            raise TableFileError(f"{path}: non-numeric data: {e}") from None

    x, y, z = nums(parts["xAxis"]), nums(parts["yAxis"]), nums(parts["zValues"])
    if z.size != x.size * y.size:
        raise TableFileError(f"{path}: {z.size} z values for {y.size}x{x.size} axes")
    try:
        return Table(kind, x_cls(x, quantum=x_quantum), y_cls(y, quantum=y_quantum),
                     z.reshape(y.size, x.size), caption=caption)
    except ValueError as e:
        raise TableFileError(f"{path}: {e}") from None
