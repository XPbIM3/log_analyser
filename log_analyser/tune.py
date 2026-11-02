"""TunerStudio tune (.msq) parser: raw constants, axes and tables by name."""
from __future__ import annotations

import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .axis import Axis, KpaAxis, RpmAxis
from .table import Table, TableKind


class TuneError(ValueError):
    """Raised when a tune file is malformed or a requested constant is missing/unsuitable."""


@dataclass(frozen=True)
class TableSpec:
    """How a 2-D table is assembled from .msq constants (the .msq itself does not say)."""

    x: str
    y: str
    kind: TableKind
    caption: str
    x_cls: type[Axis] = RpmAxis
    y_cls: type[Axis] = KpaAxis


# Speeduino 2022xx table -> axes mapping (from the firmware .ini, checked against CurrentTune.msq).
SPEEDUINO_TABLES: dict[str, TableSpec] = {
    "veTable": TableSpec("rpmBins", "fuelLoadBins", TableKind.VE, "Primary VE Table"),
    "veTable2": TableSpec("fuelRPM2Bins", "fuelLoad2Bins", TableKind.VE, "Secondary VE Table"),
    "advTable1": TableSpec("rpmBins2", "mapBins1", TableKind.IGN, "Primary Ignition Table"),
    "advTable2": TableSpec("rpmBins3", "mapBins2", TableKind.IGN, "Secondary Ignition Table"),
    "afrTable": TableSpec("rpmBinsAFR", "loadBinsAFR", TableKind.AFR, "AFR Target Table"),
}


@dataclass(frozen=True)
class Constant:
    """One ``<constant>`` element of the tune."""

    name: str
    rows: int
    cols: int
    units: str
    digits: int | None
    text: str

    def array(self) -> np.ndarray:
        """Numeric values shaped ``(rows, cols)``."""
        tokens = self.text.split()
        try:
            arr = np.array([float(t) for t in tokens], dtype=np.float64)
        except ValueError:
            raise TuneError(f"constant {self.name!r} is not numeric: {self.text.strip()[:60]!r}") from None
        if arr.size != self.rows * self.cols:
            raise TuneError(
                f"constant {self.name!r}: expected {self.rows}x{self.cols}={self.rows * self.cols} values, got {arr.size}"
            )
        return arr.reshape(self.rows, self.cols)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class Tune:
    """Parsed .msq file. Use :meth:`load` or :meth:`from_string`."""

    def __init__(self, root: ET.Element, source: str = "<string>",
                 table_specs: dict[str, TableSpec] | None = None) -> None:
        if _local(root.tag) != "msq":
            raise TuneError(f"{source}: root element is <{_local(root.tag)}>, expected <msq>")
        self.source = source
        self.table_specs = dict(SPEEDUINO_TABLES if table_specs is None else table_specs)
        self.signature = ""
        self._constants: dict[str, Constant] = {}
        for el in root.iter():
            if _local(el.tag) == "versionInfo":
                self.signature = el.get("signature", "")
            if _local(el.tag) != "constant":
                continue
            name = el.get("name")
            if not name:
                continue
            if name in self._constants:
                raise TuneError(f"{source}: duplicate constant {name!r}")
            digits = el.get("digits")
            self._constants[name] = Constant(
                name=name,
                rows=int(el.get("rows", 1)),
                cols=int(el.get("cols", 1)),
                units=el.get("units", ""),
                digits=int(digits) if digits is not None else None,
                text=el.text or "",
            )

    @classmethod
    def load(cls, path: str | os.PathLike, table_specs: dict[str, TableSpec] | None = None) -> "Tune":
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as e:
            raise TuneError(f"{path}: cannot read tune: {e}") from None
        return cls(root, source=str(path), table_specs=table_specs)

    @classmethod
    def from_string(cls, xml: str | bytes, table_specs: dict[str, TableSpec] | None = None) -> "Tune":
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as e:
            raise TuneError(f"cannot parse tune: {e}") from None
        return cls(root, table_specs=table_specs)

    # ------------------------------------------------------------------ access

    @property
    def names(self) -> list[str]:
        return list(self._constants)

    def __contains__(self, name: str) -> bool:
        return name in self._constants

    def constant(self, name: str) -> Constant:
        try:
            return self._constants[name]
        except KeyError:
            raise TuneError(f"{self.source}: no constant named {name!r}") from None

    def axis(self, name: str, cls: type[Axis] = Axis, quantum: int | None = None) -> Axis:
        """1-D constant (``rows x 1`` or ``1 x cols``) as an axis of type ``cls``."""
        c = self.constant(name)
        if 1 not in (c.rows, c.cols):
            raise TuneError(f"constant {name!r} is {c.rows}x{c.cols}, not an axis")
        try:
            return cls(c.array().ravel(), quantum=quantum)
        except ValueError as e:
            raise TuneError(f"constant {name!r} is not a valid {cls.__name__}: {e}") from None

    def table(self, name: str, *, caption: str | None = None,
              x_quantum: int | None = None, y_quantum: int | None = None) -> Table:
        """2-D table by name, axes taken from :attr:`table_specs`."""
        spec = self.table_specs.get(name)
        if spec is None:
            raise TuneError(
                f"no axis mapping for table {name!r}; known: {sorted(self.table_specs)} "
                f"(pass table_specs or use table_from)"
            )
        return self.table_from(name, spec.x, spec.y, spec.kind,
                               caption=spec.caption if caption is None else caption,
                               x_cls=spec.x_cls, y_cls=spec.y_cls,
                               x_quantum=x_quantum, y_quantum=y_quantum)

    def table_from(self, name: str, x: str, y: str, kind: TableKind, *, caption: str = "",
                   x_cls: type[Axis] = RpmAxis, y_cls: type[Axis] = KpaAxis,
                   x_quantum: int | None = None, y_quantum: int | None = None) -> Table:
        """2-D table with explicitly named axis constants."""
        c = self.constant(name)
        if kind is TableKind.AFR and c.units.lower() == "lambda":
            kind = TableKind.LAMBDA
        xa = self.axis(x, x_cls, x_quantum)
        ya = self.axis(y, y_cls, y_quantum)
        try:
            return Table(kind, xa, ya, c.array(), caption=caption)
        except ValueError as e:
            raise TuneError(f"table {name!r} ({x} x {y}): {e}") from None

    def tables(self, strict: bool = False) -> dict[str, Table]:
        """All known tables present in this tune.

        Tables that fail validation (e.g. an unused secondary table with all-zero
        axes) are skipped with a warning, or raise if ``strict``.
        """
        out: dict[str, Table] = {}
        for n in self.table_specs:
            if n not in self:
                continue
            try:
                out[n] = self.table(n)
            except TuneError as e:
                if strict:
                    raise
                warnings.warn(f"skipping table {n!r}: {e}", stacklevel=2)
        return out
