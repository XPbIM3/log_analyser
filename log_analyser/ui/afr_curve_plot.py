"""AFR(kPa) curve preview embedded in tk."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..afr_target import AfrFill, afr_curve
from ..axis import Axis


class AfrCurvePlot(ttk.Frame):
    """Continuous AFR(kPa) curve, the rounded values at axis nodes, and the knees."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 2.6), dpi=96, layout="constrained")
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show(self, kpa: Axis, fill: AfrFill) -> None:
        ax = self.ax
        ax.clear()
        lo = min(float(kpa.values[0]), fill.kpa_lean)
        hi = max(float(kpa.values[-1]), fill.kpa_rich)
        k = np.linspace(lo, hi, 400)
        ax.plot(k, afr_curve(k, fill, rounded=False), color="#1f5fa8", lw=1.5, label="кривая")
        ax.plot(kpa.values, afr_curve(kpa.values, fill), "o", ms=4, color="#d0442c",
                label="узлы оси kPa")
        for x in (fill.kpa_lean, fill.kpa_rich):
            ax.axvline(x, color="#888888", ls="--", lw=0.8)
        ax.set_xlim(lo, hi)
        pad = max(0.3, (fill.lean - fill.rich) * 0.1)
        ax.set_ylim(fill.rich - pad, fill.lean + pad)
        ax.set_xlabel("kPa")
        ax.set_ylabel("AFR")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        self.canvas.draw_idle()
