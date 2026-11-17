"""Shared figure style: one palette and one set of matplotlib defaults for every testbed.

Fixed categorical colour per optimizer (SGD/GD blue, spectral GD orange, Muon aqua, AdamW yellow),
sequential blues for ordered families (singular values, gamma sweeps), text and chrome in neutral
inks, light surface, recessive grid, one y-axis per panel. Importing this module applies the
defaults and selects the non-interactive Agg backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = {"gd": "#2a78d6", "spectral": "#eb6834", "muon": "#1baf7a", "adamw": "#eda100", "other": "#eda100",
          "fifth": "#e87ba4"}
SEQ_BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#b7d3f6", "#cde2fb"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "font.size": 9, "legend.frameon": False, "figure.dpi": 130,
})
