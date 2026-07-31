"""
colors.py
=========
Shared BPC / cluster color palette (matplotlib ``tab10``, as hex).

``bpc_color(k)`` is the color for BPC / cluster index ``k`` — the SAME order the
inflated-brain markers use, so basis-curve lines and brain markers always agree.
This is a leaf module (no project imports) so any plotter can use it without
risking an import cycle.
"""

from __future__ import annotations

# matplotlib tab10, in order
BPC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def bpc_color(k) -> str:
    """Hex color for BPC / cluster index ``k`` (cycles every 10)."""
    return BPC_COLORS[int(k) % 10]
