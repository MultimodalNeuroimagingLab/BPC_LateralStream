"""
plot_bpcs.py
============
Plot the BPC basis curves (one line per BPC) on a single time axis (Plotly).

Takes the dict returned by ``bpc_identification.identify_bpcs(...)`` (or a bare
``(n_components, n_times)`` array of basis curves) plus the time vector.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from functions.colors import bpc_color


def plot_bpcs(
    bpcs,
    times,
    xlim: tuple[float, float] | None = (0.0, 0.5),
    skip_empty: bool = True,
    title: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot BPC basis curves vs time.

    Parameters
    ----------
    bpcs : dict from ``identify_bpcs(...)`` (uses key 'Bs') OR a bare
        (n_components, n_times) ndarray.
    times : (n_times,) seconds.
    xlim : (xmin, xmax) x-range (default 0.0–0.5 s; ``None`` = full range).
    skip_empty : omit BPCs whose curve is identically zero (no pairs assigned).
    """
    Bs = np.asarray(bpcs["Bs"] if isinstance(bpcs, dict) else bpcs)
    times = np.asarray(times)
    if Bs.shape[-1] != times.shape[0]:
        raise ValueError(f"Bs last axis = {Bs.shape[-1]} but times has "
                         f"{times.shape[0]} samples.")

    fig = go.Figure()
    for i, b in enumerate(Bs):
        if skip_empty and not np.any(b):
            continue
        fig.add_trace(go.Scatter(
            x=times, y=b, mode="lines", name=f"BPC {i}",
            line=dict(color=bpc_color(i), width=2),
            hovertemplate=f"<b>BPC {i}</b><br>%{{x:.3f}} s<br>%{{y:.3g}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="lightgray", width=1))
    fig.update_layout(
        title=title or "Basis profile curves",
        xaxis=dict(title="time from stimulation (s)",
                   range=list(xlim) if xlim else None),
        yaxis_title="voltage (unit-norm)",
        template="plotly_white", height=460)
    if show:
        fig.show()
    return fig
