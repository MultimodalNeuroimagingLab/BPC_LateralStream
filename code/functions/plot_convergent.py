"""
plot_convergent.py
==================
Plot a convergent response matrix ``V`` (n_trials x n_times), grouped by stim
pair, either as a heatmap or as per-pair voltage traces (Plotly).

``kind='heatmap'`` (default): trial x time image (or pair x time if
``average_within_pair``), rows grouped by stim pair with the pair labels on the
y-axis. ``kind='trace'``: one mean-voltage line per stim pair vs time.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _pair_order(stim_sites):
    """Unique stim-pair labels in first-seen order + row indices grouped by pair."""
    stim_sites = np.asarray(stim_sites)
    pairs = list(dict.fromkeys(stim_sites))
    order = np.concatenate([np.where(stim_sites == p)[0] for p in pairs])
    return pairs, order


def plot_convergent_matrix(
    V,
    times,
    stim_sites,
    average_within_pair: bool = False,
    kind: str = "heatmap",
    title: str | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot ``V`` grouped by stim pair.

    Parameters
    ----------
    V : (n_trials, n_times).
    times : (n_times,) seconds re: stim onset.
    stim_sites : (n_trials,) stim-pair label per trial (defines grouping).
    average_within_pair : mean-collapse trials within each pair (heatmap only).
    kind : ``'heatmap'`` (default) or ``'trace'`` (per-pair mean voltage lines).
    """
    if kind not in ("heatmap", "trace"):
        raise ValueError("kind must be 'heatmap' or 'trace'.")
    V, times, stim_sites = np.asarray(V), np.asarray(times), np.asarray(stim_sites)
    pairs, order = _pair_order(stim_sites)

    if kind == "trace":
        fig = go.Figure()
        for i, p in enumerate(pairs):
            fig.add_trace(go.Scatter(
                x=times, y=V[stim_sites == p].mean(0), mode="lines",
                name=str(p), line=dict(width=1.5),
                hovertemplate=f"{p}<br>%{{x:.3f}} s<br>%{{y:.2f}}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color="lightgray", width=1))
        fig.update_layout(
            title=title or "Convergent matrix — per-pair voltage traces",
            xaxis_title="time from stimulation (s)", yaxis_title="voltage",
            template="plotly_white", height=520)
        if show:
            fig.show()
        return fig

    # heatmap
    V_ord, lab = V[order], stim_sites[order]
    if average_within_pair:
        rows = np.vstack([V_ord[lab == p].mean(0) for p in pairs])
        counts = np.ones(len(pairs), int)
    else:
        rows = V_ord
        counts = np.array([int((lab == p).sum()) for p in pairs])
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ticks = first + (counts - 1) / 2

    vmax = float(np.nanpercentile(np.abs(rows), 99))
    fig = go.Figure(go.Heatmap(
        z=rows, x=times, y=np.arange(rows.shape[0]),
        colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title="voltage"),
        hovertemplate="%{x:.3f} s<br>row %{y}<br>%{z:.2f}<extra></extra>"))
    if not average_within_pair:                              # white dividers between pairs
        for e in np.cumsum(counts)[:-1] - 0.5:
            fig.add_hline(y=float(e), line=dict(color="white", width=1))
    fig.update_layout(
        title=title or ("Convergent matrix (pair-averaged)" if average_within_pair
                        else "Convergent matrix"),
        xaxis_title="time from stimulation (s)",
        yaxis=dict(title="stimulation pair", autorange="reversed",
                   tickmode="array", tickvals=ticks, ticktext=[str(p) for p in pairs],
                   tickfont=dict(size=9)),
        template="plotly_white",
        height=max(420, 16 * len(rows) if average_within_pair else 700))
    if show:
        fig.show()
    return fig
