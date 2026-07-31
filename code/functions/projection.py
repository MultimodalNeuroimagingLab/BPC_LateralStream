"""
projection.py
=============
Cross-trial projection matrix P for the BPC pipeline.

    P = V0 @ V.T,   V0 = V / ||V_trial||_2

P[i, j] is the inner product of unit-normalized trial i with trial j — a
similarity matrix across trials.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def projection_matrix(V, stim_sites, plot: bool = False,
                      title: str | None = None) -> np.ndarray:
    """Compute the trial-by-trial cross-projection matrix P.

    V : (n_trials, n_times). stim_sites : (n_trials,) used only to group the
    heatmap when ``plot=True``. Returns P (n_trials, n_trials).
    """
    V = np.asarray(V)
    V0 = V / np.linalg.norm(V, axis=1, keepdims=True)
    P = V0 @ V.T
    if plot:
        plot_projection_matrix(P, stim_sites, title=title, show=True)
    return P


def plot_projection_matrix(P, stim_sites, title: str | None = None,
                           show: bool = False) -> go.Figure:
    """Heatmap of P with trials reordered/grouped by stim pair."""
    P, stim_sites = np.asarray(P), np.asarray(stim_sites)
    pairs = list(dict.fromkeys(stim_sites))
    order = np.concatenate([np.where(stim_sites == p)[0] for p in pairs])
    P_ord, lab = P[np.ix_(order, order)], stim_sites[order]
    counts = np.array([int((lab == p).sum()) for p in pairs])
    ticks = np.concatenate([[0], np.cumsum(counts)[:-1]]) + (counts - 1) / 2

    fig = go.Figure(go.Heatmap(
        z=P_ord, colorscale="RdBu_r", zmid=0,
        zmin=float(np.percentile(P, 1)), zmax=float(np.percentile(P, 99)),
        colorbar=dict(title="P"),
        hovertemplate="%{y} → %{x}<br>%{z:.3f}<extra></extra>"))
    for e in np.cumsum(counts)[:-1] - 0.5:                   # pair-block dividers
        fig.add_hline(y=float(e), line=dict(color="white", width=1))
        fig.add_vline(x=float(e), line=dict(color="white", width=1))
    axis = dict(tickmode="array", tickvals=ticks, ticktext=[str(p) for p in pairs],
                tickfont=dict(size=8))
    fig.update_layout(
        title=title or "Cross-trial projection matrix P",
        xaxis=dict(**axis, side="bottom"),
        yaxis=dict(**axis, autorange="reversed", scaleanchor="x", scaleratio=1),
        template="plotly_white", height=680, width=760)
    if show:
        fig.show()
    return fig
