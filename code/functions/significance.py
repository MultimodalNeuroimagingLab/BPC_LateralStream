"""
significance.py
===============
Significance matrix Xi (``tmat``) for the BPC pipeline.

For every pair of stim-pair groups (k, l), ``tmat[k, l]`` is the one-sample
t-stat of the cross-projection values in P between trials of pair k and pair l.
The within-pair (diagonal) block uses only off-diagonal entries so a trial isn't
compared against itself.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def significance_matrix(P, stim_sites, pairs=None, plot: bool = False,
                        vmax: float = 10.0, title: str | None = None):
    """Compute the significance matrix ``tmat`` from P and stim_sites.

    Returns ``(tmat, pairs)`` where ``pairs`` is the pair-order used for the axes.
    """
    stim_sites = np.asarray(stim_sites)
    if pairs is None:
        pairs = np.array(sorted(np.unique(stim_sites)))
    pairs = np.asarray(pairs)

    tmat = np.zeros((len(pairs), len(pairs)))
    for i, p1 in enumerate(pairs):
        for j, p2 in enumerate(pairs):
            b = P[np.ix_(stim_sites == p1, stim_sites == p2)]
            if i == j:
                b = np.concatenate([b[np.tril_indices(b.shape[0], k=-1)],
                                    b[np.triu_indices(b.shape[0], k=1)]])
            b = b.ravel()
            tmat[i, j] = np.mean(b) * np.sqrt(len(b)) / np.std(b, ddof=1)

    if plot:
        plot_significance_matrix(tmat, pairs, vmax=vmax, title=title, show=True)
    return tmat, pairs


def plot_significance_matrix(tmat, pairs, vmax: float = 10.0,
                             title: str | None = None, show: bool = False) -> go.Figure:
    """Heatmap of the significance matrix Xi."""
    tmat, pairs = np.asarray(tmat), np.asarray(pairs)
    axis = dict(tickmode="array", tickvals=list(range(len(pairs))),
                ticktext=[str(p) for p in pairs], tickfont=dict(size=8))
    fig = go.Figure(go.Heatmap(
        z=tmat, zmin=0, zmax=vmax, colorscale="Viridis", colorbar=dict(title="t"),
        hovertemplate="%{y} → %{x}<br>t = %{z:.2f}<extra></extra>"))
    fig.update_layout(
        title=title or "Significance matrix Ξ (t-stat)",
        xaxis=dict(**axis, side="bottom"),
        yaxis=dict(**axis, autorange="reversed", scaleanchor="x", scaleratio=1),
        template="plotly_white", height=680, width=760)
    if show:
        fig.show()
    return fig
