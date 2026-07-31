"""
nmf.py
======
Iterative NMF dimensionality reduction for the BPC pipeline.

Starts from a maximum inner dimension and decreases. At each n_components,
the model is re-fit `n_reruns` times (random init) and the lowest-error fit
is kept. Stops when the off-diagonal H@H.T "redundancy" penalty drops below
its threshold (an indication that the basis rows are no longer redundant).

Two penalty modes:
    - 'sum' : np.triu(H @ H.T, k=1).sum(); break when < 1.0   (default)
    - 'max' : np.triu(H @ H.T, k=1).max(); break when < 0.5
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import NMF


def run_nmf(
    tmat: np.ndarray,
    pairs: np.ndarray | None = None,
    cluster_dim: int = 9,
    n_reruns: int = 20,
    tol: float = 1e-5,
    random_state: int = 11,
    penalty: str = "sum",
    plot: bool = False,
    verbose: bool = True,
) -> dict:
    """Run iterative NMF on the (clipped, normalized) significance matrix.

    Parameters
    ----------
    tmat : (n_pairs, n_pairs) significance matrix Xi.
    pairs : (n_pairs,) labels, for plot axes only.
    cluster_dim : starting inner dimension (largest n_components tried).
    n_reruns : random restarts per n_components.
    penalty : 'sum' (default; break when < 1.0) or
              'max' (max upper-triangle element; break when < 0.5).
    plot : if True, plot H as a heatmap clipped to [0, 1] with stim-pair
           column labels and BPC row labels.

    Returns
    -------
    dict with keys:
        H : (n_components, n_pairs) row-L2-normalized component matrix.
        W : (n_pairs, n_components)
        n_components : final selected inner dimension.
        penalty : final penalty value.
        penalty_mode : 'sum' or 'max'.
        history : list of (n_components, penalty) tried.
        t0 : the clipped/normalized tmat actually fit.
    """
    if penalty not in ("sum", "max"):
        raise ValueError("penalty must be 'sum' or 'max'.")
    thresh = 1.0 if penalty == "sum" else 0.5

    t0 = tmat.copy()
    t0[t0 < 0] = 0
    t0[np.isnan(t0)] = 0
    if np.max(t0) > 0:
        t0 = t0 / np.max(t0)

    H = W = None
    n_components = cluster_dim
    history: list[tuple[int, float]] = []
    final_pen = np.nan

    for nc in range(cluster_dim, 1, -1):
        best_err = None
        for _ in range(n_reruns):
            model = NMF(n_components=nc, init="random", solver="mu",
                        tol=tol, max_iter=10000,
                        random_state=random_state).fit(t0)
            if best_err is None or model.reconstruction_err_ < best_err:
                best_err = model.reconstruction_err_
                W = model.transform(t0)
                H = model.components_
        H = H / np.linalg.norm(H, axis=1, keepdims=True)
        upper = np.triu(H @ H.T, k=1)
        pen = upper.sum() if penalty == "sum" else upper.max()
        history.append((nc, float(pen)))
        if verbose:
            print(f"Inner dimension: {nc}, {penalty} off-diagonal score: {pen:.4f}")
        n_components = nc
        final_pen = float(pen)
        if pen < thresh:
            break

    if plot:
        plot_nmf_H(H, pairs, show=True)

    return {
        "H": H,
        "W": W,
        "n_components": n_components,
        "penalty": final_pen,
        "penalty_mode": penalty,
        "history": history,
        "t0": t0,
    }


def plot_nmf_H(H, pairs=None, title: str | None = None, show: bool = False) -> go.Figure:
    """Heatmap of the NMF component matrix H, clipped to [0, 1]
    (rows = BPCs, cols = stim pairs)."""
    H = np.asarray(H)
    n_components, n_pairs = H.shape
    pairs = ([str(i) for i in range(n_pairs)] if pairs is None
             else [str(p) for p in np.asarray(pairs)])
    fig = go.Figure(go.Heatmap(
        z=np.clip(H, 0, 1), x=pairs, y=[f"BPC {i}" for i in range(n_components)],
        colorscale="Magma", zmin=0, zmax=1, colorbar=dict(title="H"),
        hovertemplate="%{y}<br>%{x}<br>H = %{z:.3f}<extra></extra>"))
    fig.update_layout(
        title=title or "NMF weights H (clipped to [0, 1])",
        xaxis=dict(title="stimulation pair", tickfont=dict(size=9)),
        template="plotly_white", height=120 + 60 * n_components)
    if show:
        fig.show()
    return fig