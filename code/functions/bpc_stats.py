"""
bpc_stats.py
============
Per-stim-pair statistics for a given basis set of BPCs.

For each BPC k and each stim pair assigned to k, we compute:
    alphas[trial]    : alpha coefficient of the trial onto the basis curve
    epsilon2s[trial] : sum-squared residual energy
    V2s[trial]       : sum-squared trial energy
    errxprojs[pair]  : t-stat of within-pair off-diagonal residual projections
    plotweights[pair]: SNR = mean(alpha / sqrt(epsilon2)) across trials
    p_vals[pair]     : one-sample t-test (snr vs. 0) p-value
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from functions.colors import bpc_color


def curve_stats(
    V: np.ndarray,
    Bs: np.ndarray,
    stim_sites: np.ndarray,
    pairs: np.ndarray,
    bpc_pairs: np.ndarray,
) -> dict:
    """Compute alpha / residual / SNR statistics for each (BPC, stim pair).

    Parameters
    ----------
    V : (n_trials, n_times)
    Bs : (n_components, n_times) basis curves.
    stim_sites : (n_trials,)
    pairs : (n_pairs,) ordered unique pair labels.
    bpc_pairs : (n_pairs,) float array; bpc index per pair (NaN = excluded).

    Returns
    -------
    dict with arrays:
        alphas      (n_trials,)
        epsilon2s   (n_trials,)
        V2s         (n_trials,)
        errxprojs   (n_pairs,)
        plotweights (n_pairs,)
        p_vals      (n_pairs,)
    """
    V = np.asarray(V)
    stim_sites = np.asarray(stim_sites)
    pairs = np.asarray(pairs)
    bpc_pairs = np.asarray(bpc_pairs)
    n_components = Bs.shape[0]

    alphas      = np.full(len(stim_sites), np.nan)
    epsilon2s   = np.full(len(stim_sites), np.nan)
    V2s         = np.full(len(stim_sites), np.nan)
    errxprojs   = np.full(len(pairs), np.nan)
    p_vals      = np.full(len(pairs), np.nan)
    plotweights = np.full(len(pairs), np.nan)

    V_selfproj = V @ V.T

    for k in range(n_components):
        bpc_alphas    = Bs[k] @ V.T
        bpc_epsilon2  = V - (Bs[k][:, None] @ bpc_alphas[None]).T
        errxproj      = bpc_epsilon2 @ bpc_epsilon2.T

        for pair_idx in np.where(bpc_pairs == k)[0]:
            trials = stim_sites == pairs[pair_idx]
            alphas[trials]    = bpc_alphas[trials]
            a = errxproj[np.ix_(trials, trials)]
            epsilon2s[trials] = np.diag(a)
            V2s[trials]       = np.diag(V_selfproj[np.ix_(trials, trials)])

            b = np.concatenate([a[np.tril_indices(a.shape[0], k=-1)],
                                a[np.triu_indices(a.shape[0], k=1)]])
            if len(b) > 1 and np.std(b, ddof=1) > 0:
                errxprojs[pair_idx] = np.mean(b) * np.sqrt(len(b)) / np.std(b, ddof=1)

            snr = alphas[trials] / np.sqrt(epsilon2s[trials])
            plotweights[pair_idx] = np.mean(snr)
            if len(snr) > 1:
                p_vals[pair_idx] = stats.ttest_1samp(snr, 0).pvalue

    return {
        "alphas":      alphas,
        "epsilon2s":   epsilon2s,
        "V2s":         V2s,
        "errxprojs":   errxprojs,
        "plotweights": plotweights,
        "p_vals":      p_vals,
    }


def plot_plotweights(
    stats,
    bpcs,
    pairs,
    contact: str | None = None,
    ylim_top: float | None = 5,
    p_vals=None,
    show: bool = False,
) -> go.Figure:
    """Per-stim-pair SNR bar chart, grouped & colored by BPC (Plotly).

    Pairs are ordered by (BPC index, then high-to-low SNR within each BPC);
    excluded pairs (bpc_pairs == NaN) are dropped.

    Parameters
    ----------
    stats : dict from ``curve_stats(...)`` (uses 'plotweights', and 'p_vals'
        for hover) OR a (n_pairs,) SNR array.
    bpcs : dict from ``identify_bpcs(...)`` (uses 'bpc_pairs') OR a (n_pairs,)
        bpc_pairs array.
    pairs : (n_pairs,) ordered pair labels.
    contact : optional recording-electrode name for the title.
    ylim_top : cap the y-axis top (default 5; ``None`` = autoscale).
    """
    if isinstance(stats, dict):
        plotweights = np.asarray(stats["plotweights"])
        if p_vals is None:
            p_vals = stats.get("p_vals")
    else:
        plotweights = np.asarray(stats)
    bpc_pairs = np.asarray(bpcs["bpc_pairs"] if isinstance(bpcs, dict) else bpcs)
    pairs = np.asarray(pairs)
    p_vals = None if p_vals is None else np.asarray(p_vals)

    keep = ~np.isnan(bpc_pairs)
    idx  = np.where(keep)[0]
    if len(idx) == 0:
        raise ValueError("No pairs assigned to any BPC (all bpc_pairs are NaN).")
    idx = idx[np.lexsort((-np.nan_to_num(plotweights[idx]), bpc_pairs[idx]))]

    fig = go.Figure()
    for k in sorted(set(int(b) for b in bpc_pairs[keep])):
        sel = [i for i in idx if int(bpc_pairs[i]) == k]
        if not sel:
            continue
        cdata = ([p_vals[i] for i in sel] if p_vals is not None
                 else [np.nan] * len(sel))
        fig.add_trace(go.Bar(
            x=[str(pairs[i]) for i in sel], y=[plotweights[i] for i in sel],
            marker_color=bpc_color(k), name=f"BPC {k}",
            customdata=cdata,
            hovertemplate=("%{x}<br>SNR %{y:.3f}<br>p = %{customdata:.2g}"
                           "<extra>BPC " + str(k) + "</extra>")))
    fig.add_hline(y=0, line=dict(color="gray", width=1))
    title = "BPC SNR" + (f" — {contact}" if contact is not None else "")
    fig.update_layout(
        title=title, barmode="group",
        xaxis=dict(title="stimulation pair", tickfont=dict(size=8)),
        yaxis=dict(title="SNR", range=[None, ylim_top] if ylim_top is not None else None),
        template="plotly_white", height=440, legend=dict(title="BPC"))
    if show:
        fig.show()
    return fig