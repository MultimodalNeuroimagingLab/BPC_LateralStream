"""
spectral_common.py
==================
Shared plumbing for the consensus-BPC **spectral** (wavelet spectrogram) and
**broadband** analyses (:mod:`functions.spectrogram`, :mod:`functions.broadband`).

Three jobs, kept in one place so both analyses agree:

1. :func:`load_contact_trials` — pull the RAW, artifact-containing evoked trials for
   one recording electrode on a WIDE window (default ``[-1.5, 1.5] s``), grouped by
   stim pair. This is deliberately NOT the 3-500 ms BPC window: the wavelet / IIR
   steps need the pre-stim baseline (-500 to -100 ms) and the stimulation artifact
   around ``t=0`` (which is then removed by
   :func:`functions.artifact_removal.remove_stim_artifact`). Reuses
   :func:`load_preproc.build_convergent_matrix` so run-merging + Huang current
   resolution match exactly what the saved BPC outputs were built from.

2. :func:`iter_connections` — walk a clustering ``res`` (from
   :func:`bpc_group.group_bpc_analysis_kmeans` etc.) and yield every
   ``(patient, recording contact, stim pair)`` **connection** tagged with its
   consensus-BPC cluster (or ``None`` when the connection is *unclustered* — the
   "algorithmically excluded from BPC assignment" negative-control set). This is the
   exact same bookkeeping the glass brain uses in
   :func:`cluster_bpc.pooled_bpc_clustering` (``bpc_pairs`` -> ``row_map`` ->
   ``labels``), so the spectral groups line up 1:1 with every other ``res`` view.

3. :func:`bpc_overlay_curves` — the consensus BPC voltage-domain shapes
   (``res['centroids_plot']``) on ``res['times']``, for overlaying in black on top
   of the group spectrogram / broadband plots.

"""

from __future__ import annotations

import numpy as np
from scipy.stats import false_discovery_control
from scipy.stats import t as _student_t

from functions.load_preproc import build_convergent_matrix


# --------------------------------------------------------------------------- #
# 1. Raw wide-window trials for one recording electrode, grouped by stim pair
# --------------------------------------------------------------------------- #
def load_contact_trials(
    patient: str,
    contact: str,
    *,
    compute_window: tuple[float, float] = (-1.5, 1.5),
    dirname: str | None = None,
    require_use_channel: bool = True,
) -> dict:
    """Raw evoked trials for ``contact`` on a wide window, split by stim pair.

    Wraps :func:`load_preproc.build_convergent_matrix` with ``stim_hemi=None`` (keep
    both hemispheres; the caller filters to whichever pairs the ``res`` actually
    holds) and ``resolve_current=True`` (the pipeline default), just widening the
    time window so the baseline and the stimulation artifact are included.

    Parameters
    ----------
    compute_window : ``(tmin, tmax)`` seconds. Wide on purpose so the CWT / Hilbert
        edge effects fall OUTSIDE the display + baseline windows. Default
        ``[-1.5, 1.5] s``.
    dirname : preproc derivatives folder override (defaults to the module default,
        ``bsep_basic_analysis`` — what the BPC outputs were built from).
    require_use_channel : keep only stim pairs where ``contact`` passed the pipeline's
        ``use_channels`` filter (default True, matching the saved BPC outputs).

    Returns
    -------
    dict with:
        ``V``          (n_trials, n_time) float64 — all trials for this contact,
        ``times``      (n_time,) seconds,
        ``stim_sites`` (n_trials,) stim-pair label per trial,
        ``pairs``      sorted unique stim-pair labels present,
        ``srate``      sampling rate (Hz) inferred from ``times``,
        ``by_pair``    ``{pair: (n_pair_trials, n_time) ndarray}`` view (no copy).
    """
    tmin, tmax = compute_window
    bpc = build_convergent_matrix(
        patient, contact, bpc_tmin=tmin, bpc_tmax=tmax,
        stim_hemi=None, resolve_current=True,
        require_use_channel=require_use_channel, dirname=dirname)
    V = np.asarray(bpc["V"], float)
    times = np.asarray(bpc["times"], float).ravel()
    stim_sites = np.asarray(bpc["stim_sites"])
    dt = float(np.median(np.diff(times)))
    srate = 1.0 / dt if dt > 0 else np.nan
    by_pair = {str(p): V[stim_sites == p] for p in bpc["pairs"]}
    return dict(V=V, times=times, stim_sites=stim_sites,
                pairs=np.asarray(bpc["pairs"]), srate=srate, by_pair=by_pair)


# --------------------------------------------------------------------------- #
# 2. Enumerate every (recording, stim) connection tagged with its consensus BPC
# --------------------------------------------------------------------------- #
def iter_connections(res: dict):
    """Yield one record per ``(patient, recording contact, stim pair)`` connection.

    Mirrors the clustered / unclustered bookkeeping in
    :func:`cluster_bpc.pooled_bpc_clustering` exactly: a connection is *clustered*
    when its pair has a finite local BPC index (``bpc_pairs[i]``) whose curve
    survived pooling (``row_map[b] >= 0``); its consensus cluster is then
    ``labels[row_map[b]]``. Otherwise it is *unclustered* — no BPC assigned, or the
    BPC was dropped pre-clustering by the SNR window / degeneracy — and ``cluster``
    is ``None`` (these form the negative-control set).

    Yields
    ------
    dict with ``ci`` (index into ``res['contacts_info']``), ``patient``, ``contact``,
    ``pair``, ``pair_idx`` (column into that electrode's ``pairs``/``bpc_pairs``),
    ``cluster`` (int consensus-BPC id or ``None``), and ``snr`` (the saved per-pair
    mean SNR / plotweight, or NaN).
    """
    labels = np.asarray(res["labels"])
    pw_list = res.get("pw_list")
    for ci, (patient, contact, pairs, bpc_pairs, row_map) in enumerate(res["contacts_info"]):
        pairs = np.asarray(pairs)
        bpc_pairs = np.asarray(bpc_pairs, float)
        row_map = np.asarray(row_map, int)
        pw = np.asarray(pw_list[ci], float) if pw_list is not None else None
        for i, pair in enumerate(pairs):
            b = bpc_pairs[i]
            clustered = (np.isfinite(b) and 0 <= int(b) < len(row_map)
                         and row_map[int(b)] >= 0)
            cluster = int(labels[row_map[int(b)]]) if clustered else None
            snr = float(pw[i]) if (pw is not None and i < len(pw)) else np.nan
            yield dict(ci=ci, patient=str(patient), contact=str(contact),
                       pair=str(pair), pair_idx=i, cluster=cluster, snr=snr)


def connections_by_cluster(res: dict, *, control: bool = True) -> dict:
    """Group :func:`iter_connections` records by consensus cluster.

    Returns ``{k: [record, ...]}`` for each consensus BPC ``k`` in
    ``0..n_clusters-1`` (always present, even if empty), plus — when ``control`` —
    the key ``None`` holding every unclustered connection (the negative control).
    """
    n_clusters = int(res.get("n_clusters", 0))
    out: dict = {k: [] for k in range(n_clusters)}
    if control:
        out[None] = []
    for rec in iter_connections(res):
        k = rec["cluster"]
        if k is None:
            if control:
                out[None].append(rec)
        else:
            out.setdefault(k, []).append(rec)
    return out


# --------------------------------------------------------------------------- #
# 3. Consensus BPC waveforms, for the black overlay on the group plots
# --------------------------------------------------------------------------- #
def bpc_overlay_curves(res: dict) -> dict:
    """``{k: (times, curve)}`` — the voltage-domain consensus BPC shapes.

    ``curve`` is ``res['centroids_plot'][k]`` (the un-decayed, plotted centroid, same
    as ``res['fig_curves']``) on ``res['times']`` (the 3-500 ms BPC grid). Meant to be
    drawn in black on top of the group spectrogram / broadband figures, which live on
    a wider raw-signal time axis but share the same real-time seconds.
    """
    t = np.asarray(res["times"], float).ravel()
    cents = np.asarray(res["centroids_plot"], float)
    return {int(k): (t, cents[k]) for k in range(int(res.get("n_clusters", len(cents))))}


# --------------------------------------------------------------------------- #
# 4. One-sample t 
# --------------------------------------------------------------------------- #
# Both spectral analyses answer the same statistical question — "which
# time-frequency bins / time samples differ from rest?" — as a one-sample t test
# vs 0 across the sites of a consensus BPC (each site's map/trace is ALREADY a
# fold-change / baseline-subtraction vs its own pre-stim rest, so 0 == rest). The
# resulting t's are then thresholded at a 5% false-discovery rate controlled under
# dependency by Benjamini & Yekutieli (2001) — chosen because nearby bins/samples
# are positively or negatively regression-dependent. These helpers live here so the
# spectrogram and broadband modules run an identical test.
def one_sample_t(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """One-sample t statistic vs 0 along ``axis``.

    ``stack`` holds the per-site observations (e.g. ``(n_sites, F, T)`` for a
    spectrogram or ``(n_sites, T)`` for a broadband trace); the t is taken across
    ``axis`` (default 0, the site axis) and that axis is removed. Bins with fewer
    than 2 sites or zero across-site variance return NaN (untestable) — so every
    finite output bin shares the same ``df = n_sites - 1`` and NaN propagates (no
    partial-n bins), which :func:`fdr_significant` relies on.
    """
    stack = np.asarray(stack, float)
    n = stack.shape[axis]
    out_shape = stack.shape[:axis] + stack.shape[axis + 1:]
    if n < 2:
        return np.full(out_shape, np.nan)
    mean = stack.mean(axis)
    sd = stack.std(axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (sd / np.sqrt(n))
    return np.where(sd > 0, t, np.nan)


def fdr_significant(
    tstat: np.ndarray,
    df: int,
    *,
    q: float = 0.05,
    method: str = "by",
    window_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean "significantly different from rest" mask for a one-sample-t map.

    Converts each finite one-sample ``tstat`` (two-sided, on ``df`` degrees of
    freedom) to a p-value and rejects at a ``q`` false-discovery rate, controlled
    **under dependency** by the Benjamini-Yekutieli (2001) procedure
    (``method='by'``; ``'bh'`` for the independent Benjamini-Hochberg variant) via
    :func:`scipy.stats.false_discovery_control`.

    The multiple-comparison **family is exactly the finite bins passed in one call**,
    so calling this once per consensus BPC panel corrects each panel over its own
    bins. ``window_mask`` (same shape as ``tstat``) optionally narrows the family to
    its True entries (e.g. a post-stim window); non-finite / out-of-family bins are
    never flagged.

    Parameters
    ----------
    tstat : one-sample t statistics vs 0 (any shape, e.g. ``(F, T)`` or ``(T,)``).
    df : degrees of freedom for those t's (``n_sites - 1``; a single scalar because
        untestable bins are NaN — see :func:`one_sample_t`).
    q : false-discovery rate (default 0.05).
    method : ``'by'`` (Benjamini-Yekutieli, under dependency; default) or ``'bh'``.

    Returns
    -------
    ndarray[bool] of ``tstat.shape``: True where the bin is significant at FDR ``q``.
    """
    tstat = np.asarray(tstat, float)
    out = np.zeros(tstat.shape, bool)
    if df is None or int(df) < 1:
        return out
    fam = np.isfinite(tstat)
    if window_mask is not None:
        fam &= np.asarray(window_mask, bool)
    if not fam.any():
        return out
    p = 2.0 * _student_t.sf(np.abs(tstat[fam]), int(df))        # two-sided vs 0
    adj = false_discovery_control(np.clip(p, 0.0, 1.0), method=method)
    out[fam] = adj <= q
    return out


def fdr_significant_pooled(
    tstats: dict,
    dfs: dict,
    *,
    q: float = 0.05,
    method: str = "by",
    window_mask: np.ndarray | None = None,
) -> dict:
    """Pooled "significantly different from rest" masks across several panels.

    Like :func:`fdr_significant`, but the multiple-comparison **family is the union of
    every panel's finite bins**: the two-sided p-values of all panels (each bin on its
    own panel's ``df``) are concatenated, a single Benjamini-Yekutieli correction is
    applied across the whole set, then rejections are scattered back per panel. This
    matches Huang et al. ``bpcSpectraBB.m`` (``fdr_bh`` over all consensus BPCs + the
    excluded group at once).

    Parameters
    ----------
    tstats : ``{key: ndarray}`` one-sample t maps (``(F, T)`` or ``(T,)``), per panel.
    dfs : ``{key: int}`` degrees of freedom (``n_sites - 1``) per panel.
    window_mask : optional boolean (broadcast to each map) restricting the family to a
        time window; non-finite / out-of-window / df<1 bins never enter the family.

    Returns
    -------
    ``{key: ndarray[bool]}`` with each panel's shape, True where significant at the
    pooled FDR ``q``.
    """
    out = {k: np.zeros(np.asarray(t, float).shape, bool) for k, t in tstats.items()}
    parts, index = [], []                       # p-values + (key, shape, flat-idx) per panel
    for k, t in tstats.items():
        t = np.asarray(t, float)
        df = dfs.get(k)
        fam = np.isfinite(t)
        if window_mask is not None:
            wm = (np.broadcast_to(np.asarray(window_mask, bool), t.shape)
                  if t.ndim == 2 else np.asarray(window_mask, bool))
            fam &= wm
        if df is None or int(df) < 1 or not fam.any():
            continue
        flat = np.where(fam.ravel())[0]
        parts.append(2.0 * _student_t.sf(np.abs(t.ravel()[flat]), int(df)))
        index.append((k, t.shape, flat))
    if not parts:
        return out
    adj = false_discovery_control(np.clip(np.concatenate(parts), 0.0, 1.0), method=method)
    off = 0
    for k, shape, flat in index:
        m = np.zeros(int(np.prod(shape)), bool)
        m[flat] = adj[off:off + flat.size] <= q
        out[k] = m.reshape(shape)
        off += flat.size
    return out


def _window_col(times: np.ndarray, window) -> np.ndarray | None:
    """Boolean 1-D column mask over ``times`` for a ``(lo, hi)`` seconds ``window``
    (either bound may be ``None`` for an open end); ``None`` window -> ``None``."""
    if window is None:
        return None
    times = np.asarray(times, float)
    lo, hi = window
    col = np.ones(times.shape, bool)
    if lo is not None:
        col &= times >= lo
    if hi is not None:
        col &= times < hi
    return col


def panel_masks(group: dict, fdr_q: float | None = None) -> dict:
    """Per-consensus-BPC significance masks for a spectral ``group`` dict.

    With ``fdr_q is None`` (default) returns the masks already stored by
    :func:`spectrogram.group_bpc_spectrograms` / :func:`broadband.group_bpc_broadband`
    (``group['sig']``). Pass a float to **re-threshold** at a different FDR ``q``
    without recomputing any transforms — recomputed from the stored one-sample
    ``group['tstat']`` (``(F, T)`` spectrogram or ``(T,)`` broadband), ``n_sites``,
    and the family window / method recorded in ``group['fdr']``. The correction is
    **pooled across all panels** (:func:`fdr_significant_pooled`), matching the stored
    masks.
    """
    if fdr_q is None:
        return dict(group.get("sig", {}) or {})
    tstat = group.get("tstat", {}) or {}
    n_sites = group.get("n_sites", {}) or {}
    meta = group.get("fdr", {}) or {}
    method = meta.get("method", "by")
    wcol = _window_col(group.get("times", []), meta.get("window"))
    dfs = {k: int(n_sites.get(k, 0)) - 1 for k in tstat}
    return fdr_significant_pooled(tstat, dfs, q=fdr_q, method=method, window_mask=wcol)
