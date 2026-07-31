"""
broadband.py
============
Time-varying **broadband (70-170 Hz) power** by consensus BPC category — the
companion to :mod:`functions.spectrogram`.

Pipeline:

  per connection = one (recording electrode x stim pair):
    1. raw wide-window trials, stim artifact removed (+/-8 ms,
       :func:`functions.artifact_removal.remove_stim_artifact`);
    2. zero-phase band-pass 70-170 Hz (3rd-order Butterworth, ``filtfilt``);
    3. broadband estimate = log10( |Hilbert(.)|^2 ) (log power);
    4. subtract each trial's mean broadband over baseline (-500..-100 ms);
    5. average across trials -> the connection's broadband time course.
    -> :func:`connection_broadband` / :func:`compute_contact_broadband` (cached to
       ``outputs/sub-<p>/<contact>/broadband.npz`` labeled by stim pair).

  per consensus BPC:
    average the connection broadband traces of every stim site assigned to it ->
    one time-varying broadband estimate (mean +/- SEM across sites).
    -> :func:`group_bpc_broadband` (takes a clustering ``res``); the same for all
       *unclustered* connections as a negative control.

  plotting:
    -> :func:`plot_group_broadband` — one panel per consensus BPC (+ control),
       broadband mean +/- SEM colored by cluster, consensus BPC waveform in black.

SNR < 1 sites are already excluded upstream by the group analysis's ``snr_window``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt, hilbert

import functions.bpc_group as bg
from functions.artifact_removal import remove_stim_artifact
from functions.save_outputs import contact_dir
from functions.spectral_common import (bpc_overlay_curves, connections_by_cluster,
                                        fdr_significant_pooled, load_contact_trials,
                                        one_sample_t, panel_masks)

CACHE_NAME = "broadband.npz"


# --------------------------------------------------------------------------- #
# One connection (recording electrode x stim pair)
# --------------------------------------------------------------------------- #
def connection_broadband(
    patient: str,
    contact: str,
    pair: str,
    *,
    trials: np.ndarray | None = None,
    times: np.ndarray | None = None,
    srate: float | None = None,
    band: tuple[float, float] = (70.0, 170.0),
    order: int = 3,
    baseline: tuple[float, float] = (-0.5, -0.1),
    half_ms: float = 8.0,
    win_ms: float | None = None,
    compute_window: tuple[float, float] = (-1.5, 1.5),
    out_window: tuple[float, float] = (-0.1, 0.5),
    decimate: int = 5,
    dirname: str | None = None,
) -> dict:
    """Trial-averaged broadband time course for one stim pair recorded at ``contact``.

    Artifact removal -> zero-phase Butterworth band-pass -> log10 Hilbert power ->
    per-trial baseline subtraction -> mean across trials.

    Parameters
    ----------
    trials, times, srate : preloaded wide-window trials for this pair (pass when
        looping many pairs of one contact); otherwise loaded via
        :func:`spectral_common.load_contact_trials`.
    band : band-pass edges in Hz (default 70-170).
    order : Butterworth order (default 3; ``filtfilt`` doubles the effective order).
    baseline : ``(lo, hi)`` s subtracted per trial (default -500..-100 ms).
    half_ms, win_ms, compute_window, out_window, decimate : as in
        :func:`spectrogram.connection_spectrogram`.

    Returns
    -------
    dict: ``bb`` ``(n_time_out,)`` broadband log-power (baseline-subtracted),
    ``times`` (s), ``n_trials``, ``patient``, ``contact``, ``pair``.
    """
    if trials is None or times is None or srate is None:
        rec = load_contact_trials(patient, contact,
                                  compute_window=compute_window, dirname=dirname)
        trials = rec["by_pair"].get(str(pair))
        if trials is None:
            raise ValueError(f"stim pair {pair!r} not found for {patient}/{contact}")
        times, srate = rec["times"], rec["srate"]
    trials = np.asarray(trials, float)
    times = np.asarray(times, float).ravel()
    if trials.ndim == 1:
        trials = trials[None]
    if trials.shape[0] == 0:
        raise ValueError(f"no trials for {patient}/{contact} pair {pair!r}")

    cleaned = remove_stim_artifact(trials, times, half_ms=half_ms, win_ms=win_ms)
    b, a = butter(order, [band[0], band[1]], btype="band", fs=srate)
    filt = filtfilt(b, a, cleaned, axis=-1)
    bb = np.log10(np.abs(hilbert(filt, axis=-1)) ** 2)         # (n_trials, n_time)

    bmask = (times >= baseline[0]) & (times < baseline[1])
    if not bmask.any():
        raise ValueError(f"empty baseline window {baseline} within {compute_window}")
    bb = bb - bb[:, bmask].mean(axis=1, keepdims=True)         # per-trial baseline
    site = bb.mean(axis=0)                                     # mean across trials

    keep = (times >= out_window[0]) & (times <= out_window[1])
    idx = np.where(keep)[0][:: max(1, int(decimate))]
    return dict(bb=site[idx].astype(np.float32), times=times[idx],
                n_trials=int(trials.shape[0]),
                patient=str(patient), contact=str(contact), pair=str(pair))


# --------------------------------------------------------------------------- #
# All stim pairs of one recording electrode, cached to disk
# --------------------------------------------------------------------------- #
def _params(band, order, baseline, half_ms, win_ms, out_window, decimate) -> dict:
    return dict(band=list(band), order=int(order), baseline=list(baseline),
                half_ms=half_ms, win_ms=win_ms, out_window=list(out_window),
                decimate=int(decimate))


def compute_contact_broadband(
    patient: str,
    contact: str,
    pairs: list[str] | None = None,
    *,
    recompute: bool = False,
    save: bool = True,
    verbose: bool = True,
    root: Path | None = None,
    compute_window: tuple[float, float] = (-1.5, 1.5),
    dirname: str | None = None,
    patient_label: str | None = None,
    **bb_kw,
) -> dict:
    """Compute (and cache) the broadband time course of every stim pair of one
    recording electrode.

    Saves ``broadband.npz`` under ``outputs/sub-<patient>/<contact>/`` holding ``bb``
    ``(n_pairs, n_time)`` plus ``pairs`` / ``times`` / ``n_trials`` and the parameter
    stamp; reruns with matching parameters reload, and only missing pairs are
    recomputed.

    Returns
    -------
    dict: ``by_pair`` ``{pair: (n_time,) float32}``, ``times``, ``n_trials``
    ``{pair: int}``, ``path``.
    """
    band = bb_kw.get("band", (70.0, 170.0))
    order = bb_kw.get("order", 3)
    baseline = bb_kw.get("baseline", (-0.5, -0.1))
    half_ms = bb_kw.get("half_ms", 8.0)
    win_ms = bb_kw.get("win_ms", None)
    out_window = bb_kw.get("out_window", (-0.1, 0.5))
    decimate = bb_kw.get("decimate", 5)
    params = _params(band, order, baseline, half_ms, win_ms, out_window, decimate)

    path = contact_dir(patient, contact, root) / CACHE_NAME
    cached, ctimes = _load_bb_cache(path, params) if not recompute else ({}, None)

    # Fast path: the caller named the pairs and the cache already holds all of them
    # -> return straight from cache WITHOUT loading the raw trials (the slow part is
    # load_contact_trials / build_convergent_matrix re-reading the raw preproc .mats).
    if pairs is not None and cached:
        want = [str(p) for p in pairs]
        if want and all(p in cached for p in want):
            if verbose:
                print(f"  [{patient_label or patient}/{contact}] {len(want)} cached, "
                      f"0 to compute")
            return dict(by_pair={p: cached[p]["bb"] for p in want},
                        times=np.asarray(ctimes, float),
                        n_trials={p: int(cached[p]["n_trials"]) for p in want},
                        path=path)

    rec = load_contact_trials(patient, contact,
                              compute_window=compute_window, dirname=dirname)
    want = [str(p) for p in (pairs if pairs is not None else rec["pairs"])]
    want = [p for p in want if p in rec["by_pair"]]

    by_pair: dict[str, np.ndarray] = {}
    n_trials: dict[str, int] = {}
    out_times = ctimes
    todo = [p for p in want if p not in cached]
    if verbose and cached:
        print(f"  [{patient_label or patient}/{contact}] {len(set(want) & set(cached))} "
              f"cached, {len(todo)} to compute")
    for p in want:
        if p in cached:
            by_pair[p] = cached[p]["bb"]
            n_trials[p] = int(cached[p]["n_trials"])
            continue
        cbb = connection_broadband(
            patient, contact, p, trials=rec["by_pair"][p],
            times=rec["times"], srate=rec["srate"], **bb_kw)
        by_pair[p] = cbb["bb"]
        n_trials[p] = cbb["n_trials"]
        out_times = cbb["times"] if out_times is None else out_times

    if save and by_pair:
        _save_bb_cache(path, by_pair, n_trials, out_times, params, cached)

    return dict(by_pair=by_pair, times=np.asarray(out_times, float),
                n_trials=n_trials, path=path)


def _load_bb_cache(path: Path, params: dict):
    if not path.exists():
        return {}, None
    try:
        z = np.load(path, allow_pickle=True)
        if json.loads(str(z["params"])) != params:
            return {}, None
        pairs = [str(p) for p in z["pairs"]]
        bb = z["bb"]
        nt = z["n_trials"]
        out = {p: dict(bb=bb[i], n_trials=int(nt[i])) for i, p in enumerate(pairs)}
        return out, np.asarray(z["times"], float)
    except Exception:
        return {}, None


def _save_bb_cache(path, by_pair, n_trials, times, params, prev):
    merged = {p: (by_pair[p], n_trials[p]) for p in by_pair}
    for p, d in prev.items():
        merged.setdefault(p, (d["bb"], int(d["n_trials"])))
    pairs = sorted(merged)
    bb = np.stack([merged[p][0] for p in pairs]).astype(np.float32)
    nt = np.array([merged[p][1] for p in pairs], int)
    np.savez_compressed(path, bb=bb, pairs=np.asarray(pairs),
                        times=np.asarray(times, float), n_trials=nt,
                        params=json.dumps(params))


# --------------------------------------------------------------------------- #
# Group: mean +/- SEM broadband per consensus BPC (+ control)
# --------------------------------------------------------------------------- #
def group_bpc_broadband(
    res: dict,
    *,
    recompute: bool = False,
    control: bool = True,
    verbose: bool = True,
    fdr_q: float = 0.05,
    fdr_method: str = "by",
    sig_window: tuple[float | None, float | None] | None = None,
    anonymize: bool = False,
    **bb_kw,
) -> dict:
    """Time-varying broadband estimate for each consensus BPC in ``res``.

    Averages the per-connection broadband traces (one Hilbert pass per recording
    contact, cached by :func:`compute_contact_broadband`) across all stim sites of a
    consensus BPC, giving mean +/- SEM across sites. With ``control`` the unclustered
    connections are aggregated the same way as a negative control (key ``None``).

    Also runs, per consensus BPC, a one-sample t test vs 0 at each time sample
    (``mean/SEM``; each trace is already baseline-subtracted so 0 == rest) and marks
    the samples that differ from rest at a ``fdr_q`` false-discovery rate. The
    correction **family is POOLED across every consensus BPC + the control at once**
    (matching Huang et al. ``bpcSpectraBB.m``); ``fdr_method='by'`` is
    Benjamini-Yekutieli (2001), valid under dependency (``'bh'`` for independent
    Benjamini-Hochberg). ``sig_window=(lo, hi)`` s restricts the family/test (either
    bound ``None`` = open); the default ``None`` tests the whole returned window.

    Returns
    -------
    dict: ``mean`` ``{k: (T,)}``, ``sem`` ``{k: (T,)}``, ``tstat`` ``{k: (T,)}``
    one-sample t vs rest, ``sig`` ``{k: (T,) bool}`` FDR-significant samples,
    ``n_sites`` ``{k: int}``, ``times`` (T,), ``bpc_curves`` ``{k: (times, curve)}``,
    ``fdr`` ``{q, method, window}``, ``n_clusters``, ``ylim`` suggested shared
    y-range.
    """
    groups = connections_by_cluster(res, control=control)
    need: dict[tuple, set] = defaultdict(set)
    for recs in groups.values():
        for r in recs:
            need[(r["patient"], r["contact"])].add(r["pair"])

    # anonymize=True -> progress prints show neutral S1, S2, ... instead of subject ids
    alias = {p: f"S{i + 1}" for i, p in
             enumerate(sorted({pt for pt, _ in need}))} if anonymize else {}
    bb_lookup: dict[tuple, np.ndarray] = {}
    out_times = None
    for (patient, contact), want in sorted(need.items()):
        plabel = alias.get(patient) if anonymize else None
        if verbose:
            print(f"[broadband] {plabel or patient}/{contact}: {len(want)} pair(s)")
        cc = compute_contact_broadband(
            patient, contact, pairs=sorted(want), recompute=recompute,
            verbose=verbose, patient_label=plabel, **bb_kw)
        for p, s in cc["by_pair"].items():
            bb_lookup[(patient, contact, p)] = s
        if out_times is None:
            out_times = cc["times"]
        elif not np.allclose(out_times, cc["times"]):
            raise ValueError(
                f"{patient}/{contact} produced a different broadband time grid — all "
                "connections must share one grid (same srate / out_window / decimate).")

    mean, sem, tstat, n_sites = {}, {}, {}, {}
    for k, recs in groups.items():
        traces = [bb_lookup[(r["patient"], r["contact"], r["pair"])]
                  for r in recs
                  if (r["patient"], r["contact"], r["pair"]) in bb_lookup]
        n_sites[k] = len(traces)
        if traces:
            stack = np.stack(traces).astype(float)             # (n_sites, T)
            mean[k] = stack.mean(0)
            sem[k] = (stack.std(0, ddof=1) / np.sqrt(stack.shape[0])
                      if stack.shape[0] > 1 else np.zeros(stack.shape[1]))
            tstat[k] = one_sample_t(stack, axis=0)             # == mean/SEM, NaN if n<2
        else:
            T = len(out_times) if out_times is not None else 0
            mean[k] = np.full(T, np.nan)
            sem[k] = np.full(T, np.nan)
            tstat[k] = np.full(T, np.nan)
        if verbose:
            tag = "control (unclustered)" if k is None else f"consensus BPC {k}"
            print(f"  {tag}: {n_sites[k]} site(s)")

    # ---- FDR-controlled significance vs rest, POOLED across all panels + control
    #      (Huang et al. bpcSpectraBB.m: fdr_bh over all consensus BPCs + excluded) ----
    tt = np.asarray(out_times, float) if out_times is not None else np.array([])
    wcol = None
    if sig_window is not None and tt.size:
        lo, hi = sig_window
        wcol = np.ones(tt.shape, bool)
        if lo is not None:
            wcol &= tt >= lo
        if hi is not None:
            wcol &= tt < hi
    sig = fdr_significant_pooled(tstat, {k: int(n_sites.get(k, 0)) - 1 for k in tstat},
                                 q=fdr_q, method=fdr_method, window_mask=wcol)

    allv = np.concatenate([mean[k][np.isfinite(mean[k])] + s
                           for k in mean for s in (sem[k][np.isfinite(mean[k])],
                                                   -sem[k][np.isfinite(mean[k])])]) \
        if mean else np.array([])
    if allv.size:
        lo, hi = float(np.min(allv)), float(np.max(allv))
        pad = 0.08 * max(hi - lo, 1e-6)
        ylim = (lo - pad, hi + pad)
    else:
        ylim = (-1.0, 1.0)

    return dict(mean=mean, sem=sem, tstat=tstat, sig=sig, n_sites=n_sites,
                times=np.asarray(out_times, float),
                bpc_curves=bpc_overlay_curves(res),
                fdr=dict(q=float(fdr_q), method=str(fdr_method), window=sig_window),
                n_clusters=int(res.get("n_clusters", 0)), ylim=ylim)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _rgba(color: str, alpha: float) -> str:
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def plot_group_broadband(
    group: dict,
    *,
    xlim: tuple[float, float] = (0.0, 0.5),
    ylim: tuple[float, float] | None = None,
    include_control: bool = True,
    overlay_bpc: bool = True,
    overlay_color: str = "black",
    sem_band: bool = True,
    show_significance: bool = True,
    sig_up_color: str = "#e5241f",
    sig_down_color: str = "#1c6ef2",
    sig_width: float = 6.0,
    fdr_q: float | None = None,
    height: int = 360,
    width_per: int = 360,
    show: bool = True,
) -> go.Figure:
    """Plot each consensus BPC's broadband estimate (+ the control) with the consensus
    BPC waveform overlaid in black.

    One panel per consensus BPC (then the control), sharing a fixed y-range so the
    magnitudes are comparable. Broadband lines use the same tab10 cluster colors as
    every other ``res`` figure (:func:`cluster_bpc.cluster_color`); the control is
    grey. ``overlay_bpc`` draws ``group['bpc_curves']`` on a hidden secondary y-axis.

    Parameters
    ----------
    group : the dict from :func:`group_bpc_broadband`.
    ylim : shared broadband y-range; ``None`` uses ``group['ylim']``.
    sem_band : shade mean +/- SEM across sites.
    show_significance : draw, in a strip just above each panel, a bar over every run
        of time samples that differ from rest at the FDR from
        :func:`group_bpc_broadband` — **red where broadband increases** (mean > 0),
        **blue where it decreases** (mean < 0). Needs a ``group`` carrying
        ``'sig'``/``'tstat'``.
    sig_up_color, sig_down_color, sig_width : increase color (red), decrease color
        (blue), and bar thickness.
    fdr_q : ``None`` (default) uses ``group['sig']``; pass a float to re-threshold at
        a different FDR q (via :func:`spectral_common.panel_masks`, no recompute).
    """
    times = np.asarray(group["times"], float)
    lo, hi = ylim if ylim is not None else group["ylim"]

    keys = [k for k in range(group["n_clusters"]) if k in group["mean"]]
    if include_control and None in group["mean"]:
        keys.append(None)
    ncol = len(keys)

    masks = panel_masks(group, fdr_q) if show_significance else {}
    show_sig = bool(masks) and any(np.asarray(masks.get(k, ())).any() for k in keys)
    span = (hi - lo) or 1.0
    hi_plot = hi + 0.14 * span if show_sig else hi     # headroom for the sig. strip
    y_bar = hi + 0.07 * span
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0

    fig = make_subplots(
        rows=1, cols=ncol, shared_yaxes=True, horizontal_spacing=0.04,
        specs=[[{"secondary_y": True} for _ in range(ncol)]],
        subplot_titles=[("Excluded (control)" if k is None else f"Consensus BPC {k}")
                        + f"  n={group['n_sites'].get(k, 0)}" for k in keys])

    for c, k in enumerate(keys, start=1):
        color = "grey" if k is None else bg.cluster_color(k)
        m = np.asarray(group["mean"][k], float)
        s = np.asarray(group["sem"][k], float)
        if sem_band and np.isfinite(s).any():
            fig.add_trace(
                go.Scatter(x=np.concatenate([times, times[::-1]]),
                           y=np.concatenate([m + s, (m - s)[::-1]]),
                           fill="toself", fillcolor=_rgba(color, 0.20),
                           line=dict(width=0), showlegend=False, hoverinfo="skip"),
                row=1, col=c, secondary_y=False)
        fig.add_trace(
            go.Scatter(x=times, y=m, mode="lines", line=dict(color=color, width=2),
                       showlegend=False,
                       hovertemplate="t=%{x:.3f}s<br>broadband=%{y:.3f}<extra></extra>"),
            row=1, col=c, secondary_y=False)
        fig.add_hline(y=0, line=dict(color="gray", width=0.7), row=1, col=c)

        # significance strip: a bar over each run of samples differing from rest,
        # red where broadband increases (mean>0), blue where it decreases (mean<0)
        if show_significance:
            mask = np.asarray(masks.get(k, np.zeros(times.shape, bool)), bool)
            sgn = np.sign(m)
            i, n = 0, len(times)
            while i < n:
                if i < len(mask) and mask[i] and np.isfinite(sgn[i]) and sgn[i] != 0:
                    j = i
                    while (j + 1 < n and j + 1 < len(mask)
                           and mask[j + 1] and sgn[j + 1] == sgn[i]):
                        j += 1
                    up = sgn[i] > 0
                    lbl = "increase" if up else "decrease"
                    fig.add_trace(
                        go.Scatter(
                            x=[times[i] - dt / 2.0, times[j] + dt / 2.0],
                            y=[y_bar, y_bar], mode="lines",
                            line=dict(color=sig_up_color if up else sig_down_color,
                                      width=sig_width),
                            showlegend=False,
                            hovertemplate=f"significant {lbl} vs rest"
                                          "<br>%{x:.3f}s<extra></extra>"),
                        row=1, col=c, secondary_y=False)
                    i = j + 1
                else:
                    i += 1

        if overlay_bpc and k is not None and k in group["bpc_curves"]:
            bt, bc = group["bpc_curves"][k]
            fig.add_trace(
                go.Scatter(x=np.asarray(bt, float), y=np.asarray(bc, float),
                           mode="lines", line=dict(color=overlay_color, width=1.4),
                           showlegend=False, hoverinfo="skip"),
                row=1, col=c, secondary_y=True)
            amp = float(np.nanmax(np.abs(bc))) or 1.0
            fig.update_yaxes(range=[-amp * 1.05, amp * 1.05], showticklabels=False,
                             showgrid=False, zeroline=False,
                             row=1, col=c, secondary_y=True)

        fig.update_xaxes(range=list(xlim), title_text="time (s)" if c == 1 else None,
                         row=1, col=c)
        fig.update_yaxes(range=[lo, hi_plot], row=1, col=c, secondary_y=False,
                         title_text="broadband log-power (baseline-sub.)"
                         if c == 1 else None)

    q_used = fdr_q if fdr_q is not None else (group.get("fdr", {}) or {}).get("q", 0.05)
    title = "Broadband (70-170 Hz) estimate by consensus BPC"
    if show_sig:
        title += (f" — top bar: FDR q={q_used:.2g} (BY) sig. vs rest "
                  "(red=increase, blue=decrease)")
    fig.update_layout(
        title=title,
        width=max(width_per * ncol, 480), height=height,
        margin=dict(l=70, r=40, t=70, b=50))
    if show:
        fig.show()
    return fig
