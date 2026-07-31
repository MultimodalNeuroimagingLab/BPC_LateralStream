"""
spectrogram.py
==============
Induced (single-trial) **wavelet spectrograms** by consensus BPC category.

Pipeline (Python re-implementation of the paper's MATLAB ``cwt`` analysis):

  per connection = one (recording electrode x stim pair):
    1. pull raw wide-window trials, remove the stim artifact
       (:func:`functions.artifact_removal.remove_stim_artifact`, +/-8 ms);
    2. complex-Morlet CWT of each trial (PyWavelets ``cmor``), power = |coef|^2;
    3. dB baseline per frequency: log10 power minus its mean over the baseline
       (-500..-100 ms) window == divide by the geometric-mean baseline power
       (Huang et al. bpcSpectraBB.m). NOT log10(power / arithmetic-mean power),
       which carries a per-frequency Jensen offset (rest would not sit at 0);
    4. mean across trials in log space (== geometric mean of the fold-changes)
       -> the connection's log-fold-change spectrogram (one t-test sample).
    -> :func:`connection_spectrogram` (single pair) /
       :func:`compute_contact_spectrograms` (all pairs of a contact, cached to
       ``outputs/sub-<p>/<contact>/spectrograms.npz`` labeled by stim pair).

  per consensus BPC:
    collect the log10 spectrograms of every stim site assigned to it and take the
    one-sample t statistic vs 0 at each frequency-time bin.
    -> :func:`group_bpc_spectrograms` (takes a clustering ``res``). The same is done
       for all *unclustered* connections as a negative control.

  plotting:
    -> :func:`plot_group_spectrograms` — one heatmap per consensus BPC (+ control),
       a FIXED symmetric red<->blue (green/yellow-mid) t-scale shared across panels,
       with the consensus BPC waveform overlaid in black.

Stimulation sites with SNR < 1 are already dropped upstream by the group analysis's
``snr_window`` (so no extra filtering here, per the project convention).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pywt
from plotly.subplots import make_subplots

from functions.artifact_removal import remove_stim_artifact
from functions.save_outputs import contact_dir
from functions.spectral_common import (bpc_overlay_curves, connections_by_cluster,
                                        fdr_significant_pooled, load_contact_trials,
                                        one_sample_t, panel_masks)

# Default complex-Morlet wavelet: ``cmorB-C`` with bandwidth B=1.5, center C=1.0.
DEFAULT_WAVELET = "cmor1.5-1.0"
CACHE_NAME = "spectrograms.npz"


# --------------------------------------------------------------------------- #
# Frequency grid + CWT power
# --------------------------------------------------------------------------- #
def log_frequencies(fmin: float = 2.0, fmax: float = 200.0,
                    voices_per_octave: int = 10) -> np.ndarray:
    """Log-spaced frequency vector ``fmin..fmax`` at ``voices_per_octave`` per octave."""
    n_oct = np.log2(fmax / fmin)
    n = int(np.round(n_oct * voices_per_octave)) + 1
    return fmin * 2.0 ** (np.linspace(0.0, n_oct, n))


def cwt_power(trials: np.ndarray, srate: float, freqs: np.ndarray,
              wavelet: str = DEFAULT_WAVELET, method: str = "fft") -> np.ndarray:
    """``|CWT|^2`` power of ``trials`` at ``freqs``.

    ``trials`` is ``(..., n_time)``; the transform runs along the last axis, so a
    ``(n_trials, n_time)`` stack returns ``(n_freq, n_trials, n_time)``.

    ``method`` is passed to :func:`pywt.cwt`. The default is ``'fft'`` (FFT-based
    convolution): it is numerically identical to PyWavelets' default ``'conv'``
    (direct convolution) to ~1e-12 in power, but ORDERS OF MAGNITUDE faster here,
    because low-frequency scales produce very long wavelets and direct convolution
    over the wide compute window is pathologically slow (minutes/pair vs ~1 s/pair).
    """
    freqs = np.asarray(freqs, float)
    scales = pywt.frequency2scale(wavelet, freqs / srate)      # normalized freq = f/srate
    coef, _ = pywt.cwt(np.asarray(trials, float), scales, wavelet,
                       sampling_period=1.0 / srate, axis=-1, method=method)
    return np.abs(coef) ** 2


# --------------------------------------------------------------------------- #
# One connection (recording electrode x stim pair)
# --------------------------------------------------------------------------- #
def connection_spectrogram(
    patient: str,
    contact: str,
    pair: str,
    *,
    trials: np.ndarray | None = None,
    times: np.ndarray | None = None,
    srate: float | None = None,
    freqs: np.ndarray | None = None,
    wavelet: str = DEFAULT_WAVELET,
    baseline: tuple[float, float] = (-0.5, -0.1),
    baseline_per: str = "trial",
    half_ms: float = 8.0,
    win_ms: float | None = None,
    fmin: float = 2.0,
    fmax: float = 200.0,
    voices_per_octave: int = 10,
    compute_window: tuple[float, float] = (-1.5, 1.5),
    out_window: tuple[float, float] = (-0.1, 0.5),
    decimate: int = 5,
    dirname: str | None = None,
) -> dict:
    """Average log10 wavelet spectrogram for one stim pair recorded at ``contact``.

    Runs artifact removal -> per-trial CWT power -> per-frequency **dB baseline**
    (log10 power minus its mean over the baseline window == divide by the
    geometric-mean baseline power) -> mean across trials in log space. The returned
    ``spec`` is the single per-site observation that :func:`group_bpc_spectrograms`
    t-tests.

    Parameters
    ----------
    trials, times, srate : preloaded wide-window trials for THIS pair
        (``(n_trials, n_time)``), the time vector, and the sampling rate. If any is
        ``None`` they are loaded via :func:`spectral_common.load_contact_trials` and
        the ``pair`` selected — pass them in when looping many pairs of one contact to
        avoid reloading.
    baseline : ``(lo, hi)`` seconds for the per-frequency baseline (default
        -500..-100 ms).
    baseline_per : ``'trial'`` (default; each trial divided by its own baseline, then
        geometric-mean) or ``'site'`` (divide by the across-trial mean baseline).
    half_ms, win_ms : artifact gap half-width and adjacent-window width (see
        :func:`artifact_removal.remove_stim_artifact`).
    fmin, fmax, voices_per_octave : the log frequency grid (ignored if ``freqs`` given).
    compute_window : wide window the CWT runs on (edge effects fall outside output).
    out_window : window kept in the returned/saved spectrogram.
    decimate : integer time decimation of the output spectrogram (default 5 ->
        ~960 Hz at 4800 Hz srate; the power envelope is smooth so plain slicing is
        adequate).

    Returns
    -------
    dict: ``spec`` ``(n_freq, n_time_out)`` log10 fold-change, ``freqs``, ``times``
    (output, seconds), ``n_trials``, ``patient``, ``contact``, ``pair``.
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

    if freqs is None:
        freqs = log_frequencies(fmin, fmax, voices_per_octave)
    freqs = np.asarray(freqs, float)

    cleaned = remove_stim_artifact(trials, times, half_ms=half_ms, win_ms=win_ms)
    power = cwt_power(cleaned, srate, freqs, wavelet)          # (n_freq, n_trials, n_time)

    bmask = (times >= baseline[0]) & (times < baseline[1])
    if not bmask.any():
        raise ValueError(f"empty baseline window {baseline} within {compute_window}")

    # dB baseline: divide power by the GEOMETRIC mean over the baseline window, i.e.
    # subtract the mean of log10 power (Huang et al. bpcSpectraBB.m `./geomean(...)`).
    # An arithmetic-mean baseline (log10(power / mean(power))) carries a per-frequency
    # negative Jensen offset (single-sample power vs a windowed mean), largest at high
    # frequency, so "0" would not be rest and every bin tests significant vs 0.
    logpow = np.log10(np.where(power > 0, power, np.nan))      # (n_freq, n_trials, n_time)
    if baseline_per == "trial":                               # each trial / its own baseline
        base = np.nanmean(logpow[:, :, bmask], axis=2, keepdims=True)   # (n_freq, n_trials, 1)
    elif baseline_per == "site":                              # / across-trial baseline
        base = np.nanmean(logpow[:, :, bmask], axis=(1, 2))[:, None, None]
    else:
        raise ValueError("baseline_per must be 'trial' or 'site'")
    logfc = logpow - base                                     # (n_freq, n_trials, n_time)
    # geometric mean across trials == arithmetic mean in log space
    site = np.nanmean(logfc, axis=1)                          # (n_freq, n_time)

    keep = (times >= out_window[0]) & (times <= out_window[1])
    idx = np.where(keep)[0][:: max(1, int(decimate))]
    return dict(spec=site[:, idx].astype(np.float32),
                freqs=freqs, times=times[idx],
                n_trials=int(trials.shape[0]),
                patient=str(patient), contact=str(contact), pair=str(pair))


# --------------------------------------------------------------------------- #
# All stim pairs of one recording electrode, cached to disk
# --------------------------------------------------------------------------- #
def _params(freqs, wavelet, baseline, baseline_per, half_ms, win_ms,
            out_window, decimate) -> dict:
    return dict(wavelet=wavelet, baseline=list(baseline), baseline_per=baseline_per,
                half_ms=half_ms, win_ms=win_ms, out_window=list(out_window),
                decimate=int(decimate), n_freq=int(len(freqs)),
                f0=float(freqs[0]), f1=float(freqs[-1]),
                norm="db-meanlog")   # baseline convention; bump invalidates old caches


def compute_contact_spectrograms(
    patient: str,
    contact: str,
    pairs: list[str] | None = None,
    *,
    recompute: bool = False,
    save: bool = True,
    verbose: bool = True,
    root: Path | None = None,
    freqs: np.ndarray | None = None,
    fmin: float = 2.0,
    fmax: float = 200.0,
    voices_per_octave: int = 10,
    compute_window: tuple[float, float] = (-1.5, 1.5),
    dirname: str | None = None,
    patient_label: str | None = None,
    **spec_kw,
) -> dict:
    """Compute (and cache) the average log10 spectrogram of every stim pair of one
    recording electrode.

    Saves one ``spectrograms.npz`` per contact under
    ``outputs/sub-<patient>/<contact>/`` holding ``spec`` ``(n_pairs, n_freq,
    n_time)`` plus ``pairs`` / ``freqs`` / ``times`` / ``n_trials`` and the parameter
    stamp, so a rerun with the same parameters reloads instead of recomputing. Only
    pairs missing from a valid cache are (re)computed.

    Parameters
    ----------
    pairs : restrict to these stim-pair labels; ``None`` = all pairs recorded at this
        contact. (The group driver passes exactly the pairs a contact contributes.)
    recompute : ignore any cache and recompute every requested pair.
    Other keyword args forward to :func:`connection_spectrogram`.

    Returns
    -------
    dict: ``by_pair`` ``{pair: (n_freq, n_time) float32}``, ``freqs``, ``times``,
    ``n_trials`` ``{pair: int}``, ``path`` (cache file).
    """
    if freqs is None:
        freqs = log_frequencies(fmin, fmax, voices_per_octave)
    freqs = np.asarray(freqs, float)
    baseline = spec_kw.get("baseline", (-0.5, -0.1))
    baseline_per = spec_kw.get("baseline_per", "trial")
    half_ms = spec_kw.get("half_ms", 8.0)
    win_ms = spec_kw.get("win_ms", None)
    out_window = spec_kw.get("out_window", (-0.1, 0.5))
    decimate = spec_kw.get("decimate", 5)
    params = _params(freqs, spec_kw.get("wavelet", DEFAULT_WAVELET), baseline,
                     baseline_per, half_ms, win_ms, out_window, decimate)

    path = contact_dir(patient, contact, root) / CACHE_NAME
    cached, ctimes = _load_spec_cache(path, params) if not recompute else ({}, None)

    # Fast path: the caller named the pairs and the cache already holds all of them
    # -> return straight from cache WITHOUT loading the raw trials.
    # ``load_contact_trials`` (build_convergent_matrix re-reads every raw preproc
    # .mat for this contact) is the slow part, so skip it entirely on a full hit.
    if pairs is not None and cached:
        want = [str(p) for p in pairs]
        if want and all(p in cached for p in want):
            if verbose:
                print(f"  [{patient_label or patient}/{contact}] {len(want)} cached, "
                      f"0 to compute")
            return dict(by_pair={p: cached[p]["spec"] for p in want}, freqs=freqs,
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
            by_pair[p] = cached[p]["spec"]
            n_trials[p] = int(cached[p]["n_trials"])
            continue
        cs = connection_spectrogram(
            patient, contact, p, trials=rec["by_pair"][p],
            times=rec["times"], srate=rec["srate"], freqs=freqs, **spec_kw)
        by_pair[p] = cs["spec"]
        n_trials[p] = cs["n_trials"]
        out_times = cs["times"] if out_times is None else out_times

    if save and by_pair:
        _save_spec_cache(path, by_pair, n_trials, freqs, out_times, params, cached)

    return dict(by_pair=by_pair, freqs=freqs, times=np.asarray(out_times, float),
                n_trials=n_trials, path=path)


def _load_spec_cache(path: Path, params: dict):
    """Return ``(cache_dict, times)`` from a matching cache, else ``({}, None)``.
    ``cache_dict`` maps pair -> ``{'spec', 'n_trials'}``."""
    if not path.exists():
        return {}, None
    try:
        z = np.load(path, allow_pickle=True)
        if json.loads(str(z["params"])) != params:
            return {}, None
        pairs = [str(p) for p in z["pairs"]]
        spec = z["spec"]
        nt = z["n_trials"]
        out = {p: dict(spec=spec[i], n_trials=int(nt[i])) for i, p in enumerate(pairs)}
        return out, np.asarray(z["times"], float)
    except Exception:
        return {}, None


def _save_spec_cache(path, by_pair, n_trials, freqs, times, params, prev):
    """Merge freshly-computed pairs with any still-valid cached pairs and write."""
    merged = {p: (by_pair[p], n_trials[p]) for p in by_pair}
    for p, d in prev.items():                                  # keep old pairs not recomputed
        merged.setdefault(p, (d["spec"], int(d["n_trials"])))
    pairs = sorted(merged)
    spec = np.stack([merged[p][0] for p in pairs]).astype(np.float32)
    nt = np.array([merged[p][1] for p in pairs], int)
    np.savez_compressed(path, spec=spec, pairs=np.asarray(pairs),
                        freqs=np.asarray(freqs, float), times=np.asarray(times, float),
                        n_trials=nt, params=json.dumps(params))


# --------------------------------------------------------------------------- #
# Group: one-sample t-statistic spectrogram per consensus BPC (+ control)
# --------------------------------------------------------------------------- #
def _one_sample_t(stack: np.ndarray) -> np.ndarray:
    """One-sample t vs 0 along axis 0 (``(n, F, T) -> (F, T)``); NaN if n < 2 or the
    bin has zero variance. Thin wrapper over :func:`spectral_common.one_sample_t`."""
    return one_sample_t(stack, axis=0)


def group_bpc_spectrograms(
    res: dict,
    *,
    recompute: bool = False,
    control: bool = True,
    verbose: bool = True,
    freqs: np.ndarray | None = None,
    fmin: float = 2.0,
    fmax: float = 200.0,
    voices_per_octave: int = 10,
    fdr_q: float = 0.05,
    fdr_method: str = "by",
    sig_window: tuple[float | None, float | None] | None = None,
    anonymize: bool = False,
    **spec_kw,
) -> dict:
    """One-sample t-statistic wavelet spectrogram for each consensus BPC in ``res``.

    Walks every clustered connection (:func:`spectral_common.iter_connections`),
    computes/loads its average log10 spectrogram (one CWT pass per recording contact,
    cached by :func:`compute_contact_spectrograms`), stacks the connections of each
    consensus BPC, and takes the one-sample t statistic vs 0 (no log-fold change) at
    every frequency-time bin. With ``control`` the same is done for all *unclustered*
    connections (no assigned BPC / SNR-window-dropped) as a negative control.

    Parameters
    ----------
    res : a clustering result from :func:`bpc_group.group_bpc_analysis_kmeans`
        (or any ``pooled_bpc_clustering*``).
    recompute : force recomputation of the per-contact caches.
    control : also aggregate the unclustered connections (key ``None``).
    fdr_q, fdr_method, sig_window : significance test controlling which
        time-frequency bins differ from rest. The two-sided per-bin p-values are
        thresholded at a ``fdr_q`` false-discovery rate via
        :func:`spectral_common.fdr_significant_pooled` — ``fdr_method='by'`` is the
        Benjamini-Yekutieli (2001) procedure valid **under dependency** (the default;
        ``'bh'`` for independent Benjamini-Hochberg). The correction **family is
        POOLED across every consensus BPC + the control at once** (matching Huang et
        al. ``bpcSpectraBB.m``). ``sig_window=(lo, hi)`` s restricts the family/test
        to a time window (either bound ``None`` = open); the default ``None`` tests
        the whole returned window (incl. the pre-stim sliver, a visible null).
    Other keyword args forward to :func:`compute_contact_spectrograms` /
    :func:`connection_spectrogram` (``baseline``, ``half_ms``, ``wavelet``,
    ``out_window``, ``decimate``, ``dirname`` ...).

    Returns
    -------
    dict:
        ``tstat``   ``{k: (F, T)}`` consensus-BPC t spectrograms (``k`` int; plus
                    ``None`` for the control when ``control``),
        ``sig``     ``{k: (F, T) bool}`` pooled-FDR significant-vs-rest mask per panel,
        ``n_sites`` ``{k: int}`` number of contributing connections,
        ``freqs`` (F,), ``times`` (T,) seconds,
        ``bpc_curves`` ``{k: (times, curve)}`` voltage-domain overlay,
        ``fdr`` ``{q, method, window}`` the significance settings used,
        ``n_clusters``, ``vlim`` suggested symmetric ``(-v, v)`` t color range.
    """
    if freqs is None:
        freqs = log_frequencies(fmin, fmax, voices_per_octave)
    freqs = np.asarray(freqs, float)

    groups = connections_by_cluster(res, control=control)      # {k|None: [rec,...]}
    # which pairs each contact must produce (union over all clusters)
    need: dict[tuple, set] = defaultdict(set)
    for recs in groups.values():
        for r in recs:
            need[(r["patient"], r["contact"])].add(r["pair"])

    # anonymize=True -> progress prints show neutral S1, S2, ... instead of subject ids
    alias = {p: f"S{i + 1}" for i, p in
             enumerate(sorted({pt for pt, _ in need}))} if anonymize else {}
    spec_lookup: dict[tuple, dict] = {}
    out_times = None
    for (patient, contact), want in sorted(need.items()):
        plabel = alias.get(patient) if anonymize else None
        if verbose:
            print(f"[spectrogram] {plabel or patient}/{contact}: {len(want)} pair(s)")
        cc = compute_contact_spectrograms(
            patient, contact, pairs=sorted(want), recompute=recompute,
            verbose=verbose, freqs=freqs, patient_label=plabel, **spec_kw)
        for p, s in cc["by_pair"].items():
            spec_lookup[(patient, contact, p)] = s
        if out_times is None:
            out_times = cc["times"]
        elif not np.allclose(out_times, cc["times"]):
            raise ValueError(
                f"{patient}/{contact} produced a different spectrogram time grid — "
                "all connections must share one grid (same srate / out_window / "
                "decimate). Check for a mixed sampling rate in the pooled subjects.")

    tstat, n_sites = {}, {}
    for k, recs in groups.items():
        specs = [spec_lookup[(r["patient"], r["contact"], r["pair"])]
                 for r in recs
                 if (r["patient"], r["contact"], r["pair"]) in spec_lookup]
        n_sites[k] = len(specs)
        if specs:
            stack = np.stack(specs).astype(float)
            tstat[k] = _one_sample_t(stack)
        else:
            shape = (len(freqs), len(out_times) if out_times is not None else 0)
            tstat[k] = np.full(shape, np.nan)
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

    allt = np.concatenate([np.abs(t[np.isfinite(t)]).ravel() for t in tstat.values()
                           if np.isfinite(t).any()]) if tstat else np.array([])
    v = float(np.percentile(allt, 99)) if allt.size else 5.0

    return dict(tstat=tstat, sig=sig, n_sites=n_sites,
                freqs=freqs, times=np.asarray(out_times, float),
                bpc_curves=bpc_overlay_curves(res),
                fdr=dict(q=float(fdr_q), method=str(fdr_method), window=sig_window),
                n_clusters=int(res.get("n_clusters", 0)), vlim=(-v, v))


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _mpl_colorscale(name: str = "jet", n: int = 32) -> list:
    """A Plotly colorscale sampled from a matplotlib colormap. ``'jet'`` gives the
    requested red<->blue ramp with green/yellow through the middle."""
    import matplotlib
    cmap = matplotlib.colormaps[name]
    xs = np.linspace(0, 1, n)
    return [[float(x), "rgb({},{},{})".format(*(int(round(c * 255))
            for c in cmap(x)[:3]))] for x in xs]


#: Fixed diverging color scale for the t spectrograms (red = power increase, blue =
#: decrease, green/yellow near 0). Shared across all consensus BPCs for comparability.
SPECTRO_CMAP = _mpl_colorscale("RdBu_r")


def plot_group_spectrograms(
    group: dict,
    *,
    cmap=SPECTRO_CMAP,
    vlim: tuple[float, float] | None = None,
    xlim: tuple[float, float] = (0.0, 0.5),
    yscale: str = "linear",
    flim: tuple[float, float] | None = (0.0, 200.0),
    include_control: bool = True,
    overlay_bpc: bool = True,
    overlay_color: str = "black",
    show_significance: bool = True,
    sig_style: str = "whiteout",
    nonsig_color: str = "white",
    nonsig_alpha: float = 1.0,
    sig_color: str = "#00c000",
    sig_width: float = 2.0,
    fdr_q: float | None = None,
    height: int = 380,
    width_per: int = 360,
    show: bool = True,
) -> go.Figure:
    """Plot each consensus BPC's **one-sample t-statistic** spectrogram (+ the
    control) on one FIXED color scale, with the consensus BPC waveform in black —
    the Huang et al. ``bpcSpectraBB.m`` figure. Only the FDR-significant-vs-rest bins
    are kept in color; the rest are whited out.

    Parameters
    ----------
    group : the dict from :func:`group_bpc_spectrograms` (uses ``'tstat'``/``'sig'``).
    cmap : Plotly colorscale (default :data:`SPECTRO_CMAP`, red<->blue diverging).
    vlim : symmetric color range ``(-v, v)``; ``None`` auto-fits a robust
        99th-percentile of |t| across panels. (The paper uses a fixed ``(-10, 10)``.)
    xlim : time axis limits (s). ``overlay_bpc`` draws ``group['bpc_curves']`` on a
        hidden secondary y-axis.
    yscale : frequency-axis scale, ``'linear'`` (default) or ``'log'``. NB the
        frequencies are still sampled log-spaced (10 voices/oct); this only sets how
        they are displayed. For ``'log'`` pass ``flim`` in Hz (or ``None``).
    flim : frequency-axis range in Hz (default ``(0, 200)``); ``None`` = autorange.
    show_significance : mark the FDR-significant-vs-rest bins (mask from
        :func:`group_bpc_spectrograms`). ``False`` disables it entirely — the full
        colored t-map is shown with no significance marking.
    sig_style : how significance is shown when ``show_significance`` —
        ``'whiteout'`` (default) keeps only the significant bins in color and whites
        out the rest; ``'outline'`` **keeps every bin colored** (nothing dropped) and
        draws a contour around the significant regions.
    nonsig_color, nonsig_alpha : (``'whiteout'`` only) color / opacity of the overlay
        over non-significant bins — ``nonsig_alpha=1.0`` (default) fully whites them
        out; lower leaves a faint ghost of the data.
    sig_color, sig_width : (``'outline'`` only) color (default green) and width of the
        significance contour.
    fdr_q : ``None`` (default) uses the mask stored in ``group['sig']``; pass a float
        to **re-threshold** at a different (pooled) FDR q via
        :func:`spectral_common.panel_masks` (no CWT recompute).

    Returns
    -------
    a Plotly ``Figure`` (a row of heatmaps: consensus BPC 0..k-1, then control).
    """
    freqs = np.asarray(group["freqs"], float)
    times = np.asarray(group["times"], float)
    src = group["tstat"]

    keys = [k for k in range(group["n_clusters"]) if k in src]
    if include_control and None in src:
        keys.append(None)
    ncol = len(keys)

    if sig_style not in ("whiteout", "outline"):
        raise ValueError("sig_style must be 'whiteout' or 'outline'")
    masks = panel_masks(group, fdr_q) if show_significance else {}
    maps = {k: np.asarray(src[k], float) for k in keys}

    if vlim is not None:
        vmin, vmax = vlim
    else:                                                      # auto-fit to |t|
        vals = np.concatenate([np.abs(m[np.isfinite(m)]).ravel() for m in maps.values()
                               if np.isfinite(m).any()]) if maps else np.array([])
        v = float(np.percentile(vals, 99)) if vals.size else 5.0
        vmin, vmax = -v, v

    zlabel = "t-stat"
    fig = make_subplots(
        rows=1, cols=ncol, shared_yaxes=True, horizontal_spacing=0.04,
        specs=[[{"secondary_y": True} for _ in range(ncol)]],
        subplot_titles=[("Excluded (control)" if k is None else f"Consensus BPC {k}")
                        + f"  n={group['n_sites'].get(k, 0)}" for k in keys])

    for c, k in enumerate(keys, start=1):
        fig.add_trace(
            go.Heatmap(z=maps[k], x=times, y=freqs,
                       colorscale=cmap, zmin=vmin, zmax=vmax, zmid=0.0,
                       colorbar=dict(title=zlabel, len=0.9) if c == ncol else None,
                       showscale=(c == ncol), zsmooth="best",
                       hovertemplate="t=%{x:.3f}s<br>f=%{y:.1f}Hz<br>" + zlabel +
                                     "=%{z:.3f}<extra></extra>"),
            row=1, col=c, secondary_y=False)

        # mark the FDR-significant-vs-rest bins: 'whiteout' drops the rest, 'outline'
        # keeps every bin colored and draws a contour around the significant regions
        m = masks.get(k)
        if m is not None:
            m = np.asarray(m, bool)
            if sig_style == "whiteout":
                white = np.where(m, np.nan, 1.0)      # NaN over sig (transparent), 1 elsewhere
                fig.add_trace(
                    go.Heatmap(z=white, x=times, y=freqs, showscale=False, hoverinfo="skip",
                               colorscale=[[0.0, nonsig_color], [1.0, nonsig_color]],
                               zmin=0.0, zmax=1.0, zsmooth=False, opacity=nonsig_alpha),
                    row=1, col=c, secondary_y=False)
            elif m.any():                             # 'outline' — nothing dropped
                fig.add_trace(
                    go.Contour(z=m.astype(float), x=times, y=freqs,
                               showscale=False, hoverinfo="skip",
                               colorscale=[[0.0, sig_color], [1.0, sig_color]],
                               contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                               line=dict(width=sig_width, smoothing=1.0)),
                    row=1, col=c, secondary_y=False)

        if overlay_bpc and k is not None and k in group["bpc_curves"]:
            bt, bc = group["bpc_curves"][k]
            fig.add_trace(
                go.Scatter(x=np.asarray(bt, float), y=np.asarray(bc, float),
                           mode="lines", line=dict(color=overlay_color, width=1.6),
                           showlegend=False, hoverinfo="skip"),
                row=1, col=c, secondary_y=True)
            amp = float(np.nanmax(np.abs(bc))) or 1.0
            fig.update_yaxes(range=[-amp * 1.05, amp * 1.05], showticklabels=False,
                             showgrid=False, zeroline=False,
                             row=1, col=c, secondary_y=True)

        fig.update_xaxes(range=list(xlim), title_text="time (s)" if c == 1 else None,
                         row=1, col=c)
        fig.update_yaxes(type=yscale, range=list(flim) if flim is not None else None,
                         row=1, col=c, secondary_y=False,
                         title_text="frequency (Hz)" if c == 1 else None)

    q_used = fdr_q if fdr_q is not None else (group.get("fdr", {}) or {}).get("q", 0.05)
    _mark = "rest whited out" if sig_style == "whiteout" else "sig. outlined"
    sig_note = (f"; pooled FDR q={q_used:.2g} (BY) sig. vs rest ({_mark})"
                if show_significance and masks else "")
    fig.update_layout(
        title=f"Wavelet-spectrogram t-statistic by consensus BPC "
              f"(fixed scale ±{max(abs(vmin), abs(vmax)):.2g})" + sig_note,
        width=max(width_per * ncol, 480), height=height,
        margin=dict(l=60, r=40, t=70, b=50))
    if show:
        fig.show()
    return fig
