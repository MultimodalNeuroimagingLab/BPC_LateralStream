"""
artifact_removal.py
===================
Stimulation-artifact mitigation for CCEP trials, done BEFORE any wavelet /
IIR-filter step so the artifact does not smear across time bins during the
transform (Crowther et al., 2019, J. Neurosci. Methods).

Single entry point: :func:`remove_stim_artifact`. It blanks a symmetric window
around stimulation (``t in [-half, +half]``) and refills it with a **sum of
time-reversed, tapered copies of the immediately-adjacent signal** — so the
replacement carries the same amplitude and (reversed) spectral content as the
surrounding background, and joins the real signal continuously at both edges.

Windowing follows the paper's proportions, scaled to this project's higher
sampling rate / gentler stimulation (BPC responses start ~3 ms, not ~11 ms):

    removed gap      :  [-half, +half]                 (default half = 8 ms)
    preceding window :  [-2*half, -half]   (width = half)  -> reversed, fades OUT
    following window :  [+half, +2*half]   (width = half)  -> reversed, fades IN

i.e. the two replacement windows are each HALF the gap width and sit flush against
it, exactly as in the paper (there: gap [-11, 11] ms, windows [-22, -11] and
[11, 22] ms). Reversing the preceding window makes its first sample equal the
last real sample before the gap (continuity on the left); reversing the following
window makes its last sample equal the first real sample after the gap (continuity
on the right). A linear (or Hann) cross-fade sums the two across the gap.
"""

from __future__ import annotations

import numpy as np


def _time_mask(times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Boolean mask for ``lo <= times < hi`` (half-open so adjacent windows tile
    without sharing a sample)."""
    return (times >= lo) & (times < hi)


def remove_stim_artifact(
    trace: np.ndarray,
    times: np.ndarray,
    *,
    half_ms: float = 8.0,
    win_ms: float | None = None,
    taper: str = "linear",
    return_fill: bool = False,
):
    """Replace the stimulation artifact around ``t = 0`` with reversed, tapered,
    background-matched signal.

    Operates along the LAST axis, so ``trace`` may be a single trial ``(n_time,)``,
    a trial stack ``(n_trials, n_time)``, or any ``(..., n_time)`` array; every
    lead-axis element is refilled independently from its own neighbouring samples.

    Parameters
    ----------
    trace : ``(..., n_time)`` voltage. Not modified in place (a copy is returned).
    times : ``(n_time,)`` seconds, monotonically increasing, spanning ``t = 0``.
    half_ms : half-width of the removed gap in **milliseconds**; the blanked region
        is ``t in [-half_ms, +half_ms]`` (default 8 ms -> a 16 ms gap). The paper
        used 11 ms; 8 ms is the gentler value for this dataset.
    win_ms : width (ms) of each adjacent window used to synthesize the fill. Default
        ``None`` -> ``half_ms`` (the paper's half-gap proportion). The preceding
        window is ``[-half_ms-win_ms, -half_ms]`` and the following window is
        ``[+half_ms, +half_ms+win_ms]``.
    taper : cross-fade weighting across the gap — ``'linear'`` (default) or
        ``'hann'`` (raised cosine; smoother first derivative at the edges).
    return_fill : if True, also return the ``(..., n_gap)`` fill and the gap mask,
        for inspection / plotting.

    Returns
    -------
    ``cleaned`` (same shape/dtype-family as ``trace``), or
    ``(cleaned, fill, gap_mask)`` when ``return_fill``.

    Notes
    -----
    The fill is ``w_pre * reversed_pre_extended + w_post * reversed_post_extended``,
    where each reversed window is reflect-extended to the full gap length (so the
    amplitude never dips in the middle) and ``w_pre`` ramps 1->0 while ``w_post``
    ramps 0->1 across the gap. Continuity is exact at both gap edges regardless of
    taper. If the requested windows run past the start/end of ``times`` the window
    is clipped to what is available (and reflect-extension covers the remainder).
    """
    trace = np.asarray(trace, float)
    times = np.asarray(times, float).ravel()
    if trace.shape[-1] != times.shape[0]:
        raise ValueError(
            f"trace last axis ({trace.shape[-1]}) must match times ({times.shape[0]})")

    half = half_ms / 1000.0
    win = (win_ms if win_ms is not None else half_ms) / 1000.0

    gap = _time_mask(times, -half, half)
    n_gap = int(gap.sum())
    if n_gap == 0:
        raise ValueError(
            f"no samples inside the gap [-{half_ms}, {half_ms}] ms — check that "
            "`times` is in seconds and spans t = 0.")

    pre = _time_mask(times, -half - win, -half)          # preceding window
    post = _time_mask(times, half, half + win)           # following window
    if pre.sum() == 0 or post.sum() == 0:
        raise ValueError(
            "no samples in the preceding/following window — widen the epoch or "
            "reduce win_ms/half_ms.")

    # reversed copies along the last axis: reversed_pre[..., 0] == last real sample
    # before the gap; reversed_post[..., -1] == first real sample after the gap.
    reversed_pre = trace[..., pre][..., ::-1]
    reversed_post = trace[..., post][..., ::-1]

    # extend each reversed window to the full gap length (reflect, so no amplitude
    # dip and no repeated edge sample); keep the continuity-carrying end anchored.
    pre_full = _reflect_to(reversed_pre, n_gap, anchor="start")
    post_full = _reflect_to(reversed_post, n_gap, anchor="end")

    if taper == "linear":
        w_pre = np.linspace(1.0, 0.0, n_gap)
    elif taper == "hann":
        w_pre = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n_gap)))
    else:
        raise ValueError("taper must be 'linear' or 'hann'")
    w_post = 1.0 - w_pre

    fill = pre_full * w_pre + post_full * w_post         # broadcasts over lead axes

    cleaned = trace.copy()
    cleaned[..., gap] = fill

    if return_fill:
        return cleaned, fill, gap
    return cleaned


def _reflect_to(seg: np.ndarray, n: int, *, anchor: str) -> np.ndarray:
    """Extend ``seg`` (``(..., m)``) along the last axis to length ``n`` by reflection.

    ``anchor='start'`` keeps ``seg[..., 0]`` at output index 0 and reflect-pads on the
    RIGHT; ``anchor='end'`` keeps ``seg[..., -1]`` at output index ``n-1`` and
    reflect-pads on the LEFT. If ``seg`` is already >= ``n`` it is truncated on the
    padded side. Reflection uses no-edge-repeat (`mode='reflect'`) when possible,
    falling back to edge-repeat for very short segments.
    """
    seg = np.asarray(seg, float)
    m = seg.shape[-1]
    if m == n:
        return seg
    if m > n:
        return seg[..., :n] if anchor == "start" else seg[..., -n:]
    pad = n - m
    mode = "reflect" if m > 1 else "edge"                # reflect needs >= 2 samples
    pad_width = [(0, 0)] * (seg.ndim - 1)
    pad_width.append((0, pad) if anchor == "start" else (pad, 0))
    return np.pad(seg, pad_width, mode=mode)
