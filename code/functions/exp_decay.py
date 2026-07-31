"""
exp_decay.py
============
Weight a convergent response matrix `V` by an exponential decay e^(-t/tau).

The decay is applied along the time axis (axis=1 of V), elementwise per trial.
`tau` defaults to 100 ms.
"""

from __future__ import annotations

import numpy as np


def exp_decay_weight(
    V: np.ndarray,
    times: np.ndarray,
    tau: float = 0.100, 
    clip_pre_stim: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiply V (n_trials x n_times) elementwise by exp(-times / tau).

    Parameters
    ----------
    V : (n_trials, n_times) -- also accepts (n_times,) for a single trace.
    times : (n_times,) in seconds.
    tau : decay constant in seconds (default 0.1 s = 100 ms, used in Huang).
    clip_pre_stim : if True, set the weight to 1 for times < 0 (no boost).

    Returns
    -------
    V_weighted : same shape as V
    weight     : (n_times,) the decay used (so it can be passed back to
                 exp_decay_undo for an exact inverse).
    """
    V = np.asarray(V)
    times = np.asarray(times, dtype=float)
    if V.shape[-1] != times.shape[0]:
        raise ValueError(f"V last axis = {V.shape[-1]} but times has "
                         f"{times.shape[0]} samples.")
    weight = np.exp(-times / tau)
    if clip_pre_stim:
        weight = np.where(times < 0, 1.0, weight)
    return V * weight, weight


def exp_decay_undo(
    V_decayed,
    times: np.ndarray,
    tau: float = 0.100,
    clip_pre_stim: bool = False,
    weight: np.ndarray | None = None,
):
    """Invert `exp_decay_weight`: recover V from V * weight.

    Parameters
    ----------
    V_decayed : either
        - an array with last axis = n_times, or
        - the dict returned by `identify_bpcs(...)` (must contain key 'Bs').
          A shallow copy is returned where 'Bs' is REPLACED with the
          un-decayed (voltage-domain) curves and the original decayed curves
          are preserved under the new key 'Bs_decayed'. All other keys
          (bpc_pairs, pc, ...) pass through unchanged. Idempotent: calling
          twice still leaves 'Bs' as the un-decayed curve.
    times, tau, clip_pre_stim : must match the forward call.
    weight : optional. If provided (as returned by `exp_decay_weight`), used
        directly -- exact round-trip even when `clip_pre_stim=True`. If None,
        weight is re-derived from `times`/`tau`/`clip_pre_stim`.

    Returns
    -------
    Array input  -> ndarray (un-decayed values).
    Dict input   -> dict (shallow copy of input; 'Bs' now un-decayed,
                    'Bs_decayed' added with the original 'Bs').
    """
    if isinstance(V_decayed, dict):
        if "Bs" not in V_decayed:
            raise KeyError("dict input must contain 'Bs' key (as returned by "
                           "identify_bpcs).")
        out = dict(V_decayed)
        out["Bs_decayed"] = V_decayed["Bs"]
        out["Bs"] = exp_decay_undo(
            V_decayed["Bs"], times, tau=tau,
            clip_pre_stim=clip_pre_stim, weight=weight)
        return out

    V_decayed = np.asarray(V_decayed)
    if weight is None:
        times = np.asarray(times, dtype=float)
        if V_decayed.shape[-1] != times.shape[0]:
            raise ValueError(f"V last axis = {V_decayed.shape[-1]} but times "
                             f"has {times.shape[0]} samples.")
        weight = np.exp(-times / tau)
        if clip_pre_stim:
            weight = np.where(times < 0, 1.0, weight)
    return V_decayed / weight