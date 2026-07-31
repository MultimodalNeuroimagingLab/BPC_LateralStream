"""
bpc_identification.py
=====================
Identify basis profile curves (BPCs) from NMF components via kernel PCA on the
trials assigned to each BPC.

For each BPC k, the stim pairs assigned to it are those where H[k] is the
columnwise max AND exceeds the threshold 1/(2*sqrt(n_pairs)). The basis curve is
the first principal component of the concatenated trials, sign-flipped so its
inner product with the trials is positive.
"""

from __future__ import annotations

import numpy as np


def _kpca(X: np.ndarray) -> np.ndarray:
    """Linear kernel PCA -- returns time-domain eigenvectors as columns."""
    F, S, _ = np.linalg.svd(X.T)
    ES = X @ F
    return ES / (np.ones((X.shape[0], 1)) @ S[None])


def identify_bpcs(V, H, stim_sites, pairs) -> dict:
    """Identify BPCs from NMF H and the trials matrix V.

    Parameters
    ----------
    V : (n_trials, n_times) convergent matrix.
    H : (n_components, n_pairs) NMF component matrix (row-L2-normalized).
    stim_sites : (n_trials,) stim-pair label per trial.
    pairs : (n_pairs,) ordered unique pair labels matching columns of H.

    Returns
    -------
    dict:
        Bs : (n_components, n_times) basis curves (kPCA PC1 per BPC).
        bpc_pairs : (n_pairs,) float array; bpc index assigned to each pair
                    (NaN if pair was excluded).
        bpc_to_pairs : {bpc_idx: [pair_name, ...]}
        excluded_pairs : list of pair names not assigned to any BPC.
    """
    n_components = H.shape[0]
    pairs = np.asarray(pairs)
    stim_sites = np.asarray(stim_sites)

    bpc_pairs = np.full(len(pairs), np.nan)
    Bs = np.zeros((n_components, V.shape[1]))

    col_max = np.max(H, axis=0)
    thresh  = 1.0 / (2.0 * np.sqrt(len(pairs)))

    for k in range(n_components):
        assigned = np.where((H[k] == col_max) & (H[k] > thresh))[0]
        bpc_pairs[assigned] = k
        if len(assigned) == 0:
            continue
        trials = np.concatenate([np.where(stim_sites == pairs[i])[0]
                                 for i in assigned])
        comps = _kpca(V[trials].T)
        Bs[k] = comps[:, 0]
        if np.mean(Bs[k] @ V[trials].T) < 0:
            Bs[k] *= -1

    bpc_to_pairs = {int(k): list(pairs[bpc_pairs == k]) for k in range(n_components)}
    excluded = list(pairs[np.isnan(bpc_pairs)])

    return {
        "Bs":             Bs,
        "bpc_pairs":      bpc_pairs,
        "bpc_to_pairs":   bpc_to_pairs,
        "excluded_pairs": excluded,
    }
