"""
save_outputs.py
===============
Organize per-patient, per-contact pipeline intermediates under

    code/outputs/sub-<patient>/<contact>/
        convergent.npz       V, times, stim_sites, pairs
        projection.npz       P, stim_sites, pairs
        significance.npz     tmat, pairs
        nmf.npz              H, W, n_components, penalty, penalty_mode, pairs
        bpcs_pc1.npz         Bs, times, bpc_pairs, pairs
        stats_pc1.npz        alphas, epsilon2s, V2s, errxprojs,
                             plotweights, p_vals, pairs, stim_sites

Every file includes its stim-pair labels (`pairs`) so it is self-describing.
(The ``_pc1`` suffix is a fixed filename token kept for output compatibility.)

The folder layout mirrors the per-subject convention used in
``data/derivatives/freesurfer/sub-<patient>/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "outputs"


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def contact_dir(patient: str, contact: str, root: Path | None = None) -> Path:
    base = (root or OUTPUTS_ROOT) / f"sub-{patient}" / contact
    base.mkdir(parents=True, exist_ok=True)
    return base


def _save(npz_path: Path, **arrays):
    np.savez_compressed(npz_path, **arrays)


# --------------------------------------------------------------------------- #
# Per-intermediate savers
# --------------------------------------------------------------------------- #
def save_convergent(patient, contact, V, times, stim_sites, pairs, root=None):
    p = contact_dir(patient, contact, root) / "convergent.npz"
    _save(p, V=V, times=times, stim_sites=np.asarray(stim_sites),
          pairs=np.asarray(pairs))
    return p


def save_projection(patient, contact, P, stim_sites, pairs, root=None):
    p = contact_dir(patient, contact, root) / "projection.npz"
    _save(p, P=P, stim_sites=np.asarray(stim_sites), pairs=np.asarray(pairs))
    return p


def save_significance(patient, contact, tmat, pairs, root=None):
    p = contact_dir(patient, contact, root) / "significance.npz"
    _save(p, tmat=tmat, pairs=np.asarray(pairs))
    return p


def save_nmf(patient, contact, nmf, pairs, root=None):
    """`nmf` is the dict returned by functions.nmf.run_nmf()."""
    p = contact_dir(patient, contact, root) / "nmf.npz"
    _save(p,
          H=nmf["H"], W=nmf["W"],
          n_components=np.array(nmf["n_components"]),
          penalty=np.array(nmf["penalty"]),
          penalty_mode=np.array(nmf["penalty_mode"]),
          pairs=np.asarray(pairs))
    return p


def save_bpcs(patient, contact, bpcs, times, pairs, root=None):
    """`bpcs` is the dict returned by functions.bpc_identification.identify_bpcs()."""
    p = contact_dir(patient, contact, root) / "bpcs_pc1.npz"
    _save(p,
          Bs=bpcs["Bs"], times=times,
          bpc_pairs=bpcs["bpc_pairs"],
          pairs=np.asarray(pairs))
    return p


def save_stats(patient, contact, stats, stim_sites, pairs, root=None):
    """`stats` is the dict returned by functions.bpc_stats.curve_stats()."""
    p = contact_dir(patient, contact, root) / "stats_pc1.npz"
    _save(p,
          alphas=stats["alphas"],
          epsilon2s=stats["epsilon2s"],
          V2s=stats["V2s"],
          errxprojs=stats["errxprojs"],
          plotweights=stats["plotweights"],
          p_vals=stats["p_vals"],
          stim_sites=np.asarray(stim_sites),
          pairs=np.asarray(pairs))
    return p


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_contact(patient: str, contact: str, root: Path | None = None) -> dict:
    """Load every ``.npz`` in ``outputs/sub-<patient>/<contact>/``.

    Returns ``{stem: {array_name: ndarray}}`` -- e.g.
    ``out['convergent']['V']``, ``out['bpcs_pc1']['Bs']``.
    """
    base = (root or OUTPUTS_ROOT) / f"sub-{patient}" / contact
    if not base.exists():
        raise FileNotFoundError(f"No outputs found at {base}")
    return {f.stem: dict(np.load(f, allow_pickle=True))
            for f in sorted(base.glob("*.npz"))}


# --------------------------------------------------------------------------- #
# Convenience: save everything from a single results bundle
# --------------------------------------------------------------------------- #
def save_all(patient: str, contact: str, bundle: dict,
             root: Path | None = None) -> dict:
    """Save every available intermediate for one (patient, contact).

    `bundle` may contain any of:
        V, times, stim_sites, pairs       (convergent)
        P                                 (projection)
        tmat                              (significance)
        nmf                               (dict from run_nmf)
        bpcs_pc1                          (dict from identify_bpcs)
        stats_pc1                         (dict from curve_stats)

    Returns {key: saved_path}.
    """
    paths: dict[str, Path] = {}
    pairs = bundle.get("pairs")
    stim_sites = bundle.get("stim_sites")
    times = bundle.get("times")

    if all(k in bundle for k in ("V", "times", "stim_sites", "pairs")):
        paths["convergent"] = save_convergent(
            patient, contact, bundle["V"], bundle["times"],
            bundle["stim_sites"], bundle["pairs"], root=root)

    if "P" in bundle and stim_sites is not None and pairs is not None:
        paths["projection"] = save_projection(
            patient, contact, bundle["P"], stim_sites, pairs, root=root)

    if "tmat" in bundle and pairs is not None:
        paths["significance"] = save_significance(
            patient, contact, bundle["tmat"], pairs, root=root)

    if "nmf" in bundle and pairs is not None:
        paths["nmf"] = save_nmf(patient, contact, bundle["nmf"], pairs, root=root)

    if "bpcs_pc1" in bundle and times is not None and pairs is not None:
        paths["bpcs_pc1"] = save_bpcs(
            patient, contact, bundle["bpcs_pc1"], times, pairs, root=root)

    if "stats_pc1" in bundle and stim_sites is not None and pairs is not None:
        paths["stats_pc1"] = save_stats(
            patient, contact, bundle["stats_pc1"], stim_sites, pairs, root=root)

    return paths