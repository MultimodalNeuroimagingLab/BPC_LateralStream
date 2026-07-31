"""
bpc_group.py
============
Cross-patient (group-level) BPC analysis, self-contained: the pooled KMeans
clustering engine, the response-matrix grids, the consensus-curve figures, the
per-cluster anatomy, and the loc_info readers — everything ``group_bpc_plot``
needs — plus the three front-door functions ``group_bpc_calc`` / ``group_bpc_plot``
/ ``group_bpc_analysis_kmeans``. (Absorbed the former cluster_bpc / response_grid /
consensus_validation / anatomy_dist modules.)
"""

from __future__ import annotations

import glob
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

import functions.load_preproc as lp
import functions.spatial_plot as sp
from functions.bpc_stats import curve_stats, plot_plotweights
from functions.exp_decay import exp_decay_undo, exp_decay_weight
from functions.save_outputs import load_contact
# loc_info readers now live in utils (low-level); re-exported here so
# `from functions.bpc_group import ensure_loc_info` keeps working.
from functions.utils import (ensure_loc_info, load_loc_labels,      # noqa: F401
                             load_destrieux_labels, _load_gm_wm)



def _pair_contribs(labelmap: dict[str, str], pair: str, which: str,
                   blank_label: str, tie: str) -> list[tuple[str, float]]:
    """Resolve a bipolar stim pair 'A-B' to a list of ``(label, weight)``
    contributions whose weights sum to 1 (or 0 if dropped).

    ``which`` = 'first' (anode A only), 'second' (cathode B only), or 'either'
    (default; consider both). ``tie`` governs the case where 'either' finds the
    two contacts in **different** non-blank parcels (a pair straddling a border):
      - 'anode'   : whole pair -> A's parcel (the historical default).
      - 'split'   : 0.5 to A's parcel, 0.5 to B's parcel.
      - 'combine' : whole pair -> a merged 'A/B' bucket (areas sorted).
      - 'drop'    : pair contributes nothing.
    Same-parcel / one-labeled / unlabeled pairs ignore ``tie`` (the whole pair
    goes to the single labeled parcel, or ``blank_label`` if neither is labeled).
    """
    a, _, b = pair.partition("-")
    la = labelmap.get(a.strip(), "").strip()
    lb = labelmap.get(b.strip(), "").strip() if b else ""
    if which == "first":
        return [(la or blank_label, 1.0)]
    if which == "second":
        return [(lb or blank_label, 1.0)]
    # which == 'either'
    if la and lb and la != lb:                       # genuine straddle -> tie rule
        if tie == "split":
            return [(la, 0.5), (lb, 0.5)]
        if tie == "combine":
            return [("/".join(sorted((la, lb))), 1.0)]
        if tie == "drop":
            return []
        return [(la, 1.0)]                           # 'anode'
    return [((la or lb) or blank_label, 1.0)]        # same / one-labeled / neither


def cluster_color(k):
    """Hex color for consensus-cluster index ``k``, taken from the ACTIVE matplotlib
    color cycle (``axes.prop_cycle``). With no theme applied this IS the default tab10
    cycle (so existing figures are unchanged); when a theme is applied (e.g. aquarel)
    the cluster curves / markers follow its palette. Anatomy colors (pathway / visual
    atlas) come from ``spatial_plot``, NOT here, so they stay put."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return mc.to_hex(colors[int(k) % len(colors)])


def cluster_separability(values, clusters):
    """eta^2 (variance explained) + one-way ANOVA + Kruskal-Wallis for a
    continuous ``values`` grouped by categorical ``clusters``.

    eta^2 = SS_between / SS_total is the fraction of ``values``' variance
    explained by cluster membership (0 = no separation, 1 = perfect). The ANOVA
    F-test asks (parametrically) whether the cluster means differ; Kruskal-Wallis
    is the rank-based nonparametric fallback. Returns a dict with eta2, F,
    p_anova, H, p_kw, k (clusters used), N (objects used).
    """
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    grps = [values[clusters == k] for k in np.unique(clusters)]
    grps = [g for g in grps if len(g) > 1]
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in grps)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    F, p_anova = stats.f_oneway(*grps)
    H, p_kw = stats.kruskal(*grps)
    return dict(eta2=eta2, F=F, p_anova=p_anova, H=H, p_kw=p_kw,
                k=len(grps), N=int(sum(len(g) for g in grps)))


def _load_mni_electrodes(patient: str) -> pd.DataFrame:
    """MNI electrodes.tsv (space-MNI152NLin6Sym, from the SPM pipeline) ->
    DataFrame with valid coords + a ``_hemi`` column, parallel to
    spatial_plot._load_electrodes but reading the MNI coordinates."""
    p = (lp.DERIV_ROOT / "mni152_xyz" / f"sub-{patient}"
         / f"sub-{patient}_ses-{lp.SESSION}_space-MNI152NLin6Sym_electrodes.tsv")
    df = pd.read_csv(p, sep="\t", na_values=["n/a", "N/A"])
    df = df.dropna(subset=["x", "y", "z"]).copy()
    df["name"] = df["name"].astype(str)
    if "hemisphere" in df.columns:
        df["_hemi"] = [v.strip()[0].upper() if isinstance(v, str) and v.strip()
                       else ("L" if x < 0 else "R")
                       for v, x in zip(df["hemisphere"], df["x"])]
    else:
        df["_hemi"] = np.where(df["x"].to_numpy(float) < 0, "L", "R")
    return df


# MNI305 (fsaverage surface RAS) -> MNI152 affine (FreeSurfer CoordinateSystems wiki)
_MNI305_TO_MNI152 = np.array([
    [ 0.9975, -0.0073,  0.0176, -0.0429],
    [ 0.0146,  1.0009, -0.0024,  1.5496],
    [-0.0130, -0.0093,  0.9971,  1.1840],
    [ 0.0,     0.0,     0.0,     1.0   ],
])


def _to_mni152(pts: np.ndarray) -> np.ndarray:
    """Apply the documented MNI305->MNI152 affine to fsaverage surface vertices."""
    pts = np.asarray(pts, float)
    return (np.c_[pts, np.ones(len(pts))] @ _MNI305_TO_MNI152.T)[:, :3]


def _fsavg_visual_label(fsavg_dir: Path, h: str, ref_subject: str) -> np.ndarray:
    """Resample ``ref_subject``'s combined visual-atlas labels onto the fsaverage
    surface (nearest-neighbour on the fsaverage sphere, via the subject's
    ``?h.fsaverage.sphere.reg``). Returns a (163842,) label array."""
    side = {"L": "left", "R": "right"}
    fav = np.asarray(nib.load(str(fsavg_dir / f"sphere_{side[h]}.gii.gz")).darrays[0].data, float)
    fs_dir = lp.freesurfer_dir(ref_subject)
    vert_label, _ = sp._load_visual_atlas(fs_dir, h)           # subject-native labels
    sreg, _ = nib.freesurfer.read_geometry(str(fs_dir / "surf" / f"{h.lower()}h.fsaverage.sphere.reg"))
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    idx = cKDTree(unit(np.asarray(sreg, float))).query(unit(fav))[1]
    return vert_label[idx]


def _fsaverage_surfaces(fsavg_dir: Path, hemis: list[str], atlas: str | None = None,
                        ref_subject: str = "deleted",
                        to_mni152: bool = True) -> tuple[dict[str, dict], set[int]]:
    """fsaverage pial+inflated GIFTI -> the surface dict spatial_plot expects
    (pial_pts / infl_pts / infl_tri / vcolors), one per hemisphere. With
    ``atlas='visual'``, color vertices by the visual atlas resampled from
    ``ref_subject``. With ``to_mni152`` (default), the pial+inflated vertices are
    mapped MNI305->MNI152 so the surface matches MNI152NLin6Sym electrodes (the
    per-vertex atlas labels are unaffected — they ride along with the vertices).
    Returns ``(surfaces, present_visual)``."""
    side = {"L": "left", "R": "right"}
    surfaces: dict[str, dict] = {}
    present_visual: set[int] = set()
    for h in hemis:
        pg = nib.load(str(fsavg_dir / f"pial_{side[h]}.gii.gz"))
        ig = nib.load(str(fsavg_dir / f"infl_{side[h]}.gii.gz"))
        pial_pts = np.asarray(pg.darrays[0].data, float)
        infl_pts = np.asarray(ig.darrays[0].data, float)
        infl_tri = np.asarray(ig.darrays[1].data)
        if to_mni152:
            pial_pts = _to_mni152(pial_pts)
            infl_pts = _to_mni152(infl_pts)
        if atlas == "visual":
            vlab = _fsavg_visual_label(fsavg_dir, h, ref_subject).astype(int)
            present_visual |= {int(v) for v in np.unique(vlab) if v > 0}
            vcolors = np.full((infl_pts.shape[0], 3), 0.80)    # light gray default
            valid = (vlab > 0) & (vlab <= len(sp.VISUAL_REGION_NAMES))
            vcolors[valid] = sp.VISUAL_CMAP[vlab[valid] - 1]
        else:
            vcolors = np.full((infl_pts.shape[0], 3), 0.72)    # uniform gray
        surfaces[h] = dict(pial_pts=pial_pts, infl_pts=infl_pts,
                           infl_tri=infl_tri, vcolors=vcolors)
    return surfaces, present_visual


_UNSET = object()   # sentinel: distinguishes "arg not passed" from "passed None"


def _resolve_snr_window(snr_window=(None, 10), bpc_snr_max=_UNSET):
    """Normalize an SNR-window spec into ``(lo, hi)`` (or ``None`` for no gating).

    ``snr_window=(lo, hi)`` keeps per-electrode BPCs whose MEAN assigned-pair SNR
    is within ``[lo, hi]`` (either bound ``None`` = open). The deprecated scalar
    ``bpc_snr_max`` alias, when explicitly passed, wins and maps to
    ``(None, bpc_snr_max)`` (``bpc_snr_max=None`` -> no gating)."""
    if bpc_snr_max is not _UNSET:                  # deprecated alias explicitly passed
        return None if bpc_snr_max is None else (None, bpc_snr_max)
    if snr_window is None:
        return None
    lo, hi = snr_window
    return None if (lo is None and hi is None) else (lo, hi)


def _pool_curves(groups, *, tau=0.100, normalize=True, snr_window=(None, 10)):
    """Pool the electrodes in ``groups`` into one matrix of BPC curves.

    Loads each electrode's saved BPCs (``load_contact``) and exp-decay weights
    them by ``exp(-t/tau)``. Every pooled electrode must share one time grid (the
    first electrode's ``times``); a mismatch raises rather than being resampled,
    since the saved curves are un-weighted and resampling them before re-weighting
    is not an exact round trip (the error is amplified toward late times by the
    inverse weight). With ``normalize=True`` (default) each curve is L2-normalized, so
    ``curves @ curves.T`` is the pairwise cosine-similarity matrix. Used by
    :func:`pooled_bpc_clustering`.

    Parameters
    ----------
    groups : {patient: [ {'contacts': [...]}, ... ]} — only these electrodes are
        pooled (the union of each patient's group ``contacts``).
    tau : exp-decay time constant (s) for the weighting (default 0.100).
    normalize : L2-normalize each curve after the exp-decay weighting (default
        True). Pass False to keep the raw (weighted) amplitudes — e.g. for a PCA
        that should reflect response magnitude rather than only shape.
    snr_window : ``(lo, hi)`` inclusive window on each per-electrode BPC's MEAN
        per-pair SNR (saved ``stats_pc1`` plotweights). A BPC whose mean SNR falls
        OUTSIDE the window is dropped BEFORE it enters the pool, so
        stim-artifact-inflated (too high) or barely-responsive (too low) curves
        never reach the SVD/KMeans and never pollute the consensus curves / glass
        brain / any ``res``-derived view. Either bound may be ``None`` (open);
        ``None`` / ``(None, None)`` = keep every curve. Default ``(None, 10)``
        (drop mean SNR > 10). Dropped curves get ``row_map`` == -1, exactly like
        degenerate curves, so they are excluded everywhere downstream.

    Returns
    -------
    (curves, ref_times, contacts_info, pw_list) where ``curves`` is
    (n_curves x T) exp-decay-weighted (and L2-normalized when ``normalize``);
    ``contacts_info`` is a list of ``(patient, contact, pairs, bpc_pairs,
    row_map)`` (row_map[local_bpc] -> global row, -1 for degenerate / SNR-windowed
    curves); ``pw_list`` is the per-pair SNR (saved plotweights) per electrode,
    parallel to ``contacts_info`` (used to apply ``snr_window`` here; otherwise
    kept for display/coloring).
    """
    snr_window = _resolve_snr_window(snr_window)   # normalize (lo, hi) | None
    allowed = {p: sorted({c for g in gs for c in g['contacts']}) for p, gs in groups.items()}

    # 1. gather each electrode's BPC curves (all on one shared time grid —
    #    enforced below, never resampled). row_map[local_bpc] -> global row
    #    (-1 = degenerate curve dropped) so each stim pair maps back to its cluster.
    blocks, contacts_info, pw_list, ref_times, acc, dropped, snr_dropped = \
        [], [], [], None, 0, 0, 0
    for patient, contacts in allowed.items():
        for contact in contacts:
            try:
                oc = load_contact(patient, contact)
                bp = oc['bpcs_pc1']
                Bs = np.asarray(bp['Bs']); t = np.asarray(bp['times']).ravel()
            except Exception:
                continue                    # no saved outputs for this electrode — skip silently
            if Bs.ndim != 2 or Bs.shape[0] == 0 or not np.isfinite(t).all():
                continue
            if ref_times is None:
                ref_times = t
            elif t.shape != ref_times.shape or not np.allclose(t, ref_times):
                raise ValueError(
                    f"{patient}/{contact}: BPC time grid ({t.size} samples, "
                    f"{t[0]:.4f}–{t[-1]:.4f}s) differs from the pool's reference grid "
                    f"({ref_times.size} samples, {ref_times[0]:.4f}–{ref_times[-1]:.4f}s). "
                    "Pooled curves are not resampled — every electrode must share one "
                    "grid. This is usually a sampling-rate mismatch. drop the odd subject from "
                    "`groups`, or rebuild it with a matching bpc_tmin/bpc_tmax.")
            # per-pair SNR (plotweight) from the saved single-subject stats; used
            # both to apply the bpc_snr_max cap below and for downstream coloring
            pw = np.asarray(oc.get('stats_pc1', {}).get('plotweights',
                            np.full(len(bp['pairs']), np.nan)), float)
            bpc_pairs = np.asarray(bp['bpc_pairs'], float)
            row_map = np.full(Bs.shape[0], -1, int)
            for j, row in enumerate(Bs):
                if not np.isfinite(row).all() or np.linalg.norm(row) == 0:
                    dropped += 1                            # degenerate curve, not usable
                    continue
                if snr_window is not None:                  # SNR-window gate: drop pre-clustering
                    lo, hi = snr_window
                    mj = bpc_pairs == j                     # pairs driving this local BPC
                    if mj.any() and np.isfinite(pw[mj]).any():
                        ms = np.nanmean(pw[mj])             # mean assigned-pair SNR for this BPC
                        if (lo is not None and ms < lo) or (hi is not None and ms > hi):
                            snr_dropped += 1                # leave row_map[j] = -1 -> excluded everywhere
                            continue
                blocks.append(row)
                row_map[j] = acc; acc += 1
            contacts_info.append((patient, contact, np.asarray(bp['pairs']),
                                  bpc_pairs, row_map))
            pw_list.append(pw)                              # SNR per pair, parallel to contacts_info

    if not blocks:                                          # every curve degenerate or SNR-windowed out
        raise ValueError(
            f"no BPC curves survived pooling (snr_window={snr_window}): "
            f"{dropped} degenerate, {snr_dropped} outside the SNR window across "
            f"{len(contacts_info)} electrode(s). Widen snr_window or check the inputs.")

    # 2. weight (exp decay) on the common grid, optionally L2-normalize each curve
    curves = np.vstack(blocks)
    curves, _ = exp_decay_weight(curves, ref_times, tau=tau)
    if normalize:
        curves = curves / np.linalg.norm(curves, axis=1, keepdims=True)
    snr_msg = f", {snr_dropped} outside SNR window {snr_window}" if snr_window is not None else ""
    print(f"pooled {curves.shape[0]} BPC curves from {len(contacts_info)} electrodes / "
          f"{len(groups)} patients  ({dropped} degenerate{snr_msg} skipped)")
    return curves, ref_times, contacts_info, pw_list


def pooled_bpc_clustering(groups, n_pcs=6, show=True,
                          marker_base=3.0, marker_scale=5.0,
                          surf_opacity=0.12, decimate=1, atlas=None,
                          to_mni152=True, fsaverage_dir=None,
                          n_clusters=2, random_state=0,
                          snr_window=(None, 10), bpc_snr_max=_UNSET):
    """Pool BPC curves from the electrodes in ``groups`` (across patients) and
    cluster them: exp-decay weight -> uncentered PCA (SVD) -> KMeans on the scores.

    Produces (1) the cluster-centroid curves, (2) the distribution of
    stim->recording ACPC distances colored by cluster, and (3) a rotatable 3D
    MNI152 GLASS BRAIN (plotly): a transparent pial surface with one marker per
    (stim pair -> recording) connection at the stim-pair midpoint (raw MNI152
    coords), radius = SNR (saved per-pair plotweight), color = cluster, plus the
    recording electrodes as small black dots. Plus distance/cluster separability.

    Parameters
    ----------
    groups : {patient: [ {'contacts': [...]}, ... ]} — only these electrodes are
        pooled (the union of each patient's group ``contacts``).
    n_pcs : PCA components kept (the KMeans input is the n_pcs-D score space).
    n_clusters : KMeans ``k`` (clamped to the number of pooled curves; default 2).
    random_state : KMeans seed (default 0 -> reproducible clusters/colors).
    show : display the figures (False = compute only).
    marker_base, marker_scale : glass-brain node size = marker_base + marker_scale*SNR
        (SNR is a display encoding only — it never filters which points are shown).
    surf_opacity : opacity of the transparent pial "glass" surface (default 0.12).
        Raise it (~0.3-0.5) when ``atlas='visual'`` so the cortex colors show.
    decimate : subsample surface faces for a lighter mesh (1 = full).
    atlas : None (default, plain gray glass) or 'visual' — color the glass-brain
        cortex by the project's visual atlas (Wang+Benson+Rosenke+BN, resampled
        onto fsaverage) and add an ROI legend.
    to_mni152 : map the pial surface MNI305->MNI152 so it matches the electrodes.
    fsaverage_dir : folder with pial/infl GIFTIs; None -> DERIV_ROOT/fsaverage.
    snr_window : ``(lo, hi)`` inclusive window on each per-electrode BPC's MEAN
        per-pair SNR; BPCs whose mean SNR falls OUTSIDE it are dropped BEFORE
        pooling, so artifact-inflated (too high) / barely-responsive (too low)
        curves never enter the SVD/KMeans and never pollute the consensus curves,
        glass brain, or any other ``res``-derived view. Either bound ``None`` =
        open; default ``(None, 10)``. See :func:`_pool_curves`.
    bpc_snr_max : DEPRECATED scalar alias for ``snr_window=(None, bpc_snr_max)``.

    Returns
    -------
    dict: fig_curves, fig_dist, fig_glass, dists, dclust, labels, centroids,
          centroids_plot, times, scores, embedding, variance, PCs, contacts_info,
          pw_list, separability, mni_xyz, n_curves, n_clusters, n_elec.
          ``labels`` is the KMeans cluster index per pooled curve; ``pw_list`` is
          the per-pair SNR (saved plotweights) per electrode, parallel to
          ``contacts_info``. ``PCs`` (T x n_pcs) is the PCA basis; ``curves``
          (n_curves x T) is the exp-decay-weighted signal-space matrix the SVD ran
          on; ``embedding`` is None (KMeans runs on the PCA scores directly).
    """
    # 1-2. pool BPC curves onto a common grid, exp-decay weight (NO L2 norm:
    #      PCA is run on the raw weighted amplitudes, not unit-norm shapes)
    curves, ref_times, contacts_info, pw_list = _pool_curves(
        groups, normalize=False, snr_window=_resolve_snr_window(snr_window, bpc_snr_max))
    X = curves.T

    # 3. SVD (uncentered PCA) + KMeans on the score space
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    var = S**2 / np.sum(S**2)
    cum_var = float(np.sum(var[:n_pcs]))                    # variance captured by the kept PCs
    print(f"PCA: top {n_pcs} PC(s) capture {cum_var*100:.1f}% of variance "
          f"(per-PC: {', '.join(f'{v*100:.1f}%' for v in var[:n_pcs])})")
    PCs = U[:, :n_pcs]
    scores = (S[:n_pcs, None] * Vh[:n_pcs, :]).T
    k = max(1, min(int(n_clusters), scores.shape[0]))       # KMeans needs 1 <= k <= n_samples
    labels = np.asarray(
        KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(scores).labels_)
    embedding = None                                        # KMeans runs on the PCA scores directly
    n_clusters = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    # cluster centroids in signal space: mean PCA score per cluster -> @ PCs.T
    centroids = (np.vstack([scores[labels == c].mean(0) @ PCs.T for c in range(n_clusters)])
                 if n_clusters else np.zeros((0, X.shape[0])))
    print(f"KMeans: k={n_clusters} on {n_pcs}D PCA scores (random_state={random_state})")

    ccolor = cluster_color

    # 4. FIGURE 1 — cluster centroid curves.
    #    Clustering happens in the exp-decay-WEIGHTED domain; the centroids are
    #    un-decayed back to voltage-domain SHAPE (still normalized a.u.) for display.
    centroids_plot = exp_decay_undo(centroids, ref_times)
    fig_curves, ax1 = plt.subplots(figsize=(8, 5))
    for k in range(n_clusters):
        ax1.plot(ref_times, centroids_plot[k], color=ccolor(k), lw=2,
                 label=f"cluster {k} (n={int((labels == k).sum())})")
    ax1.axhline(0, color='0.7', lw=0.6)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("amplitude (voltage-domain shape, norm; a.u.)")
    ax1.set_title(f"Pooled BPC cluster centroids (N_PCS={n_pcs}, {n_clusters} clusters)")
    ax1.legend(fontsize=8); fig_curves.tight_layout()

    # 5. stim->recording Euclidean ACPC distance per connection, tagged by cluster
    acpc = {}
    for p in groups:
        e = sp._load_electrodes(p)
        acpc[p] = {str(n): np.array([x, y, z], float)
                   for n, x, y, z in zip(e['name'], e['x'], e['y'], e['z'])}
    dists, dclust = [], []
    for patient, contact, pairs, bpc_pairs, row_map in contacts_info:
        pos = acpc[patient]; rec = pos.get(contact)
        if rec is None:
            continue
        for i, p in enumerate(pairs):
            b = bpc_pairs[i]
            if np.isnan(b) or row_map[int(b)] < 0:
                continue
            a, _, bb = str(p).partition('-')
            pa = pos.get(a); pbb = pos.get(bb) if bb else pos.get(a)
            if pa is None or pbb is None:
                continue
            dists.append(float(np.linalg.norm((pa + pbb) / 2.0 - rec)))
            dclust.append(int(labels[row_map[int(b)]]))
    dists = np.array(dists); dclust = np.array(dclust)

    # 6. FIGURE 2 — distribution of distances, colored by cluster
    fig_dist, ax2 = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, dists.max() * 1.02, 30) if dists.size else np.linspace(0, 1, 30)
    for k in range(n_clusters):
        dk = dists[dclust == k]
        if not len(dk):
            continue
        ax2.hist(dk, bins=bins, color=ccolor(k), alpha=0.45,
                 label=f"cluster {k} (n={len(dk)}, μ={dk.mean():.1f} mm)")
        ax2.axvline(dk.mean(), color=ccolor(k), lw=2, ls='--')
    ax2.set_xlabel("stim → recording distance (mm, ACPC)"); ax2.set_ylabel("count")
    ax2.set_title("Distribution of stim → recording distances by cluster")
    ax2.legend(fontsize=8); fig_dist.tight_layout()

    # 7. separability of distance by cluster (needs >=2 clusters with distances)
    if dists.size and len(np.unique(dclust)) >= 2:
        sep = cluster_separability(dists, dclust)
        print(f"separability of distance by cluster (N={sep['N']}, k={sep['k']}): "
              f"eta^2={sep['eta2']:.3f}, ANOVA p={sep['p_anova']:.2e}, KW p={sep['p_kw']:.2e}")
    else:
        sep = dict(eta2=np.nan, F=np.nan, p_anova=np.nan, H=np.nan, p_kw=np.nan,
                   k=int(len(np.unique(dclust))), N=int(dists.size))
        print(f"separability of distance by cluster: skipped "
              f"({sep['k']} cluster(s), {sep['N']} connection(s) with distances)")

    # 8. FIGURE 3 — 3D MNI152 GLASS BRAIN (rotatable plotly). A transparent pial
    #    surface + one marker per (stim pair -> recording) connection at the
    #    stim-pair midpoint (RAW MNI152 coords, floating at true positions — no
    #    snapping). Marker radius = SNR (saved per-pair plotweight), color = cluster.
    #    Both hemispheres are rendered at their true MNI positions (never mirrored).
    fsavg = Path(fsaverage_dir) if fsaverage_dir else (lp.DERIV_ROOT / "fsaverage")
    surfaces, present_visual = _fsaverage_surfaces(fsavg, ['L', 'R'], atlas=atlas, to_mni152=to_mni152)

    fig_mni = go.Figure()
    for h, s in surfaces.items():                       # transparent pial = "glass"
        pts, tri = s['pial_pts'], s['infl_tri']         # pial + inflated share topology
        if decimate > 1:
            tri = tri[::decimate]
        if atlas == 'visual':                           # color cortex by visual atlas
            mesh_color = dict(vertexcolor=["#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)) for r, g, b in s['vcolors']])
        else:
            mesh_color = dict(color='lightgray')
        fig_mni.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            opacity=surf_opacity, hoverinfo='skip',
            showscale=False, showlegend=False, name=f"glass {h}", **mesh_color))
    if atlas == 'visual':
        sp._add_visual_legend(fig_mni, present_visual)

    mni = {}                                            # patient -> {contact: MNI152 xyz}
    for patient in groups:
        try:
            elec = _load_mni_electrodes(patient)
        except Exception:
            continue                        # subject lacks MNI electrodes — skip on the map silently
        mni[patient] = {str(n): np.array([x, y, z], float)
                        for n, x, y, z in zip(elec['name'], elec['x'], elec['y'], elec['z'])}

    by_cluster = {k: dict(xyz=[], snr=[], txt=[]) for k in range(n_clusters)}
    gray = dict(xyz=[], txt=[])                          # one gray umbrella: unclustered (noise / no BPC)
    rec_sites = {}                                       # (patient, contact) -> MNI xyz
    missing = []
    for (patient, contact, pairs, bpc_pairs, row_map), plotweights in zip(contacts_info, pw_list):
        pos = mni.get(patient, {})
        for i, p in enumerate(pairs):
            a, _, bb = str(p).partition('-')
            pa = pos.get(a); pbb = pos.get(bb) if bb else pos.get(a)
            if pa is None or pbb is None:
                missing.append((patient, contact, p)); continue
            mid = (pa + pbb) / 2.0
            snr = float(plotweights[i]) if i < len(plotweights) else np.nan
            b = bpc_pairs[i]
            txt = f"{patient[-4:]} {p} (SNR {snr:.2f})"
            clustered = np.isfinite(b) and 0 <= int(b) < len(row_map) and row_map[int(b)] >= 0
            if not clustered:                            # never clustered (no BPC / noise curve)
                gray['xyz'].append(mid); gray['txt'].append(txt)
            else:                                        # clustered -> cluster color, size = SNR
                k = int(labels[row_map[int(b)]])
                by_cluster[k]['xyz'].append(mid)
                by_cluster[k]['snr'].append(snr if np.isfinite(snr) else 0.0)
                by_cluster[k]['txt'].append(txt)
            rp = pos.get(contact)                        # recording electrode position
            if rp is not None:
                rec_sites[(patient, contact)] = rp

    for k, d in by_cluster.items():                     # one trace per cluster, size=SNR
        if not d['xyz']:
            continue
        pos = np.array(d['xyz']); snr = np.array(d['snr'])
        sizes = marker_base + marker_scale * np.clip(snr, 0, None)
        fig_mni.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='markers',
            marker=dict(size=sizes, color=ccolor(k), opacity=0.85,
                        line=dict(color='black', width=0.3)),
            text=d['txt'], name=f"cluster {k} (n={len(snr)})",
            hovertemplate="%{text}<extra>cluster " + str(k) + "</extra>"))

    if gray['xyz']:                                     # unclustered (noise / no BPC) = gray group
        gpos = np.array(gray['xyz'])
        fig_mni.add_trace(go.Scatter3d(
            x=gpos[:, 0], y=gpos[:, 1], z=gpos[:, 2], mode='markers',
            marker=dict(size=marker_base, color='gray', opacity=0.4,
                        line=dict(color='black', width=0.2)),
            text=gray['txt'], name=f"unclustered (n={len(gpos)})",
            hovertemplate="%{text}<extra>unclustered</extra>"))

    if rec_sites:                                       # recording electrodes = black dots
        rpos = np.array(list(rec_sites.values()))
        rtxt = [f"{pp[-4:]} {cc}" for (pp, cc) in rec_sites]
        fig_mni.add_trace(go.Scatter3d(
            x=rpos[:, 0], y=rpos[:, 1], z=rpos[:, 2], mode='markers',
            marker=dict(size=4, color='black', opacity=0.9),
            text=rtxt, name=f"recording electrodes (n={len(rec_sites)})",
            hovertemplate="%{text}<extra>recording</extra>"))

    _title = ("Pooled stim-site glass brain — MNI152 (color = cluster, "
              "black = recording, gray = unclustered)")
    sp._blank_scene_layout(fig_mni, title=_title, width=1100, height=800)
    if missing:
        print(f"  {len(missing)} connection(s) skipped (no MNI coords)")

    if show:
        fig_mni.show()
    else:
        plt.close(fig_curves); plt.close(fig_dist)

    return dict(fig_curves=fig_curves, fig_dist=fig_dist, fig_glass=fig_mni,
                dists=dists, dclust=dclust, labels=labels,
                centroids=centroids,            # exp-decay-weighted (clustering domain)
                centroids_plot=centroids_plot,  # un-decayed voltage-domain shape (plotted)
                times=ref_times, scores=scores, embedding=embedding, variance=var,
                variance_captured=cum_var,      # fraction of variance in the n_pcs kept
                PCs=PCs, curves=curves,
                contacts_info=contacts_info, pw_list=pw_list,
                separability=sep, mni_xyz=mni,
                n_curves=curves.shape[0], n_clusters=n_clusters, n_elec=len(contacts_info))


def pooled_bpc_clustering_kmeans(groups, n_pcs=6, n_clusters=3, *,
                                 random_state=0, snr_window=(None, 10),
                                 bpc_snr_max=_UNSET, **kwargs):
    """Public entry point for the pooled KMeans clustering.

    Thin wrapper over :func:`pooled_bpc_clustering` (which is KMeans-only). KMeans
    assigns every pooled curve to a cluster (0..n_clusters-1, barring curves dropped
    pre-pooling by ``snr_window`` / degeneracy); the returned ``res`` feeds every
    downstream figure (response grids / constituents / ``plot_pca_space`` / glass
    brain). Reproducible via ``random_state``.

    Parameters
    ----------
    n_pcs : PCA components kept (the KMeans input is the n_pcs-D score space).
    n_clusters : KMeans ``k`` (clamped to the number of pooled curves).
    random_state : KMeans seed (default 0 -> reproducible clusters/colors).
    snr_window, bpc_snr_max : per-BPC SNR gate, as in :func:`pooled_bpc_clustering`.
    **kwargs : forwarded to :func:`pooled_bpc_clustering` (show / atlas /
        surf_opacity / marker_* / decimate / to_mni152 / fsaverage_dir).
    """
    return pooled_bpc_clustering(
        groups, n_pcs=n_pcs, n_clusters=n_clusters, random_state=random_state,
        snr_window=snr_window, bpc_snr_max=bpc_snr_max, **kwargs)


def plot_pca_space(
    res: dict,
    pcs: tuple[int, int, int] = (0, 1, 2),
    *,
    category: str = "cluster",
    marker_size: float = 4.0,
    width: int = 900,
    height: int = 700,
    title: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Rotatable 3D scatter of the pooled BPC curves in PCA score space.

    Takes the dict from :func:`pooled_bpc_clustering` (uses ``scores``, ``labels``,
    ``variance``, ``contacts_info``, ``n_curves``) and plots one point per pooled BPC
    curve at its ``(PCx, PCy, PCz)`` PCA scores. Hover shows
    ``'<patient> <recording contact> #<local-BPC index>'``.

    Parameters
    ----------
    res : dict returned by ``pooled_bpc_clustering``.
    pcs : which three PCs go on (x, y, z), 0-indexed; each must be < ``n_pcs``.
    category : how to color the points —
        * ``'cluster'`` (default) — one trace per consensus cluster (tab10, the same
          colors as the centroid curves / glass brain);
        * ``'patient'`` — one trace per subject (tab20).
    marker_size, width, height, show : display options.
    title : plot title; ``None`` -> auto from ``category``.

    Returns
    -------
    The Plotly ``Figure``.
    """
    scores = np.asarray(res["scores"])
    labels = np.asarray(res["labels"])
    var = np.asarray(res["variance"])
    n = int(res["n_curves"])
    n_pcs = scores.shape[1]
    for axis, p in zip("xyz", pcs):
        if not (0 <= p < n_pcs):
            raise ValueError(
                f"PC index {p} for {axis}-axis out of range — only {n_pcs} PCs "
                f"were kept for clustering. Re-run pooled_bpc_clustering with a "
                f"larger n_pcs, or pick PCs < {n_pcs}.")
    px, py, pz = pcs

    # per-curve hover label + patient (row_map -> global row)
    txt = [None] * n
    per_patient = [None] * n
    for patient, contact, _pairs, _bpc_pairs, row_map in res["contacts_info"]:
        for j, gr in enumerate(np.asarray(row_map, int)):
            if gr >= 0:
                txt[gr] = f"{patient[-4:]} {contact} #{j}"
                per_patient[gr] = str(patient)

    fig = go.Figure()
    if category in ("cluster", "patient"):                  # one trace per group
        if category == "cluster":
            keyarr = labels
            uniq = sorted({int(l) for l in labels})
            cmap = {k: cluster_color(k) for k in uniq}
            legend = {k: f"cluster {k}" for k in uniq}
        else:                                               # patient
            keyarr = np.asarray(per_patient, dtype=object)
            uniq = sorted({p for p in per_patient if p is not None})
            pal = cm.tab20(np.linspace(0, 1, 20))
            cmap = {p: mc.to_hex(pal[i % 20]) for i, p in enumerate(uniq)}
            legend = {p: str(p) for p in uniq}
        for k in uniq:
            m = np.asarray(keyarr == k, dtype=bool)
            idx = np.where(m)[0]
            fig.add_trace(go.Scatter3d(
                x=scores[m, px], y=scores[m, py], z=scores[m, pz], mode="markers",
                marker=dict(size=marker_size, color=cmap[k]),
                name=f"{legend[k]} (n={int(m.sum())})",
                text=[txt[i] for i in idx],
                hovertemplate="%{text}<extra>" + legend[k] + "</extra>"))
        auto_title = ("BPCs in PCA space" if category == "cluster"
                      else "BPCs in PCA space (color = patient)")
    else:
        raise ValueError(f"category must be 'cluster' or 'patient'; got {category!r}")

    fig.update_layout(
        title=title or auto_title,
        scene=dict(xaxis_title=f"PC{px+1} ({var[px]*100:.0f}%)",
                   yaxis_title=f"PC{py+1} ({var[py]*100:.0f}%)",
                   zaxis_title=f"PC{pz+1} ({var[pz]*100:.0f}%)"),
        width=width, height=height)
    if show:
        fig.show()
    return fig


def _pair_anatomy_label(labelmap: dict, pair: str, which: str, blank: str, tie: str) -> str:
    """Single loc_info label for a bipolar stim pair, via this module's exact
    tie logic (so it matches the anatomy bars / grid). A straddling pair under
    ``tie='split'`` becomes ``'A/B'``; otherwise one label."""
    labs: list[str] = []
    for lab, _w in _pair_contribs(labelmap, pair, which, blank, tie):
        if lab not in labs:
            labs.append(lab)
    return "/".join(labs) if labs else blank


def bpc_constituents(
    res: dict,
    *,
    include_unclustered: bool = False,
    which_contact: str = "either",
    tie: str = "anode",
    blank_label: str = "non-visual area",
    as_dict: bool = False,
):
    """List the constituent (patient, recording site, stim site) entries of each
    consensus BPC (cluster) from a pooled clustering run, with the loc_info anatomy
    (pathway + visual area) of both the stim site and the recording site.

    One row per **(recording electrode, stim pair) connection** assigned to a kept
    cluster — i.e. every stim pair whose BPC clustered into a consensus curve. The
    cluster index matches the tab10 colour / centroid row used everywhere else in
    the SAME ``res`` (glass brain, response grids, PCA scatter), so a row's
    ``cluster`` ties directly back to ``res['centroids'][cluster]`` /
    ``res['centroids_plot'][cluster]``. Stim-pair anatomy uses the same loc_info
    labels and the same ``which_contact``/``tie`` resolution as the other stim-anatomy
    code (the shared ``_pair_contribs`` tie logic), so the labels agree across figures.

    Parameters
    ----------
    res : dict returned by :func:`pooled_bpc_clustering` (uses ``contacts_info``,
        ``labels`` and ``pw_list``).
    include_unclustered : also return the connections that did NOT end in a kept
        cluster — never significant (no BPC) or a curve dropped before clustering —
        tagged ``cluster = -1`` (the glass brain's grey umbrella). Default False
        (clustered only).
    which_contact, tie : how a bipolar stim pair 'A-B' is reduced to one
        ``pathway`` / ``visual_area`` label — 'either'+'anode' (defaults: whole pair
        -> anode's parcel). Other ``tie`` values ('split'/'combine'/'drop') and
        ``which_contact`` ('first'/'second') use the shared ``_pair_contribs`` tie
        logic; 'split' renders as 'A/B'.
    blank_label : label for stim/recording sites loc_info leaves unlabeled
        (default 'non-visual area').
    as_dict : if True, return ``{cluster: DataFrame}`` (one frame per consensus BPC,
        plus key ``-1`` when ``include_unclustered``); else a single tidy DataFrame.

    Returns
    -------
    A pandas ``DataFrame`` (or dict of them) with columns:
      ``cluster``           — consensus BPC index (``-1`` = unclustered).
      ``patient``           — BIDS subject id.
      ``recording_site``    — recording electrode the BPC was identified at.
      ``recording_shaft``   — that electrode's sEEG shaft/lead (name with the trailing
                              contact number stripped, e.g. ``ROCI11`` -> ``ROCI``).
      ``recording_pathway`` — its loc_info stream (Posterior/Dorsal/Ventral/Lateral).
      ``recording_area``    — its loc_info visual area (V1/LO1/FG.../blank).
      ``recording_destrieux`` — its Destrieux/aseg parcel from electrodes.tsv
                              (``Destrieux_label_text``; mixes ``lh_G_...`` cortical
                              parcels with FS aseg WM/subcortical labels for depth
                              contacts). ``blank_label`` when ``n/a``.
      ``stim_site``         — the stimulation pair (e.g. ``'RPO1-RPO2'``).
      ``stim_pathway``      — the pair's stream (Posterior/Dorsal/Ventral/Lateral).
      ``stim_area``         — the pair's visual area.
      ``stim_destrieux``    — the pair's Destrieux/aseg parcel (same ``which_contact``/
                              ``tie`` resolution as the other stim anatomy columns).
      ``snr``               — that pair's saved per-pair SNR (plotweight); NaN if none.
      ``curve_row``         — global row in ``res['curves']`` / ``res['scores']`` /
                              ``res['labels']`` for this connection's BPC (``-1`` if
                              unclustered), so you can pull the actual curve.
      ``local_bpc``         — the BPC index within that electrode's own run.

    Notes
    -----
    A connection is one (recording electrode, stim pair); the same stim pair seen at
    several recording electrodes appears once per electrode. Pathway / visual-area
    anatomy comes from the per-subject loc_info CSVs (``code/outputs/loc_info_csv/``)
    and the Destrieux columns from the raw ``electrodes.tsv``; if either source is
    missing for a subject those columns fall back to ``blank_label`` (with a printed
    note).
    Group e.g. with ``df.groupby(['cluster','stim_pathway']).size()``.
    """
    labels = np.asarray(res["labels"])
    pw_list = res.get("pw_list")

    maps: dict[str, dict[str, dict]] = {}                   # patient -> {pathway, visualArea}

    def _maps(patient: str) -> dict[str, dict]:
        if patient not in maps:
            try:
                maps[patient] = {"pathway": load_loc_labels(patient, "pathway"),
                                 "visualArea": load_loc_labels(patient, "visualArea")}
            except Exception as e:                          # no CSV -> blank anatomy
                print(f"  (no loc_info labels for {patient}: {e}); anatomy columns blank")
                maps[patient] = {"pathway": {}, "visualArea": {}}
            try:                                            # Destrieux/aseg from electrodes.tsv
                maps[patient]["destrieux"] = load_destrieux_labels(patient)
            except Exception as e:                          # no tsv column -> blank Destrieux
                print(f"  (no Destrieux labels for {patient}: {e}); Destrieux columns blank")
                maps[patient]["destrieux"] = {}
        return maps[patient]

    rows: list[tuple] = []
    for ci, (patient, contact, pairs, bpc_pairs, row_map) in enumerate(res["contacts_info"]):
        bpc_pairs = np.asarray(bpc_pairs, float)
        row_map = np.asarray(row_map, int)
        pw = np.asarray(pw_list[ci], float) if pw_list is not None else None
        m = _maps(str(patient))
        rec_path = m["pathway"].get(str(contact), "") or blank_label
        rec_area = m["visualArea"].get(str(contact), "") or blank_label
        rec_destr = m["destrieux"].get(str(contact), "") or blank_label
        rec_shaft = str(contact).rstrip("0123456789") or str(contact)   # lead, digits stripped
        for i, p in enumerate(pairs):
            b = bpc_pairs[i]
            snr = float(pw[i]) if (pw is not None and i < len(pw)) else np.nan
            local = int(b) if np.isfinite(b) else -1
            gr = row_map[local] if (0 <= local < len(row_map)) else -1
            clustered = gr >= 0
            if clustered:
                k = int(labels[gr])
            elif include_unclustered:                       # grey umbrella -> cluster -1
                k, gr = -1, int(gr)
            else:
                continue
            stim_path = _pair_anatomy_label(m["pathway"], str(p), which_contact, blank_label, tie)
            stim_area = _pair_anatomy_label(m["visualArea"], str(p), which_contact, blank_label, tie)
            stim_destr = _pair_anatomy_label(m["destrieux"], str(p), which_contact, blank_label, tie)
            rows.append((k, str(patient), str(contact), rec_shaft, rec_path, rec_area, rec_destr,
                         str(p), stim_path, stim_area, stim_destr, snr, int(gr), local))

    df = pd.DataFrame(rows, columns=[
        "cluster", "patient", "recording_site", "recording_shaft",
        "recording_pathway", "recording_area", "recording_destrieux",
        "stim_site", "stim_pathway", "stim_area", "stim_destrieux",
        "snr", "curve_row", "local_bpc"])
    df = df.sort_values(["cluster", "stim_pathway", "patient", "recording_site", "stim_site"],
                        kind="stable", ignore_index=True)
    if as_dict:
        return {k: g.reset_index(drop=True) for k, g in df.groupby("cluster")}
    return df


def plot_bpc_matrix(
    df,
    *,
    area_col: str = "stim_area",
    xlim: tuple[float, float] | None = (0.0, 0.5),
    show_trials: bool = True,
    color_bpc_by_cluster: bool = True,
    sort_by: str | None = None,
    order_by_gmwm: bool = False,
    dot_size: float = 90.0,
    trace_lw: float = 1.4,
    figwidth: float = 12.0,
    row_h: float = .8,
    ncols: int = 1,
    anonymize: bool = False,
    show: bool = True,
) -> dict:
    """Matrix figure where each **row is one BPC**, with five columns:
    (1) patient, (2) recording electrode, (3) the BPC's stim pairs as dots labeled
    by name and colored by visual sub-area, (4) the raw voltage traces (per stim
    pair) that fed the BPC, all on one axes, and (5) the BPC curve in time.

    Operates on an already-filtered constituents DataFrame (e.g.
    ``df[(df.cluster==0) & (df.stim_pathway=='Posterior')]``). Rows are grouped into
    one figure row per unique ``(patient, recording_site, local_bpc)``; the stim
    pairs shown (cols 3-4) are exactly those present in ``df`` for that BPC. Raw
    traces and the BPC curve are read from the saved per-electrode outputs
    (``convergent`` V and ``bpcs_pc1`` Bs) via ``load_contact``.

    Parameters
    ----------
    df : filtered DataFrame from :func:`bpc_constituents` (needs ``patient``,
        ``recording_site``, ``local_bpc``, ``stim_site``, ``cluster`` and
        ``area_col``).
    area_col : column used to color stim pairs by visual sub-area (default
        'stim_area'; V1/V2/V3... each get a distinct ``spatial_plot.area_color``).
    xlim : time window (s) for the trace + BPC panels (default 0-0.5).
    show_trials : draw the individual single-trial traces (thin) behind the per-pair
        mean (bold) in column 4; False = means only.
    color_bpc_by_cluster : color the column-5 BPC curve by its consensus cluster
        (else black).
    sort_by : row order — ``None`` (default: alphabetical patient/recording/BPC),
        ``'rel_dist'`` (ascending recording gm/wm RelativeDistance: white-matter/deep
        first → gray/superficial last), or ``'snr'`` (ascending mean per-pair SNR of
        the BPC). Rows with no value sort last. BOTH the rel_dist (raw mm, from
        loc_info's gm_wm.tsv via ``_load_gm_wm``) and the mean SNR are
        shown under every recording label regardless of the sort key.
    order_by_gmwm : deprecated alias — ``True`` == ``sort_by='rel_dist'``.
    ncols : number of side-by-side **block-columns** to tile the BPC rows across
        (default 1 = one tall column, as before). With ``ncols>1`` the rows are
        filled newspaper-style — down block-column 1, then down block-column 2,
        ... — so a long matrix becomes wider and shorter. Column headers and the
        time-axis labels repeat at the top / bottom of each block-column. The
        figure width scales as ``figwidth * ncols``.
    dot_size, trace_lw, figwidth, row_h, show : layout / style controls
        (``figwidth`` is the width of a *single* block-column).

    Returns
    -------
    dict with ``fig`` and ``axes`` (a list of per-row ``(ax_stim, ax_traces,
    ax_bpc)`` triples), plus ``colors`` (the sub-area -> color map).
    """
    import matplotlib.gridspec as gridspec
    from functions.save_outputs import load_contact

    if df.empty:
        raise ValueError("empty df — nothing to plot")
    keys = ["patient", "recording_site", "local_bpc"]
    groups = list(df.groupby(keys, sort=True))

    # anonymize=True -> column 1 shows neutral S1, S2, ... instead of the subject id
    _apt = ({p: f"S{i + 1}" for i, p in
             enumerate(sorted(df["patient"].astype(str).unique()))}
            if anonymize else None)


    if sort_by is None and order_by_gmwm:                   # deprecated alias
        sort_by = "rel_dist"
    if sort_by not in (None, "rel_dist", "snr"):
        raise ValueError("sort_by must be None, 'rel_dist' or 'snr'")

    # per-row recording rel_dist (raw mm from loc_info) + mean per-pair SNR over the
    # BPC's stim pairs present in df — both shown under every recording label and
    # used for the optional row ordering.
    gmwm_cache: dict[str, dict] = {}
    def _reldist(pt, rec):
        pt, rec = str(pt), str(rec)
        if pt not in gmwm_cache:
            gmwm_cache[pt] = _load_gm_wm(pt)
        return gmwm_cache[pt].get(rec, np.nan)
    reldist_of: dict[tuple, float] = {}
    meansnr_of: dict[tuple, float] = {}
    for (pt, rec, lbpc), g in groups:
        reldist_of[(str(pt), str(rec))] = _reldist(pt, rec)
        meansnr_of[(str(pt), str(rec), int(lbpc))] = (
            float(np.nanmean(g["snr"].to_numpy(float))) if "snr" in g else np.nan)

    if sort_by == "rel_dist":                               # ascending; NaN (no value) sorts last
        def _key(kg):
            d = reldist_of[(str(kg[0][0]), str(kg[0][1]))]
            return (0, d) if np.isfinite(d) else (1, 0.0)
        groups.sort(key=_key)
    elif sort_by == "snr":
        def _key(kg):
            s = meansnr_of[(str(kg[0][0]), str(kg[0][1]), int(kg[0][2]))]
            return (0, s) if np.isfinite(s) else (1, 0.0)
        groups.sort(key=_key)
    n = len(groups)

    areas = sorted(df[area_col].astype(str).unique())
    acolor = {a: sp.area_color(a) for a in areas}
    ccolor = cluster_color

    cache: dict[tuple, dict] = {}
    def _oc(pt, rec):
        if (pt, rec) not in cache:
            cache[(pt, rec)] = load_contact(pt, rec)
        return cache[(pt, rec)]

    headers = ["Patient", "Recording", "Stim pairs", "Raw voltage traces", "BPC (time)"]
    widths = [0.7, 1.0, 1.9, 2.4, 1.8]

    # tile the n BPC rows across `ncols` block-columns, filled column-major
    # (down col 1, then down col 2, ...) so ordering is preserved within a column.
    ncols = max(1, int(ncols))
    rows_per_col = int(np.ceil(n / ncols)) if n else 1
    fig = plt.figure(figsize=(figwidth * ncols, max(2.0, row_h * rows_per_col) + 0.7))
    outer = gridspec.GridSpec(rows_per_col, ncols, figure=fig,
                              hspace=0.55, wspace=0.16,
                              left=0.04, right=0.86 if ncols == 1 else 0.90,
                              top=0.90, bottom=0.08)

    def _row_axes(idx):                                     # 5 axes for BPC row idx
        c, r = divmod(idx, rows_per_col)                   # column-major placement
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[r, c], width_ratios=widths, wspace=0.28)
        return r, c, [fig.add_subplot(inner[0, k]) for k in range(5)]

    axes = []
    for i, ((pt, rec, lbpc), g) in enumerate(groups):
        lbpc = int(lbpc)
        r_pos, c_pos, (ax_pt, ax_rec, ax_s, ax_t, ax_b) = _row_axes(i)
        col_bottom = i == min((c_pos + 1) * rows_per_col, n) - 1   # last row in its block-column
        oc = _oc(str(pt), str(rec))
        conv, bp = oc["convergent"], oc["bpcs_pc1"]
        V = np.asarray(conv["V"], float)
        tt = np.asarray(conv["times"], float).ravel()
        ss = np.asarray(conv["stim_sites"]).astype(str)
        Bs = np.asarray(bp["Bs"], float)
        bt = np.asarray(bp["times"], float).ravel()
        stim_pairs = list(g["stim_site"].astype(str))
        stim_areas = list(g[area_col].astype(str))
        kcol = ccolor(int(g["cluster"].iloc[0])) if color_bpc_by_cluster else "black"

        ax_pt.axis("off")
        _pt_lbl = _apt[str(pt)] if _apt is not None else str(pt)
        ax_pt.text(0.5, 0.5, _pt_lbl, ha="center", va="center", fontsize=10, fontweight="bold")
        ax_rec.axis("off")
        d = reldist_of.get((str(pt), str(rec)), np.nan)
        s = meansnr_of.get((str(pt), str(rec), lbpc), np.nan)
        rec_lbl = f"{rec}\nlocal BPC {lbpc}"
        rec_lbl += f"\ngm/wm {d:+.1f}" if np.isfinite(d) else "\ngm/wm n/a"
        rec_lbl += f"\nSNR {s:.2f}" if np.isfinite(s) else "\nSNR n/a"
        ax_rec.text(0.5, 0.5, rec_lbl, ha="center", va="center", fontsize=9)

        # col 3 — stim-pair dots + name labels, colored by visual sub-area
        ax_s.axis("off")
        ax_s.set_xlim(0, 1); ax_s.set_ylim(0, 1)
        m = len(stim_pairs)
        ys = np.linspace(0.85, 0.15, m) if m > 1 else [0.5]
        for sp_name, ar, y in zip(stim_pairs, stim_areas, ys):
            ax_s.scatter(0.1, y, s=dot_size, color=acolor[ar], edgecolors="black",
                         linewidths=0.4, zorder=3)
            ax_s.text(0.22, y, sp_name, ha="left", va="center", fontsize=8)

        # col 4 — raw voltage traces (per-pair mean bold, optional thin trials)
        for sp_name, ar in zip(stim_pairs, stim_areas):
            tr = V[ss == sp_name]
            if not len(tr):
                continue
            if show_trials:
                for row in tr:
                    ax_t.plot(tt, row, color=acolor[ar], lw=0.3, alpha=0.25, zorder=1)
            ax_t.plot(tt, tr.mean(0), color=acolor[ar], lw=trace_lw, alpha=0.95, zorder=2)
        ax_t.axhline(0, color="0.8", lw=0.5)
        if xlim:
            ax_t.set_xlim(*xlim)

        # col 5 — the BPC curve in time
        if 0 <= lbpc < Bs.shape[0]:
            ax_b.plot(bt, Bs[lbpc], color=kcol, lw=1.8)
        ax_b.axhline(0, color="0.8", lw=0.5)
        if xlim:
            ax_b.set_xlim(*xlim)

        if r_pos == 0:                                      # column headers atop each block-column
            for ax, htxt in zip((ax_pt, ax_rec, ax_s, ax_t, ax_b), headers):
                ax.set_title(htxt, fontsize=11, fontweight="bold", pad=8)
        for ax in (ax_t, ax_b):
            if col_bottom:                                  # time axis at the foot of each block-column
                ax.set_xlabel("time (s)", fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
        ax_t.tick_params(labelsize=8); ax_b.tick_params(labelsize=8)
        axes.append((ax_s, ax_t, ax_b))

    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=acolor[a], mec="black",
                          mew=0.4, label=a) for a in areas]
    fig.legend(handles=handles, title="visual sub-area", loc="center right",
               bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    fig.suptitle("BPC matrix — stim pairs, raw traces, and BPC per row", fontsize=13)
    if show:
        plt.show()               # emit as its own display_data (don't rely on the
    else:                        # inline auto-flush, which some frontends bundle into
        plt.close(fig)           # the returned dict's text/plain repr and then drop)
    return dict(fig=fig, axes=axes, colors=acolor)


LATERAL_DEFAULT = ("LO1", "LO2", "TO1", "TO2")
_CANON = {  # within-stream row order; extras get appended alphabetically
    "Posterior": ["V1", "V2", "V3"],
    "Dorsal":    ["V3A", "V3B", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5", "SPL1"],
    "Ventral":   ["hOc4v", "FG1", "FG2", "FG3", "FG4", "A37"],
    "Lateral":   ["LO1", "LO2", "TO1", "TO2"],   # lateral stim sources (always shown)
}


def _norm_area(a: str) -> str:
    return "A37" if a.startswith("A37") else a


def _target_color(name):
    """Header swatch color for a recording-target column. Visual areas keep their
    VISUAL_CMAP atlas color; anything else (pathways, ventral/dorsal areas, custom
    labels) falls back to the shared area palette."""
    nm = "A37a" if name == "A37" else name
    if nm in sp.VISUAL_REGION_NAMES:
        return tuple(sp.VISUAL_CMAP[sp.VISUAL_REGION_NAMES.index(nm)])
    return sp.area_color(name)


def _normalize_recording_targets(recording_targets):
    """-> ({label: {patient: [contacts]}}, ordered_labels).

    Accepts BOTH the label-major ``{label: {patient: [dict(contacts=[...])] |
    [names]}}`` form (the original ``recording_targets``) and the
    ``electrodes_by_region`` patient-major ``{patient: {label: [names]}}`` form
    (auto-detected by MSEL-prefixed outer keys), and leaves that are either
    contact-name lists or ``dict(contacts=...)`` groups."""
    is_patient = lambda k: isinstance(k, str) and k.upper().startswith("MSEL")
    patient_major = len(recording_targets) > 0 and all(is_patient(k) for k in recording_targets)

    def _contacts(leaf):
        if isinstance(leaf, str):
            return [leaf]
        out = []
        for x in leaf:
            if isinstance(x, dict) and "contacts" in x:
                out.extend(x["contacts"])
            elif isinstance(x, str):
                out.append(x)
        return out

    norm = {}
    for outer, inner in recording_targets.items():
        for key, leaf in inner.items():
            label, patient = (key, outer) if patient_major else (outer, key)
            norm.setdefault(label, {}).setdefault(patient, [])
            norm[label][patient] += _contacts(leaf)
    labels = sorted(norm) if patient_major else list(norm)     # sort areas; keep explicit order
    norm = {lab: {p: sorted(set(cs)) for p, cs in norm[lab].items()} for lab in labels}
    return norm, labels


def aggregate_cells(res, targets, streams, rec_target=None,
                    snr_window=(None, 10), bpc_snr_max=_UNSET):
    """-> (denom, color_all, lightgrey, snrgrey, area_path) keyed by (source_area, target).

    ONE pass classifies every (recording electrode, stim pair) connection -- each
    labeled contact of the bipolar pair counts 0.5 -- into one of four states:
      * colored   -- the pair's per-electrode BPC joined a consensus cluster
                     (tallied per cluster in ``color_all``).
      * snrgrey   -- assigned a per-electrode BPC dropped by ``snr_window`` (its mean
                     per-pair SNR is outside the window), so it never entered clustering.
      * lightgrey -- assigned a per-electrode BPC that did NOT join a cluster.
      * white     -- never assigned a per-electrode BPC (implicit remainder).

    ``targets`` = recording-target column labels to keep. ``rec_target`` (optional):
    {(patient, contact): label} explicit recording-site -> column map (else loc_info
    ``visualArea``; stim-source rows always come from loc_info). ``snr_window``:
    ``(lo, hi)`` inclusive window on a per-electrode BPC's MEAN per-pair SNR; drop
    every connection of any BPC whose mean SNR falls OUTSIDE it BEFORE tallying (into
    ``snrgrey``), so those curves never enter the colored count (default ``(None, 10)``;
    either bound ``None`` = open; ``None`` = off). This MUST match the ``snr_window``
    used for the clustering that produced ``res``. ``bpc_snr_max`` is the deprecated
    scalar alias (``(None, bpc_snr_max)``).
    """
    snr_window = _resolve_snr_window(snr_window, bpc_snr_max)
    labels = np.asarray(res["labels"])
    patients = sorted({ci[0] for ci in res["contacts_info"]})
    VA   = {p: load_loc_labels(p, "visualArea") for p in patients}
    PATH = {p: load_loc_labels(p, "pathway")    for p in patients}
    area_path = {}
    for p in patients:
        for name, a in VA[p].items():
            if a:
                area_path.setdefault(_norm_area(a), PATH[p].get(name, ""))

    denom     = defaultdict(float)
    color_all = defaultdict(lambda: defaultdict(float))
    lightgrey = defaultdict(float)
    snrgrey   = defaultdict(float)
    for (patient, R, pairs, bpc_pairs, row_map), pw in zip(res["contacts_info"], res["pw_list"]):
        Rarea = rec_target.get((patient, R), "") if rec_target is not None else VA[patient].get(R, "")
        if Rarea not in targets:                        # recording electrode not a target column
            continue
        bpc_pairs = np.asarray(bpc_pairs, float); row_map = np.asarray(row_map, int)
        pw = np.asarray(pw, float)

        # snr_window: per-electrode BPCs whose MEAN per-pair SNR is outside (lo, hi)
        # (mirrors the pre-clustering gate in _pool_curves so the grid agrees with it)
        excluded_bpcs = set()
        if snr_window is not None:
            lo, hi = snr_window
            for b in np.unique(bpc_pairs[np.isfinite(bpc_pairs)]):
                m = bpc_pairs == b
                if not m.any() or not np.isfinite(pw[m]).any():
                    continue
                mean_snr = np.nanmean(pw[m])
                if (lo is not None and mean_snr < lo) or (hi is not None and mean_snr > hi):
                    excluded_bpcs.add(int(b))

        for i, S in enumerate(pairs):
            a, _, b = str(S).partition("-")
            areas = [_norm_area(VA[patient].get(c.strip(), "")) for c in (a, b)]
            areas = [x for x in areas if x and area_path.get(x) in streams]   # labeled, non-lateral
            if not areas:
                continue
            bp = bpc_pairs[i]
            assigned = np.isfinite(bp) and int(bp) >= 0
            bad_snr = assigned and int(bp) in excluded_bpcs   # BPC dropped by snr_window
            clust = None
            if assigned and not bad_snr:
                bpi = int(bp)
                pooled = int(row_map[bpi]) if bpi < len(row_map) else -1
                if pooled >= 0 and int(labels[pooled]) >= 0:
                    clust = int(labels[pooled])           # joined a consensus cluster
            for area in areas:                            # split: each labeled contact = 0.5
                cell = (area, Rarea)
                denom[cell] += 0.5
                if not assigned:
                    pass                                  # white: never assigned a BPC
                elif bad_snr:
                    snrgrey[cell] += 0.5                  # snr grey: BPC mean SNR outside snr_window
                elif clust is None:
                    lightgrey[cell] += 0.5                # light grey: not clustered
                else:
                    color_all[cell][clust] += 0.5         # colored (joined a consensus cluster)
    return denom, color_all, lightgrey, snrgrey, area_path


def _collapse_hemi_label(lab: str) -> str:
    """Merge the L/R variants of a Destrieux/aseg parcel into one label by stripping
    the hemisphere marker: ``'Left_Hippocampus'`` / ``'Right_Hippocampus'`` ->
    ``'Hippocampus'``, ``'lh_G_parietal_sup'`` / ``'rh_G_parietal_sup'`` ->
    ``'G_parietal_sup'``. Labels with no hemisphere marker (e.g. ``'CSF'``) are
    returned unchanged."""
    low = lab.lower()
    for pre in ("left_", "right_", "lh_", "rh_", "left-", "right-", "lh-", "rh-"):
        if low.startswith(pre):
            return lab[len(pre):]
    return lab


def aggregate_cells_nonvisual(res, targets, *, snr_window=(None, 10),
                              bpc_snr_max=_UNSET, rec_target=None,
                              dest_column="Destrieux_label_text", drop_labels=(),
                              collapse_hemi=True):
    """Like :func:`aggregate_cells`, but the source rows are the **Destrieux / aseg
    parcels of the NON-visual stim contacts** (those loc_info leaves without a
    ``visualArea``) rather than visual areas grouped by stream.

    Visual-labeled contacts are skipped here (they are counted by
    :func:`aggregate_cells` for the visual grid), so the visual and non-visual
    grids **partition** the stim contacts — each labeled contact of a bipolar pair
    still counts 0.5. Non-visual contacts with no Destrieux label (or one in
    ``drop_labels``) are dropped. ``collapse_hemi`` (default True) merges the left
    and right variants of each parcel into one row (strips the ``Left_``/``Right_``
    or ``lh_``/``rh_`` marker). Returns the same tuple as ``aggregate_cells`` minus
    ``area_path``, keyed by ``(destrieux_parcel, recording_target)``.
    """
    snr_window = _resolve_snr_window(snr_window, bpc_snr_max)
    labels = np.asarray(res["labels"])
    patients = sorted({ci[0] for ci in res["contacts_info"]})
    VA   = {p: load_loc_labels(p, "visualArea") for p in patients}
    DEST = {p: load_destrieux_labels(p, dest_column) for p in patients}
    drop = set(drop_labels)

    denom     = defaultdict(float)
    color_all = defaultdict(lambda: defaultdict(float))
    lightgrey = defaultdict(float)
    snrgrey   = defaultdict(float)
    for (patient, R, pairs, bpc_pairs, row_map), pw in zip(res["contacts_info"], res["pw_list"]):
        Rarea = rec_target.get((patient, R), "") if rec_target is not None else VA[patient].get(R, "")
        if Rarea not in targets:                        # recording electrode not a target column
            continue
        bpc_pairs = np.asarray(bpc_pairs, float); row_map = np.asarray(row_map, int)
        pw = np.asarray(pw, float)

        excluded_bpcs = set()                           # snr_window gate (mirrors aggregate_cells)
        if snr_window is not None:
            lo, hi = snr_window
            for b in np.unique(bpc_pairs[np.isfinite(bpc_pairs)]):
                m = bpc_pairs == b
                if not m.any() or not np.isfinite(pw[m]).any():
                    continue
                mean_snr = np.nanmean(pw[m])
                if (lo is not None and mean_snr < lo) or (hi is not None and mean_snr > hi):
                    excluded_bpcs.add(int(b))

        for i, S in enumerate(pairs):
            srcs = []                                   # non-visual contacts of the pair -> parcels
            for c in str(S).split("-"):
                c = c.strip()
                if not c or VA[patient].get(c, ""):     # empty, or VISUAL (belongs to the visual grid)
                    continue
                lab = DEST[patient].get(c, "")
                if not lab or lab in drop:              # unlabeled or explicitly dropped (raw name)
                    continue
                if collapse_hemi:                       # merge L/R into one parcel row
                    lab = _collapse_hemi_label(lab)
                if lab in drop:                         # also honor drop on the collapsed name
                    continue
                srcs.append(lab)
            if not srcs:
                continue
            bp = bpc_pairs[i]
            assigned = np.isfinite(bp) and int(bp) >= 0
            bad_snr = assigned and int(bp) in excluded_bpcs
            clust = None
            if assigned and not bad_snr:
                bpi = int(bp)
                pooled = int(row_map[bpi]) if bpi < len(row_map) else -1
                if pooled >= 0 and int(labels[pooled]) >= 0:
                    clust = int(labels[pooled])         # joined a consensus cluster
            for src in srcs:                            # each non-visual contact = 0.5
                cell = (src, Rarea)
                denom[cell] += 0.5
                if not assigned:
                    pass                                # white: never assigned a BPC
                elif bad_snr:
                    snrgrey[cell] += 0.5
                elif clust is None:
                    lightgrey[cell] += 0.5
                else:
                    color_all[cell][clust] += 0.5
    return denom, color_all, lightgrey, snrgrey


_LIGHT_GREY = "0.80"   # assigned a BPC but not clustered
_SNR_GREY   = "0.90"   # BPC dropped by snr_window (bad mean SNR) — between light grey and white


def _draw_cell(ax, x, y, w, h, d, csplit, n_light, n_snr, n_clusters, ccolor, pad=0.02):
    """One grid cell. The box fills left->right by DENOM: colored BPC segments, then
    light grey (assigned a BPC but not clustered), then snr grey (BPC dropped by
    snr_window for bad mean SNR). The white remainder = connections never assigned a
    BPC. Text = colored / denom."""
    if d == 0:
        ax.text(x + w/2, y + h/2, "·", ha="center", va="center", color="0.6", fontsize=18)
        return
    bx, by, bw, bh = x + pad, y + pad, w - 2*pad, h - 2*pad     # inset so cells don't merge
    numer = sum(csplit.get(k, 0) for k in range(n_clusters))
    xx = bx
    def _seg(v, fc):
        nonlocal xx
        if v <= 1e-9:
            return
        sw = bw * v / d                                 # scale by DENOM -> box fills to the ratio
        ax.add_patch(Rectangle((xx, by), sw, bh, fc=fc, ec="none", zorder=1))
        xx += sw
    for k in range(n_clusters):                         # colored: clustered into a consensus BPC
        _seg(csplit.get(k, 0), ccolor(k))
    _seg(n_light, _LIGHT_GREY)                           # light grey: not clustered
    _seg(n_snr, _SNR_GREY)                               # snr grey: BPC dropped by snr_window
    # white remainder (never assigned a BPC) is left unfilled
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec="0.5", lw=1.1, zorder=2))
    ax.text(x + w/2, y + h/2, f"{numer:g}/{d:g}", ha="center", va="center", fontsize=14, zorder=3,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.78))


def plot_response_ratio_grid(
    groups=None, *, recording_targets=None,
    n_pcs=5, n_clusters=2, snr_window=(None, 10), bpc_snr_max=_UNSET,
    lateral=LATERAL_DEFAULT, stream_order=("Posterior", "Dorsal", "Ventral", "Lateral"),
    transpose=False,
    cell_w=1.5, cell_h=0.8, show=True, res=None,
    atlas="visual", surf_opacity=0.4,
):
    """Build the data-driven ratio/consensus grid (no-SNR-filter).

    Returns ``(fig, data, res)``. Cell fill, drawn left->right and scaled so the box
    fills to the denominator:
      * **colored** (tab10) — clustered into a consensus BPC;
      * **light grey** — assigned a per-electrode BPC but NOT placed in a cluster;
      * **snr grey** (between light grey and white) — assigned a per-electrode BPC
        that was dropped by ``snr_window`` (its mean per-pair SNR is outside the
        window), so it never entered the clustering / PCA space;
      * **white** (unfilled) — never assigned a per-electrode BPC.
    Cell text = colored / total tested.

    Columns are the **recording targets**; they are NOT restricted to the lateral
    stream. Define them via ``recording_targets``, which accepts EITHER:
      * the original label-major form ``{'LO1': {patient: [dict(contacts=[...])]},
        ...}`` (your explicit per-column electrode lists), OR
      * an ``electrodes_by_region(...)`` dict ``{patient: {area: [contacts]}}`` thrown
        in directly — the visual sub-areas (or whatever the inner keys are) become the
        columns (e.g. pass ``electrodes_by_region('Ventral')`` for ventral-area
        columns).
    Either way the listed electrodes become both the columns and the clustered set.
    If ``recording_targets`` is omitted, ``groups`` is clustered and the columns are
    the ``lateral`` visual areas (the original default). Stim-source rows always come
    from loc_info, grouped by ``stream_order``.

    The pooled KMeans clustering is run once **inside** this function (unless a
    pre-built ``res`` is passed); the returned ``res`` carries the matching
    atlas-colored glass brain (``res['fig_glass']``). Pass ``res=...`` back in to
    reuse it without re-clustering.

    ``snr_window`` (default ``(None, 10)``, ``None`` = off): ``(lo, hi)`` inclusive
    window on a per-electrode BPC's MEAN per-pair SNR — BPCs outside it (artifact-
    inflated too high, or barely-responsive too low) are dropped at BOTH stages:
    ``pooled_bpc_clustering`` drops them BEFORE the SVD/KMeans (so they never shape
    the consensus curves or appear on ``res['fig_glass']``), and ``aggregate_cells``
    also removes their connections from the grid denominators — so the grid, glass
    brain, and consensus curves stay consistent. ``bpc_snr_max`` is the deprecated
    scalar alias (``(None, bpc_snr_max)``). A pre-built ``res`` must have used the
    same ``snr_window`` for the two stages to agree.
    """
    snr_window = _resolve_snr_window(snr_window, bpc_snr_max)
    if "Lateral" not in stream_order:                  # lateral stim sources are always included
        stream_order = tuple(stream_order) + ("Lateral",)
    rec_target = None
    target_labels = list(lateral)                      # columns = recording targets (default lateral)
    if recording_targets is not None:                  # any region(s), either supported shape
        rt, target_labels = _normalize_recording_targets(recording_targets)
        rec_target, clust = {}, defaultdict(list)
        for tgt, pdict in rt.items():
            for patient, cs in pdict.items():
                for c in cs:
                    rec_target[(patient, c)] = tgt
                clust[patient].append(dict(name=tgt, contacts=cs))
        groups = {p: gs for p, gs in clust.items()}    # cluster over exactly the listed sites
    if groups is None:
        raise ValueError("pass either `groups` or `recording_targets`")
    if res is None:
        res = pooled_bpc_clustering(groups, n_pcs=n_pcs, n_clusters=n_clusters,
                                       snr_window=snr_window,
                                       atlas=atlas, surf_opacity=surf_opacity, show=False)
    n_clusters = int(np.asarray(res["centroids"]).shape[0])   # KMeans cluster count
    denom, color_all, lightgrey, snrgrey, area_path = aggregate_cells(
        res, set(target_labels), set(stream_order),
        rec_target=rec_target, snr_window=snr_window)
    ccolor = cluster_color

    # source areas grouped/ordered by stream (shared); targets = recording columns
    present = sorted({s for (s, _t) in denom})
    src_areas, src_stream = [], []
    for st in stream_order:
        order = _CANON.get(st, [])
        if st == "Lateral":                            # show ALL lateral sources (symmetric with
            areas = list(lateral)                      # the target columns), even empty ones
        else:
            areas = sorted([a for a in present if area_path.get(a) == st],
                           key=lambda a: (order.index(a) if a in order else 99, a))
        src_areas += areas; src_stream += [st] * len(areas)
    targets = list(target_labels)

    n_rows = len(targets) if transpose else len(src_areas)
    n_cols = len(src_areas) if transpose else len(targets)
    SP = 0.85                                           # stream-panel band thickness
    BAND_GAP = 0.80                                     # stream band sits this far above `top`
    #                                                     (just above the rotated area labels)
    x0, head = (1.05, SP + BAND_GAP) if transpose else (1.35 + SP, 0.95)
    top = n_rows * cell_h

    def _render(colors):
        """Draw the grid: `colors` = per-cell {cluster: n} (color_all)."""
        fig, ax = plt.subplots(figsize=(x0 + n_cols*cell_w + 3.0, top + head + 0.4))

        for i in range(n_rows):                         # cells (src x target either way)
            yb = top - (i + 1) * cell_h
            for j in range(n_cols):
                src, t = (src_areas[j], targets[i]) if transpose else (src_areas[i], targets[j])
                cell = (src, t)
                _draw_cell(ax, x0 + j*cell_w, yb, cell_w, cell_h,
                           denom.get(cell, 0), colors.get(cell, {}),
                           lightgrey.get(cell, 0),
                           snrgrey.get(cell, 0), n_clusters, ccolor)

        for ti, t in enumerate(targets):                # recording-target labels + atlas swatch
            if transpose:                               # -> rows (left)
                cy = top - (ti + 0.5) * cell_h
                ax.add_patch(Rectangle((x0 - 0.36, cy - 0.13), 0.26, 0.26,
                             fc=_target_color(t), ec="black", lw=0.6, zorder=2))
                ax.text(x0 - 0.44, cy, f"{t} →", ha="right", va="center", fontsize=15, weight="bold")
            else:                                       # -> column headers (top)
                cx = x0 + ti*cell_w + cell_w/2
                ax.add_patch(Rectangle((cx - 0.16, top + 0.1), 0.32, 0.28,
                             fc=_target_color(t), ec="black", lw=0.6, zorder=2))
                ax.text(cx, top + 0.5, f"→ {t}", ha="center", va="bottom", fontsize=16, weight="bold")

        for si, s in enumerate(src_areas):              # stim-source-area labels
            if transpose:                               # -> column labels (top, rotated)
                ax.text(x0 + si*cell_w + cell_w/2, top + 0.12, s, ha="left", va="bottom",
                        rotation=55, fontsize=11)
            else:                                       # -> row labels (left)
                ax.text(x0 - 0.12, top - (si + 0.5) * cell_h, s, ha="right", va="center", fontsize=14)

        for st in stream_order:                         # stream panels (pathway color)
            idx = [k for k, s in enumerate(src_stream) if s == st]
            if not idx:
                continue
            lo, hi = min(idx), max(idx)
            pc = sp.PATHWAY_COLOR.get(st, "0.5")
            if transpose:                               # horizontal band on top over its columns
                xa, xb, ya = x0 + lo*cell_w, x0 + (hi + 1)*cell_w, top + head - SP
                ax.add_patch(FancyBboxPatch((xa + 0.05, ya), (xb - xa) - 0.1, SP - 0.12,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             fc=mc.to_rgba(pc, 0.18), ec=pc, lw=1.2, zorder=0))
                ax.text((xa + xb)/2, ya + (SP - 0.12)/2, st, ha="center", va="center",
                        fontsize=14, weight="bold", color=pc)
            else:                                       # vertical band on left over its rows
                y1, y0 = top - lo*cell_h, top - (hi + 1)*cell_h
                ax.add_patch(FancyBboxPatch((0.08, y0 + 0.04), SP - 0.05, (y1 - y0) - 0.08,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             fc=mc.to_rgba(pc, 0.18), ec=pc, lw=1.2, zorder=0))
                ax.text(0.08 + (SP - 0.05)/2, (y0 + y1)/2, st, ha="center", va="center",
                        rotation=90, fontsize=16, weight="bold", color=pc)

        handles = [Rectangle((0, 0), 1, 1, fc=ccolor(k), ec="none") for k in range(n_clusters)]
        labels_ = [f"consensus BPC {k}" for k in range(n_clusters)]
        handles.append(Rectangle((0, 0), 1, 1, fc=_LIGHT_GREY, ec="none"))
        labels_.append("not clustered (dropped before pooling)")
        if snr_window is not None:                       # snr grey: BPC excluded by snr_window
            handles.append(Rectangle((0, 0), 1, 1, fc=_SNR_GREY, ec="0.7"))
            labels_.append(f"BPC dropped: mean SNR ∉ {snr_window}")
        handles.append(Rectangle((0, 0), 1, 1, fc="white", ec="0.5"))
        labels_.append("not assigned a BPC")
        leg = ax.legend(handles, labels_, title="cell fill", loc="upper left",
                        bbox_to_anchor=(1.005, 1.0), fontsize=12, frameon=False)
        leg.get_title().set_fontsize(12)

        ax.set_xlim(0, x0 + n_cols*cell_w + 0.15)
        ax.set_ylim(-0.2, top + head + 0.2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("# clustered responses / # connections tested\n"
                     "box fill = colored/total · white = no BPC · light grey = not clustered"
                     + (" · snr grey = bad mean SNR" if snr_window is not None else ""),
                     fontsize=13)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    fig, _ = _render(color_all)
    data = dict(denom=denom, color_all=color_all,
                lightgrey=lightgrey, snrgrey=snrgrey,
                area_path=area_path)          # stim visual-area -> broader stream (for roll-ups)
    return fig, data, res


def plot_nonvisual_destrieux_grid(
    groups=None, *, recording_targets=None,
    n_pcs=5, n_clusters=2, snr_window=(None, 10), bpc_snr_max=_UNSET,
    lateral=LATERAL_DEFAULT, transpose=False,
    dest_column="Destrieux_label_text", label_order=None,
    drop_labels=(), min_denom=0.0, max_rows=None, collapse_hemi=True,
    cell_w=1.5, cell_h=0.8, show=True, res=None,
    atlas="visual", surf_opacity=0.4,
):
    """Non-visual counterpart of :func:`plot_response_ratio_grid`.

    Identical columns (recording targets) and cell semantics (box fills to
    colored / total, colored by consensus BPC, greys for the same four states),
    but the **rows are the Destrieux / aseg parcels of the NON-visual stim
    contacts** — the pairs the visual grid drops (loc_info leaves them without a
    ``visualArea``) — instead of the visual stim areas grouped by stream. There
    are no pathway/stream panels (non-visual sites have no stream).

    The visual grid and this one **partition** the stim contacts, so pass the SAME
    ``res`` to both (``res=`` from a prior ``plot_response_ratio_grid`` /
    ``pooled_bpc_clustering``) to keep the consensus-BPC colors identical. Returns
    ``(fig, data, res)`` like its sibling.

    Extra parameters beyond :func:`plot_response_ratio_grid`
    --------------------------------------------------------
    dest_column : electrodes.tsv column for the parcel label (default
        ``'Destrieux_label_text'``; ``'Destrieux_label'`` for the numeric code).
    drop_labels : parcels to hide (e.g. ``('Left_Cerebral_White_Matter',
        'Right_Cerebral_White_Matter', 'Unknown')`` to suppress white matter).
    min_denom : hide parcels with fewer than this many total connections.
    max_rows : keep only the top-N parcels by total connections (after the above).
    collapse_hemi : merge the left/right variants of each parcel into one row
        (default True) — e.g. Left_/Right_Hippocampus -> Hippocampus,
        lh_/rh_G_parietal_sup -> G_parietal_sup.
    label_order : explicit top->bottom row order (default: descending total
        connections).
    """
    snr_window = _resolve_snr_window(snr_window, bpc_snr_max)
    rec_target = None
    target_labels = list(lateral)                      # columns = recording targets (default lateral)
    if recording_targets is not None:                  # same normalization as the visual grid
        rt, target_labels = _normalize_recording_targets(recording_targets)
        rec_target, clust = {}, defaultdict(list)
        for tgt, pdict in rt.items():
            for patient, cs in pdict.items():
                for c in cs:
                    rec_target[(patient, c)] = tgt
                clust[patient].append(dict(name=tgt, contacts=cs))
        groups = {p: gs for p, gs in clust.items()}
    if groups is None:
        raise ValueError("pass either `groups` or `recording_targets`")
    if res is None:
        res = pooled_bpc_clustering(groups, n_pcs=n_pcs, n_clusters=n_clusters,
                                       snr_window=snr_window,
                                       atlas=atlas, surf_opacity=surf_opacity, show=False)
    n_clusters = int(np.asarray(res["centroids"]).shape[0])
    denom, color_all, lightgrey, snrgrey = aggregate_cells_nonvisual(
        res, set(target_labels), snr_window=snr_window,
        rec_target=rec_target, dest_column=dest_column, drop_labels=drop_labels,
        collapse_hemi=collapse_hemi)
    ccolor = cluster_color

    # rows = Destrieux parcels present, ordered by descending total connections
    tot = defaultdict(float)
    for (s, _t), v in denom.items():
        tot[s] += v
    src_labels = ([s for s in label_order if s in tot] if label_order
                  else sorted(tot, key=lambda s: (-tot[s], s)))
    src_labels = [s for s in src_labels if tot[s] >= min_denom]
    if max_rows is not None:
        src_labels = src_labels[:int(max_rows)]
    if not src_labels:
        raise ValueError("no non-visual Destrieux parcels to plot (check recording_targets / "
                         "min_denom / drop_labels).")
    targets = list(target_labels)

    n_rows = len(targets) if transpose else len(src_labels)
    n_cols = len(src_labels) if transpose else len(targets)
    # generous left gutter (non-transposed) for the long parcel names
    x0, head = (1.05, 2.2) if transpose else (3.4, 0.95)
    top = n_rows * cell_h

    def _render(colors):
        fig, ax = plt.subplots(figsize=(x0 + n_cols*cell_w + 3.6, top + head + 0.4))
        for i in range(n_rows):
            yb = top - (i + 1) * cell_h
            for j in range(n_cols):
                src, t = (src_labels[j], targets[i]) if transpose else (src_labels[i], targets[j])
                cell = (src, t)
                _draw_cell(ax, x0 + j*cell_w, yb, cell_w, cell_h,
                           denom.get(cell, 0), colors.get(cell, {}),
                           lightgrey.get(cell, 0),
                           snrgrey.get(cell, 0), n_clusters, ccolor)

        for ti, t in enumerate(targets):                # recording-target labels + atlas swatch
            if transpose:                               # -> rows (left)
                cy = top - (ti + 0.5) * cell_h
                ax.add_patch(Rectangle((x0 - 0.36, cy - 0.13), 0.26, 0.26,
                             fc=_target_color(t), ec="black", lw=0.6, zorder=2))
                ax.text(x0 - 0.44, cy, f"{t} →", ha="right", va="center", fontsize=15, weight="bold")
            else:                                       # -> column headers (top)
                cx = x0 + ti*cell_w + cell_w/2
                ax.add_patch(Rectangle((cx - 0.16, top + 0.1), 0.32, 0.28,
                             fc=_target_color(t), ec="black", lw=0.6, zorder=2))
                ax.text(cx, top + 0.5, f"→ {t}", ha="center", va="bottom", fontsize=16, weight="bold")

        for si, s in enumerate(src_labels):             # non-visual Destrieux parcel labels
            if transpose:                               # -> column labels (top, rotated)
                ax.text(x0 + si*cell_w + cell_w/2, top + 0.12, s, ha="left", va="bottom",
                        rotation=55, fontsize=9)
            else:                                       # -> row labels (left)
                ax.text(x0 - 0.12, top - (si + 0.5) * cell_h, s, ha="right", va="center", fontsize=10)

        handles = [Rectangle((0, 0), 1, 1, fc=ccolor(k), ec="none") for k in range(n_clusters)]
        labels_ = [f"consensus BPC {k}" for k in range(n_clusters)]
        handles.append(Rectangle((0, 0), 1, 1, fc=_LIGHT_GREY, ec="none"))
        labels_.append("not clustered (dropped before pooling)")
        if snr_window is not None:
            handles.append(Rectangle((0, 0), 1, 1, fc=_SNR_GREY, ec="0.7"))
            labels_.append(f"BPC dropped: mean SNR ∉ {snr_window}")
        handles.append(Rectangle((0, 0), 1, 1, fc="white", ec="0.5"))
        labels_.append("not assigned a BPC")
        leg = ax.legend(handles, labels_, title="cell fill", loc="upper left",
                        bbox_to_anchor=(1.005, 1.0), fontsize=12, frameon=False)
        leg.get_title().set_fontsize(12)

        ax.set_xlim(0, x0 + n_cols*cell_w + 0.15)
        ax.set_ylim(-0.2, top + head + 0.2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("NON-visual stim (Destrieux parcel) × recording target — "
                     "# clustered responses / # connections tested\n"
                     "box fill = colored/total · white = no BPC · light grey = not clustered"
                     + (" · snr grey = bad mean SNR" if snr_window is not None else ""),
                     fontsize=13)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    fig, _ = _render(color_all)
    data = dict(denom=denom, color_all=color_all,
                lightgrey=lightgrey, snrgrey=snrgrey, src_labels=src_labels)
    return fig, data, res


# --------------------------------------------------------------------------- #
# patient x stim-region matrix, cells colored by BPC (cluster) composition
# --------------------------------------------------------------------------- #


def _display_domain(res, *, undo_decay: bool, normalize: bool):
    """The pooled curves + cluster centroids in ONE consistent display domain.

    ``res['curves']`` lives in the exp-decay-WEIGHTED domain the clustering ran in
    (``exp(-t/tau)``, tau=0.100, NOT L2-normalized) and ``res['centroids']`` is the
    mean PCA score of each cluster reprojected into that same domain — so the two are
    directly overlayable. With ``undo_decay`` (default) BOTH are divided back out by
    the decay, matching what ``res['fig_curves']`` plots (``centroids_plot``); the
    centroid then really is the mean of the curves drawn beneath it.

    ``normalize`` L2-normalizes each curve AND each centroid in the display domain —
    shape-only comparison, at the cost of the amplitude the clustering actually saw.

    Returns ``(curves, centroids, times, ylabel)``.
    """
    times = np.asarray(res["times"], float).ravel()
    curves = np.asarray(res["curves"], float)
    cents = np.asarray(res["centroids"], float)
    if undo_decay:
        curves = exp_decay_undo(curves, times)
        cents = np.asarray(res["centroids_plot"], float)      # same undo, already computed
    unit = "voltage-domain" if undo_decay else "exp-decay-weighted"
    if normalize:
        def _l2(A):
            A = np.asarray(A, float)
            n = np.linalg.norm(A, axis=1, keepdims=True)
            return A / np.where(n == 0, 1.0, n)
        curves, cents = _l2(curves), _l2(cents)
    ylabel = f"amplitude ({unit}{', L2-norm' if normalize else ''}; a.u.)"
    return curves, cents, times, ylabel


def plot_cluster_constituents(
    res: dict,
    *,
    undo_decay: bool = True,
    normalize: bool = False,
    ncols: int = 1,
    xlim: tuple | None = (0, 0.5),
    ylim: tuple | None = (-0.2, 0.2),
    alpha: float = 0.35,
    lw: float = 0.7,
    centroid_lw: float = 2.4,
    color_members: bool = True,
    sharey: bool = True,
    show_axes: bool = True,
    width: int | None = None,
    height: int | None = None,
    show: bool = True,
) -> dict:
    """ONE SUBPLOT PER CONSENSUS CLUSTER (Plotly): every constituent electrode-level
    BPC in that cluster, with the cluster CENTROID overlaid on top.

    Shows what each consensus BPC is actually made of — the spread of its members
    around the centroid that ``res['fig_curves']`` shows alone. unclustered curves
    (label ``-1``) are excluded, so the per-panel n matches the counts used by the
    response grids and glass brain. Each member curve hovers as
    ``<patient> <contact> #<local BPC>``.

    ``show_axes`` (default True) labels the shared time (x) / amplitude (y) axes and
    shows ticks; pass ``show_axes=False`` for the bare poster look (no ticks, tick
    labels, or axis titles — only the colored ``consensus BPC k`` panel titles).
    There is no figure title or legend in either mode; everything is on hover.

    Parameters
    ----------
    res : a clustering result from :func:`group_bpc_analysis_kmeans` /
        :func:`pooled_bpc_clustering` (needs ``curves``, ``centroids``,
        ``centroids_plot``, ``labels``, ``times``).
    undo_decay : plot in the voltage domain (divide the ``exp(-t/tau)`` weighting back
        out of BOTH members and centroid — default, matches ``fig_curves``). ``False``
        plots the exp-decay-weighted domain the clustering ran in.
    normalize : L2-normalize each member curve and centroid (shape-only view; hides
        the amplitude differences the clustering used).
    ncols : subplot columns — default 1, i.e. the clusters STACK vertically (one row
        each, full-width tall panels) so the shared voltage axis is readable. Raise it
        to lay the clusters out side by side instead.
    xlim : shared x-range (default the 0–0.5 s BPC window).
    ylim : shared y-range (default ``(-0.2, 0.2)``, which frames the Lateral group's
        voltage-domain curves). ``None`` = autoscale; with ``sharey`` the autoscale
        range is computed once over ALL panels.
    alpha, lw : member-curve styling. centroid_lw : centroid line width.
    color_members : members in the cluster color (centroid black) — default. ``False``
        draws members grey with the centroid in the cluster color.
    sharey : share the y-axis across panels so cluster amplitudes are comparable.
    width, height : figure size in px (defaults scale with the panel grid).

    Returns
    -------
    dict with ``fig`` (a ``plotly.graph_objects.Figure``) and ``data``
    (``{cluster: {'curves', 'centroid', 'members': [(patient, contact, local_bpc), ...]}}``).
    """
    curves, cents, times, _ylabel = _display_domain(
        res, undo_decay=undo_decay, normalize=normalize)
    labels = np.asarray(res["labels"])
    n_clusters = int(res.get("n_clusters", 0))
    if not n_clusters:
        raise ValueError("res has no clusters to plot (n_clusters=0)")
    owner = _row_owner(res)

    nrows = -(-n_clusters // ncols)                            # ceil division
    titles = [f"Consensus BPC {k+1}" for k in range(n_clusters)]
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True,
                        subplot_titles=titles, vertical_spacing=0.10,
                        horizontal_spacing=0.06)

    data = {}
    for k in range(n_clusters):
        r, c = k // ncols + 1, k % ncols + 1
        m = labels == k
        col = cluster_color(k)
        mem_col = _rgba(col if color_members else "0.6", alpha)
        cen_col = "black" if color_members else col
        for i in np.where(m)[0]:
            own = owner[i]
            nm = f"{own[0][-4:]} {own[1]} #{own[2]}" if own else f"curve {i}"
            fig.add_trace(go.Scattergl(                        # gl: 100+ dense traces
                x=times, y=curves[i], mode="lines", name=nm,
                line=dict(color=mem_col, width=lw), showlegend=False,
                hovertemplate="%{fullData.name}<extra>BPC " + str(k) + "</extra>"),
                row=r, col=c)
        fig.add_trace(go.Scattergl(
            x=times, y=cents[k], mode="lines", name="centroid",
            line=dict(color=cen_col, width=centroid_lw), showlegend=False,
            hovertemplate="centroid<extra>BPC " + str(k) + "</extra>"), row=r, col=c)
        fig.add_hline(y=0, line=dict(color="lightgray", width=1), row=r, col=c)
        fig.layout.annotations[k].font.color = col             # tint the panel title
        data[k] = dict(curves=curves[m], centroid=cents[k],
                       members=[owner[i] for i in np.where(m)[0]])

    yr = list(ylim) if ylim is not None else (                 # one range over ALL panels
        _shared_range(curves[labels >= 0], cents) if sharey else None)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=show_axes,
                     showticklabels=show_axes, ticks="outside" if show_axes else "",
                     range=list(xlim) if xlim else None)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=show_axes,
                     showticklabels=show_axes, ticks="outside" if show_axes else "",
                     range=yr)
    if show_axes:                                              # label the shared axes
        fig.update_xaxes(title_text="time (s)", row=nrows)
        for r in range(1, nrows + 1):
            fig.update_yaxes(title_text="amplitude (a.u.)", row=r, col=1)
    fig.update_layout(
        title=None, showlegend=False,
        template="plotly_white", hovermode="closest",
        width=width or (900 if ncols == 1 else ncols * 420),
        height=height or (nrows * 300))
    if show:
        fig.show()
    return dict(fig=fig, data=data)


def plot_cluster_constituents_by_patient(
    res: dict,
    *,
    undo_decay: bool = True,
    normalize: bool = False,
    xlim: tuple | None = (0, 0.5),
    ylim: tuple | None = (-0.2, 0.2),
    alpha: float = 0.55,
    lw: float = 0.8,
    centroid: bool = True,
    centroid_lw: float = 1.8,
    sharey: bool = True,
    width: int | None = None,
    height: int | None = None,
    show: bool = True,
) -> dict:
    """CLUSTER × PATIENT grid (Plotly): one panel per (consensus cluster, patient),
    holding that patient's BPCs that landed in that cluster, colored by cluster.

    Rows = consensus clusters, columns = patients — so each patient's contribution to
    each cluster is separated out, and every curve carries its cluster's color (the
    same tab10 colors as the grids / centroid figure). Patients are anonymized to
    ``Patient 1, Patient 2, ...`` (in first-seen sorted order) and those labels sit at
    the BOTTOM of the grid; the MSEL id -> number map is returned in ``patient_num``.
    Axis tick labels are suppressed (the panels are a shape grid, not readable axes).
    The consensus centroid is dashed over each panel for reference (``centroid=False``
    to drop it). Empty (cluster, patient) combinations stay as blank panels, so the
    grid stays rectangular and gaps are visible; per-cell counts are returned in
    ``counts`` rather than drawn. Curves hover as ``P<n> <contact> #<local BPC>``.

    Parameters as in :func:`plot_cluster_constituents` (``undo_decay`` again defaults
    to the voltage domain, dividing the ``exp(-t/tau)`` clustering weight back out;
    ``ylim`` again defaults to ``(-0.2, 0.2)``, ``None`` = autoscale).

    Returns
    -------
    dict with ``fig`` (a ``plotly.graph_objects.Figure``), ``patients`` (column order,
    real MSEL ids), ``patient_num`` (``{msel_id: 1-based number}``) and ``counts``
    (``{(cluster, patient): n}``).
    """
    curves, cents, times, ylabel = _display_domain(
        res, undo_decay=undo_decay, normalize=normalize)
    labels = np.asarray(res["labels"])
    n_clusters = int(res.get("n_clusters", 0))
    if not n_clusters:
        raise ValueError("res has no clusters to plot (n_clusters=0)")
    owner = _row_owner(res)

    rows_by: dict = defaultdict(list)                          # (cluster, patient) -> [row idx]
    for i, own in enumerate(owner):
        if own is None or labels[i] < 0:                       # unassigned / unclustered
            continue
        rows_by[(int(labels[i]), own[0])].append(i)
    patients = sorted({p for (_k, p) in rows_by})
    if not patients:
        raise ValueError("no clustered curves could be attributed to a patient")
    pnum = {p: j + 1 for j, p in enumerate(patients)}       

    ncols = len(patients)
    hspace = 0.012
    vspace = 0.006                                            # rows nearly flush
    fig = make_subplots(rows=n_clusters, cols=ncols, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=vspace, horizontal_spacing=hspace)
    counts = {}
    for k in range(n_clusters):
        col = cluster_color(k)
        line_col = _rgba(col, alpha)
        for j, p in enumerate(patients):
            r, c = k + 1, j + 1
            idx = rows_by.get((k, p), [])
            counts[(k, p)] = len(idx)
            for i in idx:
                own = owner[i]
                fig.add_trace(go.Scattergl(
                    x=times, y=curves[i], mode="lines",
                    name=f"P{pnum[p]} {own[1]} #{own[2]}",
                    line=dict(color=line_col, width=lw), showlegend=False,
                    hovertemplate="%{fullData.name}<extra>BPC " + str(k) + "</extra>"),
                    row=r, col=c)
            if centroid:
                fig.add_trace(go.Scattergl(
                    x=times, y=cents[k], mode="lines", name=f"BPC {k} centroid",
                    line=dict(color="rgba(0,0,0,0.8)", width=centroid_lw, dash="dash"),
                    showlegend=False,
                    hovertemplate="centroid<extra>BPC " + str(k) + "</extra>"), row=r, col=c)
            fig.add_hline(y=0, line=dict(color="#eeeeee", width=1), row=r, col=c)

    yr = list(ylim) if ylim is not None else (
        _shared_range(curves[labels >= 0], cents) if sharey else None)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                     ticks="", range=list(xlim) if xlim else None)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     ticks="", range=yr)
    for k in range(n_clusters):                                  # colored row labels, col 1 only
        fig.update_yaxes(title_text=f"<b>Consensus BPC {k}</b>", col=1, row=k + 1,
                         title_font=dict(size=11, color=cluster_color(k)))

    # "Patient N" labels centered UNDER each column at the bottom of the grid
    col_w = (1.0 - hspace * (ncols - 1)) / ncols
    for j, p in enumerate(patients):
        xc = j * (col_w + hspace) + col_w / 2
        fig.add_annotation(text=f"<b>Patient {pnum[p]}</b>", x=xc, y=-0.03,
                           xref="paper", yref="paper", showarrow=False,
                           xanchor="center", yanchor="top", font=dict(size=13, color="black"))

    fig.update_layout(
        title=("Constituent BPCs by consensus cluster (rows) × patient (columns)"
               f"<br><sup>{ylabel}</sup>"),
        template="plotly_white", hovermode="closest",
        margin=dict(b=60),                                       # room for the bottom labels
        width=width or max(260 * ncols, 900),
        height=height or (n_clusters * 200 + 130))
    if show:
        fig.show()
    return dict(fig=fig, patients=patients, patient_num=pnum, counts=counts)


def plot_patient_bpc_overlay(
    res: dict,
    *,
    undo_decay: bool = True,
    normalize: bool = False,
    xlim: tuple | None = (0, 0.5),
    ylim: tuple | None = (-0.2, 0.2),
    alpha: float = 0.8,
    lw: float = 1.2,
    centroid: bool = False,
    centroid_lw: float = 1.6,
    width: int = 560,
    height: int | None = None,
    show: bool = True,
) -> dict:
    """ONE ROW PER PATIENT (Plotly): every clustered BPC that patient contributed,
    all overlaid in a single panel and colored by the consensus cluster it joined.

    The by-cluster counterpart of :func:`plot_cluster_constituents`, transposed to the
    patient: instead of asking "what is this consensus BPC made of", each row asks
    "which consensus BPCs does this patient carry, and what do they look like".
    unclustered curves (label ``-1``) and curves with no patient are excluded.

    Deliberately bare for poster use: **no title, no axis titles, no ticks, no
    legend** — the only text is a black ``Patient <n>`` label to the left of each row.
    Patients are anonymized ``1, 2, 3, ...`` in sorted MSEL order (the same numbering
    :func:`plot_cluster_constituents_by_patient` uses) and the map is returned in
    ``patient_num``. The figure is sized **1:2 (width:height)** by default.

    Parameters
    ----------
    undo_decay, normalize : display domain, as in :func:`plot_cluster_constituents`.
    xlim, ylim : shared ranges (``ylim=None`` autoscales once over all panels).
    alpha, lw : curve styling.
    centroid : also overlay each cluster's consensus centroid (dashed, in the cluster
        color) on every patient row that contributes to that cluster. Default off.
    width, height : px. ``height`` defaults to ``2 * width`` for the 1:2 aspect.

    Returns
    -------
    dict with ``fig``, ``patients`` (row order, real MSEL ids), ``patient_num``
    (``{msel_id: 1-based number}``), ``counts`` (``{patient: n_curves}``) and
    ``counts_by_cluster`` (``{(patient, cluster): n}``).
    """
    curves, cents, times, _ylabel = _display_domain(
        res, undo_decay=undo_decay, normalize=normalize)
    labels = np.asarray(res["labels"])
    n_clusters = int(res.get("n_clusters", 0))
    if not n_clusters:
        raise ValueError("res has no clusters to plot (n_clusters=0)")
    owner = _row_owner(res)

    rows_by: dict = defaultdict(list)                          # patient -> [row idx]
    for i, own in enumerate(owner):
        if own is None or labels[i] < 0:                       # unassigned / unclustered
            continue
        rows_by[own[0]].append(i)
    patients = sorted(rows_by)
    if not patients:
        raise ValueError("no clustered curves could be attributed to a patient")
    pnum = {p: j + 1 for j, p in enumerate(patients)}          

    nrows = len(patients)
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.006)
    counts, by_cluster = {}, {}
    for j, p in enumerate(patients):
        r = j + 1
        idx = sorted(rows_by[p], key=lambda i: int(labels[i]))  # cluster order = draw order
        counts[p] = len(idx)
        fig.add_hline(y=0, line=dict(color="#eeeeee", width=1), row=r, col=1)
        for i in idx:
            k = int(labels[i])
            by_cluster[(p, k)] = by_cluster.get((p, k), 0) + 1
            own = owner[i]
            fig.add_trace(go.Scattergl(
                x=times, y=curves[i], mode="lines",
                name=f"P{pnum[p]} {own[1]} #{own[2]}",
                line=dict(color=_rgba(cluster_color(k), alpha), width=lw),
                showlegend=False,
                hovertemplate="%{fullData.name}<extra>BPC " + str(k) + "</extra>"),
                row=r, col=1)
        if centroid:
            for k in sorted({int(labels[i]) for i in idx}):
                fig.add_trace(go.Scattergl(
                    x=times, y=cents[k], mode="lines", name=f"BPC {k} centroid",
                    line=dict(color=cluster_color(k), width=centroid_lw, dash="dash"),
                    showlegend=False,
                    hovertemplate="centroid<extra>BPC " + str(k) + "</extra>"), row=r, col=1)
        fig.add_annotation(                                    # black "Patient n", left of the row
            text=f"<b>Patient {pnum[p]}</b>", row=r, col=1, showarrow=False,
            xref="x domain", yref="y domain", x=-0.02, y=0.5,
            xanchor="right", yanchor="middle", font=dict(size=13, color="black"))

    yr = list(ylim) if ylim is not None else _shared_range(curves[labels >= 0], cents)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False,          # bare panels
                     showticklabels=False, ticks="",
                     range=list(xlim) if xlim else None)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False,
                     showticklabels=False, ticks="", range=yr)
    fig.update_layout(
        title=None, showlegend=False,
        template="plotly_white", hovermode="closest",
        margin=dict(l=90, r=20, t=20, b=20),                   # room for the patient labels
        width=width, height=height or 2 * width)               # 1:2 width:height
    if show:
        fig.show()
    return dict(fig=fig, patients=patients, patient_num=pnum,
                counts=counts, counts_by_cluster=by_cluster)


# --------------------------------------------------------------------------- #
# response composition by stim region
# --------------------------------------------------------------------------- #
REGION_STREAMS = ("Posterior", "Dorsal", "Ventral", "Lateral")
REGION_DISPLAY = {"Posterior": "Early visual", "Dorsal": "Dorsal",
                  "Ventral": "Ventral", "Lateral": "Lateral"}
_INSIG_COLOR = "#cfcfcf"                                   # grey — insignificant responses


def _region_response_counts(res, *, targets, streams):
    """Per stim-region tally of the (recording electrode, stim-pair) connections in
    ``res``, reusing ``aggregate_cells`` (each bipolar pair split
    half-and-half across its two contacts' visual areas), then summed over recording
    targets to one row per stream.

    Returns ``({stream: {'denom', 'colored': {k: n}, 'grey'}}, n_clusters)`` where
    ``denom`` = total connections tested from that stream, ``colored[k]`` = those whose
    per-electrode BPC joined consensus cluster ``k``, and ``grey`` = everything else
    (SNR-window-filtered + unclustered + never-assigned-a-BPC = "insignificant").

    ``aggregate_cells`` is called with ``snr_window=None`` so ``colored`` is exactly the
    clustered connections (per ``res['labels']``) and ``grey = denom - sum(colored)`` —
    independent of whichever snr_window produced ``res``.
    """
    if targets is None:                                   # recording-site areas present in res
        patients = sorted({ci[0] for ci in res["contacts_info"]})
        VA = {p: load_loc_labels(p, "visualArea") for p in patients}
        targets = sorted({VA[p].get(R, "") for (p, R, *_) in res["contacts_info"]
                          if VA[p].get(R, "")})
    denom, color_all, _lg, _sg, area_path = aggregate_cells(
        res, targets, streams, snr_window=None)
    n_clusters = int(res.get("n_clusters", 0))
    per = {s: dict(denom=0.0, colored={k: 0.0 for k in range(n_clusters)}, grey=0.0)
           for s in streams}
    for (area, tgt), d in denom.items():
        s = area_path.get(area)
        if s not in per:
            continue
        per[s]["denom"] += d
        for k, v in color_all.get((area, tgt), {}).items():
            per[s]["colored"][k] += v
    for rec in per.values():
        rec["grey"] = rec["denom"] - sum(rec["colored"].values())
    return per, n_clusters


def plot_response_ratio_by_region(
    res: dict,
    *,
    targets=None,
    streams=REGION_STREAMS,
    significant_only: bool = True,
    show: bool = True,
    width: int = 780,
    height: int = 430,
) -> dict:
    """Response composition per big visual stim region as normalized horizontal bars.

    One bar per stim region (early-visual = Posterior, Dorsal, Ventral, Lateral),
    normalized to 100 %. Each bar is split into one segment per consensus BPC — sized
    by the share of that region's stim–recording connections whose per-electrode BPC
    joined that cluster, colored with the SAME ``cluster_color`` group_bpc_analysis_kmeans
    uses everywhere — plus a grey "insignificant responses" segment (SNR-filtered +
    unclustered + never-assigned-a-BPC). Each segment's percentage is written above
    the bar in that segment's color, and the total number of connections tested is
    labeled to the right of each bar (bipolar pairs count 0.5 per labeled contact, so
    totals can be fractional).

    Parameters
    ----------
    res : a clustering result from :func:`bpc_group.group_bpc_analysis_kmeans` /
        :func:`pooled_bpc_clustering` (needs ``contacts_info``, ``pw_list``,
        ``labels``, ``n_clusters``).
    targets : recording-target visualArea labels to count connections to. ``None``
        (default) = every recording-site area present in ``res`` (for the Lateral group
        that is LO1/LO2/TO1/TO2).
    streams : stim-region streams to show as bars (default early-visual/Posterior,
        Dorsal, Ventral, Lateral); regions with no tested connections are dropped.
    significant_only : drop the grey "insignificant" segment and renormalize each bar
        to its SIGNIFICANT (clustered) connections only — so each bar shows the share
        of *relevant* responses that are each consensus BPC. The ``n=`` label then
        counts the significant connections, not all tested. Regions with no clustered
        connections are dropped.
    width, height : figure size in px.

    Returns
    -------
    dict with ``fig`` (a ``plotly.graph_objects.Figure``), ``counts`` (the per-stream
    tally from :func:`_region_response_counts`) and ``n_clusters``.
    """
    per, n_clusters = _region_response_counts(res, targets=targets, streams=streams)

    def _denom(s):                                        # normalization base per bar
        return (sum(per[s]["colored"].values()) if significant_only
                else per[s]["denom"])

    active = [s for s in streams if _denom(s) > 0]
    if not active:
        raise ValueError("no {}stim connections from any of the requested regions"
                         .format("significant " if significant_only else ""))
    order = list(reversed(active))                        # plotly puts y[0] at the bottom
    ylab = [REGION_DISPLAY.get(s, s) for s in order]

    fig = go.Figure()
    for k in range(n_clusters):                           # colored consensus-BPC segments
        xk = [per[s]["colored"][k] / _denom(s) * 100 for s in order]
        fig.add_trace(go.Bar(
            y=ylab, x=xk, orientation="h", name=f"BPC {k}",
            marker=dict(color=cluster_color(k), line=dict(color="white", width=0.5)),
            hovertemplate="BPC " + str(k) + ": %{x:.1f}%<extra></extra>"))
    if not significant_only:                              # grey insignificant segment
        xg = [per[s]["grey"] / _denom(s) * 100 for s in order]
        fig.add_trace(go.Bar(
            y=ylab, x=xg, orientation="h", name="insignificant responses",
            marker=dict(color=_INSIG_COLOR, line=dict(color="white", width=0.5)),
            hovertemplate="insignificant: %{x:.1f}%<extra></extra>"))

    # per-segment % above each bar (in the segment color) + total N to the right
    for s in order:
        rec, d, yl = per[s], _denom(s), REGION_DISPLAY.get(s, s)
        segs = [(rec["colored"][k], cluster_color(k)) for k in range(n_clusters)]
        if not significant_only:
            segs.append((rec["grey"], _INSIG_COLOR))
        cum = 0.0
        for cnt, color in segs:
            frac = cnt / d * 100
            if cnt > 0:
                fig.add_annotation(x=cum + frac / 2, y=yl, text=f"{frac:.0f}%",
                                   showarrow=False, yshift=17, xanchor="center",
                                   font=dict(size=15, color=color))
            cum += frac


    sub = ("share of SIGNIFICANT stim–recording pairs per consensus BPC "
           "(insignificant responses excluded)" if significant_only else
           "share of stim–recording pairs per consensus BPC · "
           "grey = insignificant (filtered / unclustered / no BPC)")
    xtitle = ("% of significant stim–recording pairs" if significant_only
              else "% of stim–recording pairs tested")
    fig.update_layout(
        barmode="stack", template="plotly_white", bargap=0.45,
        font=dict(size=15),                                   # bump everything up
        title=f"Response composition by stim region<br><sup>{sub}</sup>",
        xaxis=dict(title=dict(text=xtitle, font=dict(size=12)), range=[0, 100],
                   ticksuffix="%", tickfont=dict(size=12)),
        yaxis=dict(title="", tickfont=dict(size=20)),         # region labels — largest
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0,
                    font=dict(size=12)),
        margin=dict(r=90, t=110), width=width, height=height)
    if show:
        fig.show()
    return dict(fig=fig, counts=per, n_clusters=n_clusters)


# --------------------------------------------------------------------------- #
# helpers (module-level, shared by the group front-door functions)
# --------------------------------------------------------------------------- #
def _groups_from_electrodes(electrodes) -> dict:
    """``electrodes_by_region`` dict -> the ``{patient: [dict(contacts=[...])]}``
    groups shape the clustering engine expects (reuses the grid's normalizer so a
    patient-major or label-major dict both work)."""
    rt, _labels = _normalize_recording_targets(electrodes)
    clust: dict[str, list] = defaultdict(list)
    for tgt, pdict in rt.items():
        for patient, cs in pdict.items():
            clust[patient].append(dict(name=tgt, contacts=cs))
    return {p: gs for p, gs in clust.items()}


def _row_owner(res) -> list:
    """``global curve row -> (patient, contact, local_bpc_index)``, inverted from
    ``res['contacts_info']`` row maps (rows never assigned — degenerate /
    SNR-windowed curves carry ``row_map == -1``) stay ``None``."""
    idx = [None] * int(res["n_curves"])
    for patient, contact, _pairs, _bpc_pairs, row_map in res["contacts_info"]:
        for j, gr in enumerate(np.asarray(row_map, int)):
            if 0 <= gr < len(idx):
                idx[gr] = (str(patient), str(contact), int(j))
    return idx


def _rgba(color: str, alpha: float) -> str:
    """``'#1f77b4'`` / any matplotlib color -> a Plotly ``rgba(...)`` string (Plotly
    has no per-trace line alpha, so the opacity has to ride in the color itself)."""
    r, g, b = mc.to_rgb(color)
    return f"rgba({int(round(r*255))},{int(round(g*255))},{int(round(b*255))},{alpha})"


def _shared_range(*arrays) -> list:
    """A symmetric-ish y-range covering every finite value in ``arrays``, +5% padding."""
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays if np.size(a)])
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * max(hi - lo, 1e-9)
    return [lo - pad, hi + pad]


def _print_region_cluster_counts(grid_data, res, *, label="visual-area stim"):
    """Print, per broader stimulation region (the visual STREAM each stim area
    belongs to) and per consensus cluster, the summed (stim -> recording) connection
    counts, rolled up from the no-filter grid tally (grid units = 0.5 per labeled
    bipolar contact)."""
    color_all = grid_data.get("color_all", {})
    denom     = grid_data.get("denom", {})
    area_path = grid_data.get("area_path", {})
    n_clusters = int(res.get("n_clusters", 0))

    reg_denom: dict[str, float] = defaultdict(float)
    reg_num: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for (area, _rec), d in denom.items():
        reg_denom[area_path.get(area, "?")] += d
    for (area, _rec), per_cluster in color_all.items():
        reg = area_path.get(area, "?")
        for k, v in per_cluster.items():
            reg_num[reg][int(k)] += v

    print(f"[bpc_group] {label}: (stim -> recording) connections per consensus BPC, "
          f"summed per broader stim region (grid units = 0.5 per labeled contact):")
    for reg in sorted(reg_denom):
        d = reg_denom[reg]
        print(f"  {reg}: {sum(reg_num[reg].values()):.1f} clustered / {d:.1f} tested")
        for k in range(n_clusters):
            print(f"      consensus BPC {k}: {reg_num[reg].get(k, 0.0):.1f} / {d:.1f}")


def _nonvisual_grid(res, electrodes, *, transpose, snr_window, atlas, surf_opacity,
                    show, **kw):
    """The NON-visual (Destrieux/aseg-parcel) response matrix, reusing ``res`` so the
    consensus-BPC colors match the visual grid. Returns ``(fig, data)``, or
    ``(None, None)`` with a printed note if this electrode set drives no non-visual
    stim inputs. ``**kw`` forwards trim knobs (``drop_labels`` / ``min_denom`` /
    ``max_rows`` / ``dest_column`` / ``label_order``)."""
    try:
        fig, data, _ = plot_nonvisual_destrieux_grid(
            res=res, recording_targets=electrodes, transpose=transpose,
            snr_window=snr_window, atlas=atlas, surf_opacity=surf_opacity,
            show=show, **kw)
        return fig, data
    except ValueError as e:                                  # no non-visual parcels present
        print(f"[bpc_group] non-visual Destrieux grid skipped: {e}")
        return None, None


# --------------------------------------------------------------------------- #
# 1. calc  2. plot  3. combine
# --------------------------------------------------------------------------- #
def group_bpc_calc(electrodes, *, n_dims: int = 7, n_clusters: int = 2,
                   snr_window: tuple = (0.5, 10), random_state: int = 0,
                   atlas: str = "visual", surf_opacity: float = 0.4,
                   show: bool = False) -> dict:
    """Run the pooled cross-patient KMeans clustering once and return ``res``.

    ``res`` (from :func:`pooled_bpc_clustering_kmeans`) carries the
    cluster labels, centroids, pooled curves, per-pair SNRs, and the MNI152 glass
    brain (``res['fig_glass']``) — everything :func:`group_bpc_plot` needs.

    Parameters
    ----------
    electrodes : ``electrodes_by_region`` dict — the recording electrodes to pool.
    n_dims : PCA components kept (the KMeans input dimensionality).
    n_clusters : KMeans ``k`` — the number of consensus BPCs (default 2).
    snr_window : ``(lo, hi)`` per-electrode-BPC mean-SNR gate applied BEFORE
        clustering; BPCs outside it never enter the pool (default ``(0.5, 10)``).
    random_state : KMeans seed (default 0 -> reproducible clusters/colors).
    atlas, surf_opacity : glass-brain options.
    """
    return pooled_bpc_clustering_kmeans(
        _groups_from_electrodes(electrodes),
        n_pcs=n_dims, n_clusters=n_clusters, random_state=random_state,
        snr_window=snr_window, atlas=atlas, surf_opacity=surf_opacity, show=show)


def group_bpc_plot(res: dict, electrodes, *, transpose: bool = True,
                   atlas: str = "visual", surf_opacity: float = 0.4,
                   snr_window: tuple = (0.5, 10), show: bool = True,
                   nonvisual_kwargs: dict | None = None) -> dict:
    """Turn one clustering ``res`` into every group figure (all sharing its colors).

    Produces, and (if ``show``) displays:
      * ``fig_visual``            — visual-area response matrix (no SNR filter);
      * ``fig_nonvisual``         — non-visual (Destrieux/aseg) response matrix;
      * ``fig_constituents``      — per-cluster consensus constituent curves, axes on
        (:func:`plot_cluster_constituents`);
      * ``fig_response_by_region``— response composition per stim region as bars
        (:func:`plot_response_ratio_by_region`);
      * ``fig_glass``             — the MNI152 glass brain (``res['fig_glass']``).

    ``snr_window`` MUST match the one used to build ``res`` for the grid tallies to
    agree with the clustering. ``nonvisual_kwargs`` forwards trim knobs to the
    non-visual grid (e.g. ``{'drop_labels': (...), 'min_denom': 1}``).
    """

    fig_visual, grid_data, res = plot_response_ratio_grid(
        recording_targets=electrodes, res=res, transpose=transpose,
        snr_window=snr_window, atlas=atlas, surf_opacity=surf_opacity, show=show)
    if show:
        _print_region_cluster_counts(grid_data, res)

    fig_nonvisual, nonvis_data = _nonvisual_grid(
        res, electrodes, transpose=transpose, snr_window=snr_window,
        atlas=atlas, surf_opacity=surf_opacity, show=show, **(nonvisual_kwargs or {}))

    constituents = plot_cluster_constituents(res, show_axes=True, show=show)
    by_region = plot_response_ratio_by_region(res, show=show)

    fig_glass = res.get("fig_glass")
    if show and fig_glass is not None:
        fig_glass.show()

    return dict(res=res, fig_visual=fig_visual, fig_nonvisual=fig_nonvisual,
                fig_constituents=constituents["fig"],
                fig_response_by_region=by_region["fig"], fig_glass=fig_glass,
                grid_data=grid_data, nonvis_data=nonvis_data)


def group_bpc_analysis_kmeans(electrodes, *, n_dims: int = 7, n_clusters: int = 2,
                              snr_window: tuple = (0.5, 10), random_state: int = 0,
                              transpose: bool = True, atlas: str = "visual",
                              surf_opacity: float = 0.4, show: bool = True,
                              res: dict | None = None,
                              nonvisual_kwargs: dict | None = None) -> dict:
    """Full group KMeans BPC analysis: :func:`group_bpc_calc` then
    :func:`group_bpc_plot`, in one call.

    Returns the :func:`group_bpc_plot` dict (``res`` + every figure). Pass a
    pre-built ``res`` to skip re-clustering. Parameters as in :func:`group_bpc_calc`
    / :func:`group_bpc_plot`.
    """
    if res is None:
        res = group_bpc_calc(electrodes, n_dims=n_dims, n_clusters=n_clusters,
                             snr_window=snr_window, random_state=random_state,
                             atlas=atlas, surf_opacity=surf_opacity, show=False)
    return group_bpc_plot(res, electrodes, transpose=transpose, atlas=atlas,
                          surf_opacity=surf_opacity, snr_window=snr_window,
                          show=show, nonvisual_kwargs=nonvisual_kwargs)