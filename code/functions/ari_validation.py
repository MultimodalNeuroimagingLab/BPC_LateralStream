"""Consensus-clustering stability via the Adjusted Rand Index (ARI).

Extracted from ``consensus_curve_validation.ipynb`` so the ARI machinery lives in one
importable place. Two groups of entry points:

**Core helpers** (leave-one-patient-out matrix, as in the notebook)
  * :func:`element_labels` — ``res`` -> ``{(patient, contact, local_bpc): cluster}``.
  * :func:`run_partition` — run ``bg.group_bpc_analysis_kmeans`` with one patient held out.
  * :func:`ari_common` — ARI between two partitions over their shared elements.
  * :func:`consensus_ari_matrix` — patient x patient ARI + shared-count matrices.
  * :func:`plot_ari_matrix` — heatmap of that matrix.

**Three-level perturbation validation** (new)
  * :func:`ari_three_level_validation` — re-cluster after (1) dropping each patient,
    (2) dropping each lateral subregion (LO1/LO2/TO1/TO2), and (3) dropping N random
    groups of K electrodes; score each level by the pairwise ARI AMONG its own
    perturbed partitions and draw the ARI scatter.
  * :func:`plot_ari_scatter` — the ARI scatter figure (one column of pairwise-ARI
    points per level). ``plot_ari_boxplots`` is a backward-compat alias.

Every clustering call goes through ``bg.group_bpc_analysis_kmeans`` with
``show=False`` and ``plt.close('all')`` so memory stays flat across the many runs.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import adjusted_rand_score

import functions.bpc_group as bg


# --------------------------------------------------------------------------- #
# core helpers (verbatim from consensus_curve_validation.ipynb)
# --------------------------------------------------------------------------- #
def element_labels(res):
    """Map every clustered pooled BPC to its cluster label.

    Returns ``{(patient, contact, local_bpc_index): cluster_label}``. The key is a
    stable identity for one electrode-level BPC, so labels from different runs line
    up. Curves that were degenerate / SNR-gated have ``row_map == -1`` and are
    omitted, so only BPCs assigned to a real cluster appear. ``res['labels'][gr]`` is
    the cluster of global pooled row ``gr``.
    """
    labels = np.asarray(res["labels"])
    out = {}
    for patient, contact, _pairs, _bpc_pairs, row_map in res["contacts_info"]:
        rm = np.asarray(row_map, int)
        for j, gr in enumerate(rm):
            if gr >= 0:
                out[(patient, contact, int(j))] = int(labels[gr])
    return out


def _partition(electrodes, *, analysis_fn=None, **cluster_kw):
    """Cluster an electrode selection and return its ``element_labels`` partition.

    ``analysis_fn`` is the group-analysis entry point to run (default
    :func:`bpc_group.group_bpc_analysis_kmeans`). Pass ``n_clusters=`` /
    ``n_dims=`` / ``snr_window=`` in ``cluster_kw``. Figures are suppressed and
    closed so memory stays flat across many runs.
    """
    fn = analysis_fn or bg.group_bpc_analysis_kmeans
    out = fn(electrodes, show=False, **cluster_kw)
    plt.close("all")
    return element_labels(out["res"])


def run_partition(electrodes, exclude=None, *, analysis_fn=None, **cluster_kw):
    """Run the group analysis on ``electrodes`` with one patient held out.

    Returns ``{element_id: cluster_label}`` for the resulting partition. ``exclude
    =None`` uses the full cohort. ``analysis_fn`` defaults to KMeans
    (:func:`bpc_group.group_bpc_analysis_kmeans`) — see :func:`_partition`. Figures
    are suppressed (``show=False``) and closed so memory stays flat across many runs.
    """
    sub = {p: v for p, v in electrodes.items() if p != exclude}
    return _partition(sub, analysis_fn=analysis_fn, **cluster_kw)


def ari_common(labels_a, labels_b):
    """ARI between two partitions, scored only on the elements they share.

    Returns ``(ari, n_common)``. ARI needs >= 2 shared elements, else ``NaN``.
    """
    common = sorted(set(labels_a) & set(labels_b))
    if len(common) < 2:
        return np.nan, len(common)
    a = [labels_a[k] for k in common]
    b = [labels_b[k] for k in common]
    return adjusted_rand_score(a, b), len(common)


def _pairwise_ari(partitions):
    """All ``C(m, 2)`` pairwise ARIs *among* a level's partitions (not vs a reference).

    ``partitions`` is a list of ``element_labels`` dicts. Returns ``(ari, n, pairs)``
    where ``ari[t]`` / ``n[t]`` are :func:`ari_common` for the two partitions in
    ``pairs[t] = (i, j)`` (``i < j``). A level with < 2 partitions yields empty arrays.
    """
    aris, ns, pairs = [], [], []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            a, nc = ari_common(partitions[i], partitions[j])
            aris.append(a); ns.append(nc); pairs.append((i, j))
    return np.array(aris, float), np.array(ns, int), pairs


def consensus_ari_matrix(loo_labels, patients):
    """Patient x patient ARI matrix + shared-element counts.

    ``loo_labels[p]`` is the partition (an ``element_labels`` dict) from leaving
    patient ``p`` out. ``M[i, j]`` = ARI(leave-``patients[i]``-out,
    leave-``patients[j]``-out); ``N[i, j]`` = number of shared elements. Both are
    symmetric; the diagonal of ``M`` is 1.
    """
    n = len(patients)
    M = np.full((n, n), np.nan)
    N = np.zeros((n, n), int)
    for i in range(n):
        for j in range(i, n):
            ari, nc = ari_common(loo_labels[patients[i]], loo_labels[patients[j]])
            M[i, j] = M[j, i] = ari
            N[i, j] = N[j, i] = nc
    return M, N


def plot_ari_matrix(M, N, patients, *, cmap="viridis",
                    title="Leave-one-patient-out clustering agreement (ARI)",
                    show=True):
    """Heatmap of the patient x patient ARI matrix.

    Each cell is annotated with the ARI and the number of shared elements. The color
    scale spans the off-diagonal values (the diagonal is 1 by construction) so the
    between-partition variation is visible.
    """
    labels = [p[-4:] for p in patients]
    n = len(patients)
    off = M[~np.eye(n, dtype=bool)]
    off = off[np.isfinite(off)]
    vmin = float(np.floor(off.min() * 20) / 20) if off.size else 0.0

    fig, ax = plt.subplots(figsize=(1.15 * n + 2.0, 1.15 * n + 1.5))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=1.0)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("held-out patient"); ax.set_ylabel("held-out patient")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="0.5")
                continue
            r, g, b, _ = im.cmap(im.norm(v))
            tc = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}\n(n={N[i, j]})", ha="center", va="center",
                    color=tc, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="adjusted Rand index")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


# --------------------------------------------------------------------------- #
# three-level perturbation validation
# --------------------------------------------------------------------------- #
def _flatten_electrodes(electrodes):
    """Flatten a ``{patient: {subregion: [elec, ...]}}`` (or ``{patient: [elec,...]}``)
    selection to a list of ``(patient, subregion, electrode)`` triples."""
    out = []
    for p, d in electrodes.items():
        if isinstance(d, dict):
            for s, elecs in d.items():
                for e in elecs:
                    out.append((p, s, e))
        else:                                                  # flat per-patient list
            for e in d:
                out.append((p, None, e))
    return out


def _drop_subregion(electrodes, subregion):
    """Copy of ``electrodes`` with every electrode in ``subregion`` (case-insensitive
    visualArea key) removed across all patients; patients left empty are dropped."""
    key = str(subregion).strip().lower()
    out = {}
    for p, d in electrodes.items():
        nd = {s: list(v) for s, v in d.items() if str(s).strip().lower() != key}
        if nd:
            out[p] = nd
    return out


def _drop_electrodes(electrodes, remove):
    """Copy of ``electrodes`` with the given ``(patient, subregion, electrode)`` triples
    removed; empty subregions and empty patients are pruned."""
    remove = set(remove)
    out = {}
    for p, d in electrodes.items():
        nd = {}
        for s, elecs in d.items():
            keep = [e for e in elecs if (p, s, e) not in remove]
            if keep:
                nd[s] = keep
        if nd:
            out[p] = nd
    return out


def ari_three_level_validation(
    electrodes,
    *,
    subregions=("LO1", "LO2", "TO1", "TO2"),
    n_random=20,
    random_size=10,
    seed=0,
    show=True,
    analysis_fn=None,
    **cluster_kw,
):
    """Consensus-clustering stability against three kinds of cohort perturbation.

    The clustering is re-run on three families of perturbed cohorts, and within each
    family every perturbed partition is scored against the OTHER perturbed partitions
    of the same family — the full set of pairwise ARIs (:func:`_pairwise_ari`), over
    the elements each pair shares. (This measures how much the perturbations agree with
    *each other*, not with the unperturbed full-cohort clustering.)

    1. **Leave-one-patient-out** — drop each patient in turn.
    2. **Leave-one-subregion-out** — drop every electrode of each lateral subregion
       (``subregions``, default LO1/LO2/TO1/TO2) across all patients.
    3. **Random −K electrodes** — drop ``n_random`` independent random groups of
       ``random_size`` electrodes (sampled without replacement within each group).

    A reference clustering on the FULL selection is still computed and returned under
    ``full`` for convenience, but it no longer enters the ARI scoring.

    Parameters
    ----------
    electrodes : the nested ``{patient: {subregion: [electrode, ...]}}`` selection from
        :func:`anatomy_dist.electrodes_by_region` (e.g. ``electrodes_by_region('Lateral')``).
    subregions : lateral visualArea labels to leave out one at a time.
    n_random, random_size : number of random draws and electrodes removed per draw.
    seed : RNG seed for the random draws (reproducible).
    show : draw the ARI scatter (:func:`plot_ari_scatter`).
    analysis_fn : group-analysis entry point (default
        :func:`bpc_group.group_bpc_analysis_kmeans`).
    **cluster_kw : forwarded verbatim to ``analysis_fn`` — the SAME knobs as the
        reference group run, e.g. ``n_dims=7, snr_window=(0.5, 10), n_clusters=2``.

    Returns
    -------
    dict with ``full`` (reference partition, not used for scoring), ``cluster_kw``,
    ``fig`` (or ``None``), and one entry per level:
      * ``patient``   — ``dict(ids=[patient...], partitions=[...], ari, n, pairs)``
      * ``subregion`` — ``dict(ids=[sub...],     partitions=[...], ari, n, pairs)``
      * ``random``    — ``dict(ids=[...], partitions=[...], ari, n, pairs,
        removed=[[(p,s,e)...], ...], size=K)``
    where ``ari[t]`` / ``n[t]`` is the pairwise ARI / shared-element count between the
    two partitions in ``pairs[t] = (i, j)`` (indices into that level's ``partitions``).
    """
    rng = np.random.default_rng(seed)
    full = run_partition(electrodes, None, analysis_fn=analysis_fn, **cluster_kw)  # reference only

    # 1. leave-one-patient-out ------------------------------------------------
    patients = sorted(electrodes)
    pat_parts = [run_partition(electrodes, p, analysis_fn=analysis_fn, **cluster_kw)
                 for p in patients]

    # 2. leave-one-subregion-out ---------------------------------------------
    sub_ids = list(subregions)
    sub_parts = [_partition(_drop_subregion(electrodes, s), analysis_fn=analysis_fn, **cluster_kw)
                 for s in sub_ids]

    # 3. random −K electrodes -------------------------------------------------
    all_elec = _flatten_electrodes(electrodes)
    k = min(int(random_size), len(all_elec))
    rand_parts, removed_sets = [], []
    for r in range(int(n_random)):
        idx = rng.choice(len(all_elec), size=k, replace=False)
        rem = [all_elec[i] for i in idx]
        rand_parts.append(_partition(_drop_electrodes(electrodes, rem),
                                     analysis_fn=analysis_fn, **cluster_kw))
        removed_sets.append(rem)

    # score each level by the pairwise ARI AMONG its own perturbed partitions
    def _pack(ids, parts, tag):
        ari, n, pairs = _pairwise_ari(parts)
        vf = ari[np.isfinite(ari)]
        summ = (f"mean={vf.mean():.3f}  min={vf.min():.3f}  max={vf.max():.3f}"
                if vf.size else "no comparable pairs")
        print(f"[{tag:<9}] {len(parts)} partitions -> {ari.size} pairwise ARI  ({summ})")
        return dict(ids=ids, partitions=parts, ari=ari, n=n, pairs=pairs)

    rand_ids = [f"draw{r + 1}" for r in range(len(rand_parts))]
    results = {
        "full": full,
        "cluster_kw": dict(cluster_kw),
        "patient":   _pack(patients, pat_parts, "patient"),
        "subregion": _pack(sub_ids,  sub_parts, "subregion"),
        "random":    {**_pack(rand_ids, rand_parts, "random"),
                      "removed": removed_sets, "size": k},
    }
    results["fig"] = plot_ari_scatter(results) if show else None
    return results


def plot_ari_scatter(results, *, show=True, seed=0):
    """ARI scatter (Plotly) for :func:`ari_three_level_validation`.

    One column per perturbation level (patient / subregion / random) on a shared ARI
    axis. Each point is one PAIRWISE ARI between two perturbed partitions of that level
    (jittered horizontally since the values cluster). ``results`` is the dict returned
    by :func:`ari_three_level_validation`. ``seed`` seeds the horizontal jitter
    (reproducible). Returns a ``plotly.graph_objects.Figure``.
    """
    panels = [
        ("patient",   "Leave one patient out",   "102,194,165"),   # muted teal-green    #66c2a5
        ("subregion", "Leave one subregion out", "252,141,98"),    # soft coral-orange   #fc8d62
        ("random",    f"Random {results['random']['size']} electrodes", "141,160,203"),  # periwinkle #8da0cb
    ]
    rng = np.random.default_rng(seed)
    fig = go.Figure()
    ymin, tickvals, ticktext = 1.0, [], []
    for x, (key, title, rgb) in enumerate(panels):
        v = results[key]["ari"]
        vf = v[np.isfinite(v)]
        n_nan = int(v.size - vf.size)
        if vf.size:
            ymin = min(ymin, float(vf.min()))
            jit = (rng.random(vf.size) - 0.5) * 0.30
            fig.add_trace(go.Scatter(
                x=np.full(vf.size, x, float) + jit, y=vf, mode="markers", name=title,
                marker=dict(color=f"rgba({rgb},0.75)", size=12,
                            line=dict(width=0.5, color="black")),
                showlegend=False, hovertemplate="ARI=%{y:.3f}<extra></extra>"))
        tickvals.append(x)
        ticktext.append(f"{title}<br>(N={vf.size} pairs"
                        f"{f', +{n_nan} n/a' if n_nan else ''})")

    fig.add_hline(y=1.0, line=dict(color="lightgray", dash="dot"))  # identical-clustering ref
    fig.update_layout(
        template="plotly_white", showlegend=False,
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext,
                   showticklabels=True, tickfont=dict(size=14), automargin=True,
                   range=[-0.5, len(panels) - 0.5]),
        yaxis=dict(title="pairwise ARI (perturbation vs perturbation)",
                   range=[min(0.0, ymin) - 0.03, 1.03]),
        width=720, height=430, margin=dict(b=90))
    if show:
        fig.show()
    return fig


plot_ari_boxplots = plot_ari_scatter          # backward-compat alias (now a scatter)
