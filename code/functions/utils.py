"""
utils.py
========
One import surface for the cross-cutting BPC utilities.

  * :func:`electrodes_by_region` — region -> ``{patient: {area: [contacts]}}`` (defined here)
  * :func:`bpc_constituents` / :func:`plot_bpc_matrix` — re-exported from ``bpc_group``
    (tidy per-connection table from a group ``res``, and its per-BPC matrix figure)

Defined here (cached-output helpers + two figure builders lifted from
``fetch_individual_bpcs.ipynb``):
  * :func:`load_bpc` / :func:`snr_by_bpc` — read one contact's saved ``.npz`` outputs
  * :func:`plot_group_bpc_snr` — group figure: BPC curves (top) + per-pair SNR
    distribution (bottom), each BPC's mean SNR written on the distribution's x-axis
  * :func:`show_pipeline_figures` — fetch and show the full single-electrode
    ``simple_bpc_pipeline`` figure set for one (patient, contact) from cache

Voltage traces (from ``plot_voltage_elec_run.ipynb``):
  * :func:`extract_pair_traces` / :func:`plot_stim_traces` — ONE (contact, stim pair):
    every single-trial voltage trace + the mean (Plotly)
  * :func:`stim_pair_traces` / :func:`plot_stim_pair_traces` — ONE stim pair, the
    trial-averaged trace of each recording electrode (Plotly)

CCEP aura animations (the whole ``aura_animation`` module was folded in here):
  * :func:`animate_stim_response_interactive` — interactive Plotly aura (base env)
  * :func:`animate_stim_response` — PyVista GIF (needs the ``code/.venv-aura`` venv)
    One stim pair's evoked response drawn as distance-decaying polarity auras on the
    subject's native (ACPC mm) surfaces; the companion voltage plot is
    :func:`plot_stim_pair_traces`.
"""

from __future__ import annotations

from collections import defaultdict
import glob
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree

import functions.load_preproc as lp
from functions.colors import bpc_color
from functions.save_outputs import load_contact, OUTPUTS_ROOT
from functions.spatial_plot import _load_electrodes, _load_gifti, _snap_to_inflated

# bpc_constituents / plot_bpc_matrix are defined in bpc_group; expose them on this
# single utility surface via LAZY wrappers so importing utils never triggers a
# bpc_group import at load time. (bpc_group imports the loc_info readers below FROM
# utils, so a top-level import here would create a cycle.)
def bpc_constituents(*args, **kwargs):
    """Per-connection constituents table from a group ``res``. See ``bpc_group``."""
    from functions.bpc_group import bpc_constituents as _f
    return _f(*args, **kwargs)


def plot_bpc_matrix(*args, **kwargs):
    """Per-BPC matrix figure from a constituents DataFrame. See ``bpc_group``."""
    from functions.bpc_group import plot_bpc_matrix as _f
    return _f(*args, **kwargs)


# --------------------------------------------------------------------------- #
# loc_info readers (moved here from bpc_group — low-level, depend only on
# load_preproc; electrodes_by_region + the group analysis both use them)
# --------------------------------------------------------------------------- #
# CSVs written by code/external/export_loc_info_csv.m + export_loc_info_full_csv.m
_LOC_CSV_DIR = lp.OUTPUTS_ROOT / "loc_info_csv"
_EXPORTER_M = lp.OUTPUTS_ROOT.parent / "external" / "export_loc_info_csv.m"        # loc_info_csv/
_EXPORTER_FULL_M = lp.OUTPUTS_ROOT.parent / "external" / "export_loc_info_full_csv.m"  # sub-*/loc_info.csv


def ensure_loc_info(force: bool = False) -> None:
    """(Re)generate the loc_info CSVs from the MATLAB ``loc_info.mat`` tables.

    ``loc_info.mat`` is an MCOS MATLAB table that Python cannot read, so every
    anatomy reader here (``load_loc_labels`` / ``_load_gm_wm`` /
    ``electrodes_by_region``) uses per-subject CSVs exported by MATLAB. Call
    this ONCE at the start of the pipeline to build them (needs MATLAB on PATH or in
    /Applications): it runs both exporters — ``export_loc_info_csv.m`` (pathway /
    visualArea) and ``export_loc_info_full_csv.m`` (adds ``gm_wm_relativeDistance``).
    ``force`` re-exports even if a first CSV already exists.
    """
    if not force and (_LOC_CSV_DIR.glob("sub-*_loc_info.csv") and
                      any(_LOC_CSV_DIR.glob("sub-*_loc_info.csv"))):
        print("loc_info CSVs already present (pass force=True to re-export).")
        return
    matlab = _find_matlab()
    if matlab is None:
        raise FileNotFoundError("MATLAB not found; cannot export the loc_info CSVs.")
    for exporter in (_EXPORTER_M, _EXPORTER_FULL_M):
        if not exporter.exists():
            print(f"  [skip] {exporter.name} not found")
            continue
        print(f"  running {exporter.name} ...")
        subprocess.run([matlab, "-batch", f"addpath('{exporter.parent}'); {exporter.stem}"],
                       cwd=str(lp.PROJECT_ROOT), check=False)
    print("loc_info CSVs generated.")


def _find_matlab() -> str | None:
    """Best-effort locate the MATLAB executable (PATH or /Applications)."""
    from shutil import which
    m = which("matlab")
    if m:
        return m
    cands = sorted(glob.glob("/Applications/MATLAB_*.app/bin/matlab"), reverse=True)
    return cands[0] if cands else None


def _ensure_loc_csv(patient: str) -> Path:
    """Path to ``sub-<patient>_loc_info.csv``; regenerate via MATLAB if absent."""
    csv = _LOC_CSV_DIR / f"sub-{patient}_loc_info.csv"
    if csv.exists():
        return csv
    matlab = _find_matlab()
    if matlab and _EXPORTER_M.exists():
        print(f"loc_info CSV for {patient} missing -- running {_EXPORTER_M.name} ...")
        subprocess.run(
            [matlab, "-batch",
             f"addpath('{_EXPORTER_M.parent}'); export_loc_info_csv"],
            cwd=str(lp.PROJECT_ROOT), check=False,
        )
    if csv.exists():
        return csv
    raise FileNotFoundError(
        f"No loc_info CSV for {patient} at {csv}. Generate the CSVs once by "
        f"running MATLAB:\n    addpath('code/external'); export_loc_info_csv\n"
        f"(loc_info.mat is an MCOS table that Python cannot read directly)."
    )


def load_loc_labels(patient: str, column: str) -> dict[str, str]:
    """``{electrode_name: label}`` for one subject's ``column`` of loc_info.

    Blank / missing labels map to ``''`` (the caller turns those into the
    ``blank_label`` bucket). ``column`` is 'pathway' or 'visualArea'.
    """
    csv = _ensure_loc_csv(patient)
    df = pd.read_csv(csv)
    if column not in df.columns:
        raise KeyError(
            f"column {column!r} not in {csv.name} (have {list(df.columns)}). "
            f"Use 'pathway' or 'visualArea'."
        )
    out: dict[str, str] = {}
    for name, lab in zip(df["name"].astype(str), df[column]):
        s = "" if (lab is None or (isinstance(lab, float) and np.isnan(lab))) else str(lab).strip()
        out[name.strip()] = s
    return out


def load_destrieux_labels(patient: str, column: str = "Destrieux_label_text") -> dict[str, str]:
    """``{electrode_name: Destrieux/aseg parcel}`` for one subject, from the raw
    ``electrodes.tsv`` (loc_info's CSV export carries only pathway / visualArea).

    ``column`` is the electrodes.tsv column to read (default the human-readable
    ``Destrieux_label_text``; pass ``'Destrieux_label'`` for the numeric code).
    For depth sEEG this mixes true Destrieux cortical parcels (``lh_G_...``) with
    FreeSurfer aseg labels for white-matter / subcortical / CSF contacts
    (``Right_Hippocampus``, ``Left_Cerebral_White_Matter``, ...). Blank / ``n/a``
    entries map to ``''`` (the caller buckets or drops those). Used to break the
    'non-visual area' bucket into anatomy — see
    ``bpc_group.plot_nonvisual_destrieux_grid``.
    """
    df = pd.read_csv(lp.electrodes_tsv(patient), sep="\t", dtype=str)
    if column not in df.columns:
        raise KeyError(
            f"column {column!r} not in electrodes.tsv for {patient} "
            f"(have {list(df.columns)}).")
    out: dict[str, str] = {}
    for name, lab in zip(df["name"].astype(str), df[column]):
        s = "" if lab is None else str(lab).strip()
        if s.lower() in ("n/a", "nan", "none"):
            s = ""
        out[name.strip()] = s
    return out


def _load_gm_wm(patient: str) -> dict[str, float]:
    """``{electrode: gm_wm relative distance (mm)}`` from the per-patient
    ``outputs/sub-<patient>/loc_info.csv`` (column ``gm_wm_relativeDistance``).

    Positive = gray/superficial, negative = white matter, ~0 = GM/WM border. This
    is the loc_info source of truth (exported from loc_info.mat) and covers every
    subject — unlike the FreeSurfer-pipeline ``gm_wmDistance`` derivative, which
    only exists for a few. Values are identical to that derivative where both
    exist. Empty dict if the file/column is missing."""
    f = lp.OUTPUTS_ROOT / f"sub-{patient}" / "loc_info.csv"
    if not f.exists():
        return {}
    g = pd.read_csv(f, na_values=["n/a", "N/A"])
    if "name" not in g.columns or "gm_wm_relativeDistance" not in g.columns:
        return {}
    return {str(n).strip(): float(d)
            for n, d in zip(g["name"], g["gm_wm_relativeDistance"]) if pd.notna(d)}

# --------------------------------------------------------------------------- #
# electrodes_by_region 
# --------------------------------------------------------------------------- #
def electrodes_by_region(
    region,
    patients=None,
    *,
    level: str = "pathway",
    subdivide: str | None = "visualArea",
    gm_wm: str | None = None,
    rel_dist: tuple | None = None,
    require_bpc: bool = False,
    as_dataframe: bool = False,
):
    """All recording electrodes located in a region of interest, per patient,
    subdivided by a finer loc_info label.

    Looks every electrode up in the per-subject loc_info CSVs (``name`` / ``pathway``
    / ``visualArea``) and keeps those whose ``level`` column matches ``region``,
    grouped within each patient by ``subdivide``.

    Parameters
    ----------
    region : value(s) to match in the ``level`` column — e.g. ``'Ventral'`` (a
        pathway; the default level) or ``['FG1','FG2']`` with ``level='visualArea'``.
        A string or a list/set; matching is case-insensitive exact.
    patients : subject ids to scan (default: every subject with a loc_info CSV)
    level : loc_info column ``region`` is matched against — 'pathway' (default) or
        'visualArea'.
    subdivide : loc_info column to group the matched electrodes by within each
        patient — 'visualArea' (default) or 'pathway', or ``None`` for a flat list.
    gm_wm : optional gray/white-matter filter on the recording electrode, using its
        gm_wm RelativeDistance (positive = gray/superficial, negative = white matter,
        ~0 = GM/WM border). ``'gm'``/``'gray'`` keeps only electrodes with
        RelativeDistance > 0; ``'wm'``/``'white'`` keeps only < 0; ``None`` (default)
        keeps all. Electrodes with no gm_wm distance are dropped when a filter is set.
    rel_dist : an explicit inclusive ``(low, high)`` window on the gm_wm
        RelativeDistance — keep electrodes with ``low <= RelativeDistance <= high``.
        Either bound may be ``None`` for an open end, e.g. ``(None, -1)`` = "clearly
        WM" (``<= -1``), ``(1, None)`` = "clearly GM" (``>= 1``), ``(-0.5, 0.5)`` =
        near the border. Supersedes the ``gm_wm`` shorthand and cannot be combined
        with it. Electrodes with no gm_wm distance are dropped when ``rel_dist`` is
        set. ``None`` (default) = no range filter.
    require_bpc : if True, keep only electrodes that have saved BPC outputs
        (``outputs/sub-<p>/<contact>/bpcs_pc1.npz``) — i.e. ones you actually have a
        BPC for. Default False (pure loc_info listing).
    as_dataframe : return a tidy DataFrame (``patient, electrode, pathway,
        visualArea, gm_wm_dist``) instead of the nested dict.

    Returns
    -------
    nested dict ``{patient: {sub_label: [electrode, ...]}}`` (sorted; electrodes
    with the region but a blank ``subdivide`` go under ``'(unlabeled)'``), or a
    DataFrame if ``as_dataframe``.
    """

    targets = ({region.strip().lower()} if isinstance(region, str)
               else {str(r).strip().lower() for r in region})
    gw_mode = None
    if gm_wm is not None:
        if rel_dist is not None:
            raise ValueError("pass either gm_wm or rel_dist, not both")
        s = str(gm_wm).strip().lower()
        if s in ("gm", "gray", "grey", "g"):
            gw_mode = "gm"
        elif s in ("wm", "white", "w"):
            gw_mode = "wm"
        else:
            raise ValueError("gm_wm must be 'gm'/'gray' or 'wm'/'white' (or None)")
    rd_lo, rd_hi = None, None
    if rel_dist is not None:
        if not (isinstance(rel_dist, (tuple, list)) and len(rel_dist) == 2):
            raise ValueError("rel_dist must be a (low, high) pair; use None for an open bound")
        rd_lo, rd_hi = rel_dist
    if patients is None:
        patients = sorted(f.name[len("sub-"):-len("_loc_info.csv")]
                          for f in _LOC_CSV_DIR.glob("sub-*_loc_info.csv"))
        patients = [p for p in patients]

    out: dict[str, dict] = {}
    records: list[tuple] = []
    for p in patients:
        pathmap = load_loc_labels(p, "pathway")
        vamap = load_loc_labels(p, "visualArea")
        gw_active = gw_mode is not None or rel_dist is not None
        gw = _load_gm_wm(p) if (gw_active or as_dataframe) else {}
        lvl = pathmap if level == "pathway" else vamap
        submap = (pathmap if subdivide == "pathway" else vamap) if subdivide else {}
        grp: dict[str, list] = {}
        for name, val in lvl.items():
            if val.strip().lower() not in targets:
                continue
            if require_bpc and not (OUTPUTS_ROOT / f"sub-{p}" / name / "bpcs_pc1.npz").exists():
                continue
            rd = gw.get(name, float("nan"))
            if gw_active:                                      # gray/white-matter / rel_dist filter
                if not np.isfinite(rd):
                    continue                                   # no distance -> drop
                if gw_mode == "gm" and not rd > 0:
                    continue
                if gw_mode == "wm" and not rd < 0:
                    continue
                if rd_lo is not None and rd < rd_lo:           # inclusive RelativeDistance window
                    continue
                if rd_hi is not None and rd > rd_hi:
                    continue
            key = (submap.get(name, "").strip() or "(unlabeled)") if subdivide else "all"
            grp.setdefault(key, []).append(name)
            records.append((p, name, pathmap.get(name, "").strip(),
                            vamap.get(name, "").strip(), rd))
        out[p] = {k: sorted(v) for k, v in sorted(grp.items())}

    if as_dataframe:
        return pd.DataFrame(records, columns=["patient", "electrode", "pathway",
                                              "visualArea", "gm_wm_dist"]).sort_values(
            ["patient", "visualArea", "electrode"], ignore_index=True)
    return out


# --------------------------------------------------------------------------- #
# cached-output helpers
# --------------------------------------------------------------------------- #
def load_bpc(patient: str, contact: str, pc: int = 1) -> dict | None:
    """Cached pipeline outputs for one (patient, contact), or ``None`` if nothing is
    saved. Returns a flat dict: ``times, V, stim_sites, pairs, P, tmat, H, W, Bs,
    bpc_pairs, plotweights, p_vals`` (missing stems come back as ``None``)."""
    try:
        out = load_contact(patient, contact)
    except FileNotFoundError:
        return None
    bkey, skey = f"bpcs_pc{pc}", f"stats_pc{pc}"
    if bkey not in out:
        return None
    b, s = out[bkey], out.get(skey, {})
    d = dict(patient=patient, contact=contact, pc=pc,
             times=b["times"], Bs=b["Bs"],
             bpc_pairs=b["bpc_pairs"], pairs=b["pairs"],
             plotweights=s.get("plotweights"), p_vals=s.get("p_vals"))
    for stem, keys in (("convergent", ("V", "stim_sites")),
                       ("projection", ("P",)),
                       ("significance", ("tmat",)),
                       ("nmf", ("H", "W"))):
        for k in keys:
            d[k] = out[stem][k] if stem in out and k in out[stem] else None
    return d


def snr_by_bpc(d: dict) -> dict:
    """``{bpc_index: per-pair SNR array}`` for the pairs assigned to each BPC (NaNs
    dropped). ``d`` is a :func:`load_bpc` bundle."""
    if d is None or d["plotweights"] is None:
        return {}
    bp, pw = np.asarray(d["bpc_pairs"], float), np.asarray(d["plotweights"], float)
    out = {}
    for k in range(d["Bs"].shape[0]):
        v = pw[bp == k]
        out[k] = v[~np.isnan(v)]
    return out


def _region_contacts(region: str, patient: str | None) -> list[tuple[str, str]]:
    """De-duplicated ``[(patient, contact), ...]`` for a region (optionally one patient)."""
    reg = electrodes_by_region(region)
    items = set()
    for p, sub in reg.items():
        if patient is not None and p != patient:
            continue
        for contacts in sub.values():
            for c in contacts:
                items.add((p, c))
    return sorted(items)


# --------------------------------------------------------------------------- #
# group figure: BPC curves + per-pair SNR distribution
# --------------------------------------------------------------------------- #
def plot_group_bpc_snr(region: str = "Lateral", patient: str | None = None, *,
                       pc: int = 1, xlim: tuple = (0.0, 0.5), jitter: float = 0.18,
                       seed: int = 0, anonymize: bool = False,
                       show: bool = True) -> go.Figure:
    """Group-level figure (Plotly), two rows colored **by electrode** (BPC indices
    are arbitrary across electrodes):

      * **top**    — every electrode-level BPC curve in the region;
      * **bottom** — the per-pair SNR *distribution* of each (electrode, BPC) as a
        jittered strip (no box), one column per (electrode, BPC) labeled
        ``"BPC k, <electrode>"`` with its **mean SNR**.

    Reads the saved per-contact outputs (nothing is recomputed). ``patient=None``
    pools every patient with electrodes in ``region``; pass a subject id to restrict.
    """
    import plotly.colors as pcolors

    rng = np.random.default_rng(seed)
    items = []                              # one entry per (electrode, BPC) with pairs
    for p, c in _region_contacts(region, patient):
        d = load_bpc(p, c, pc=pc)
        if d is None:
            continue
        t = np.asarray(d["times"])
        by_k = snr_by_bpc(d)
        for k in range(d["Bs"].shape[0]):
            v = np.asarray(by_k.get(k, []), float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            items.append(dict(elec=c, k=int(k), times=t, curve=d["Bs"][k], snr=v))
    if not items:
        raise ValueError(f"no cached BPCs found for region {region!r}"
                         + (f" / patient {patient}" if patient else ""))

    # one color per electrode
    elecs = sorted({it["elec"] for it in items})
    if len(elecs) > 1:
        pal = pcolors.sample_colorscale("turbo", [i / (len(elecs) - 1) for i in range(len(elecs))])
    else:
        pal = ["#1f77b4"]
    ecolor = {e: pal[i] for i, e in enumerate(elecs)}

    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.14,
                        subplot_titles=("Basis profile curves", "Per-pair SNR distribution"))

    seen = set()
    for it in items:                                        # row 1 — curves colored by electrode
        e = it["elec"]
        fig.add_trace(go.Scatter(
            x=it["times"], y=it["curve"], mode="lines",
            line=dict(color=ecolor[e], width=2.6), opacity=0.85,
            name=e, legendgroup=e, showlegend=(e not in seen),
            hovertemplate=f"{e} · BPC {it['k']}<br>%{{x:.3f}} s<br>%{{y:.3g}}<extra></extra>"),
            row=1, col=1)
        seen.add(e)
    fig.add_hline(y=0, line=dict(color="lightgray", width=1), row=1, col=1)

    xtickvals, xticktext = [], []
    for i, it in enumerate(items):                          # row 2 — one column per (electrode, BPC)
        v = it["snr"]
        x = i + rng.uniform(-jitter, jitter, size=v.size)
        fig.add_trace(go.Scatter(
            x=x, y=v, mode="markers",
            marker=dict(color=ecolor[it["elec"]], size=7, opacity=0.75,
                        line=dict(color="black", width=0.3)),
            legendgroup=it["elec"], showlegend=False,
            hovertemplate=f"{it['elec']} · BPC {it['k']}<br>SNR %{{y:.3f}}<extra></extra>"),
            row=2, col=1)
        fig.add_trace(go.Scatter(                           # mean marker (dash)
            x=[i - 0.3, i + 0.3], y=[v.mean(), v.mean()], mode="lines",
            line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"),
            row=2, col=1)
        xtickvals.append(i)
        xticktext.append(f"BPC {it['k']}, {it['elec']}<br>μ={v.mean():.2f}")

    fig.update_xaxes(title_text="time from stimulation (s)",
                     range=list(xlim) if xlim else None, row=1, col=1)
    fig.update_yaxes(title_text="voltage (unit-norm)", row=1, col=1)
    fig.update_xaxes(tickmode="array", tickvals=xtickvals, ticktext=xticktext,
                     tickangle=-45, row=2, col=1)
    fig.update_yaxes(title_text="SNR", row=2, col=1)
    title = f"{region} — group BPCs and per-pair SNR" + (
        f" ({patient})" if patient and not anonymize else "")
    fig.update_layout(title=title, template="plotly_white", height=820,
                      legend=dict(title="electrode"))
    if show:
        fig.show()
    return fig


# --------------------------------------------------------------------------- #
# fetch the whole single-electrode figure set from cache
# --------------------------------------------------------------------------- #
def show_pipeline_figures(patient: str, contact: str, *, hemisphere: str = "both",
                          pc: int = 1, convergent_kind: str = "heatmap",
                          xlim: tuple = (0.0, 0.5), spatial: bool = True,
                          anonymize: bool = False) -> dict:
    """Fetch one electrode's cached outputs and show the full ``simple_bpc_pipeline``
    figure set (Plotly), reusing the same step plotters: convergent matrix,
    projection ``P``, significance ``Ξ``, NMF ``H``, BPC curves, per-pair SNR, and the
    inflated-brain plot. Returns ``{name: go.Figure}``.

    ``anonymize=True`` drops the subject id from the inflated-brain title (the only
    figure in the set that shows it)."""
    from functions.plot_convergent import plot_convergent_matrix
    from functions.projection import plot_projection_matrix
    from functions.significance import plot_significance_matrix
    from functions.nmf import plot_nmf_H
    from functions.plot_bpcs import plot_bpcs
    from functions.bpc_stats import plot_plotweights
    from functions.spatial_plot import plot_bpcs_inflated
    from functions.load_preproc import has_freesurfer

    d = load_bpc(patient, contact, pc=pc)
    if d is None:
        raise FileNotFoundError(f"no cached outputs for {patient}/{contact}")

    figs: dict[str, go.Figure] = {}
    if d["V"] is not None:
        figs["convergent"] = plot_convergent_matrix(
            d["V"], d["times"], d["stim_sites"], kind=convergent_kind, show=True)
    if d["P"] is not None:
        figs["projection"] = plot_projection_matrix(d["P"], d["stim_sites"], show=True)
    if d["tmat"] is not None:
        figs["significance"] = plot_significance_matrix(d["tmat"], d["pairs"], show=True)
    if d["H"] is not None:
        figs["nmf"] = plot_nmf_H(d["H"], d["pairs"], show=True)
    figs["bpcs"] = plot_bpcs(d["Bs"], d["times"], xlim=xlim, show=True)
    if d["plotweights"] is not None:
        figs["snr"] = plot_plotweights(d["plotweights"], d["bpc_pairs"], d["pairs"],
                                       contact=contact, p_vals=d["p_vals"], show=True)
    if spatial and has_freesurfer(patient):
        hemi = contact[0].upper() if contact[:1].upper() in ("L", "R") else hemisphere
        f = plot_bpcs_inflated(patient, contact, bpc_pairs=d["bpc_pairs"],
                               pairs=d["pairs"], plotweights=d["plotweights"],
                               hemi=hemi, anonymize=anonymize)
        f.show()
        figs["inflated"] = f
    return figs


# --------------------------------------------------------------------------- #
# voltage traces 
# --------------------------------------------------------------------------- #
def _preproc_paths_for_pair(patient, stim_pair, dirname, runs):
    """All preproc .mat files matching ``stim_pair`` (across runs unless restricted)."""
    pdir = lp.preproc_dir(patient, dirname)
    hits = []
    for p in sorted(pdir.glob(f"*_{stim_pair}_preproc_run-*.mat")):
        info = lp.parse_filename(p.name)
        if info is None or info["pair"] != stim_pair:
            continue
        if runs is not None and info["run"] not in runs:
            continue
        hits.append(p)
    return hits


def extract_pair_traces(patient, contact, stim_pair, *, xlim=(-0.5, 0.5),
                        dirname=None, require_use_channel=False):
    """The ``(trials x time)`` voltage traces for ONE ``(contact, stim_pair)``.

    Loads only that stim pair's preproc file(s) (much faster than
    ``build_convergent_matrix``); trials are concatenated across runs. ``t = 0`` is
    stim onset. Returns ``(V (trials x time), times)``.
    """
    paths = sorted(lp.preproc_dir(patient, dirname).glob(
        f"{patient}_{stim_pair}_preproc_run-*.mat"))
    if not paths:
        avail = sorted({lp.parse_filename(p.name)["pair"]
                        for p in lp.preproc_dir(patient, dirname).glob(
                            f"{patient}_*_preproc_run-*.mat")})
        raise FileNotFoundError(f"no files for stim pair {stim_pair!r} at {patient}. "
                                f"available: {avail}")
    blocks, times = [], None
    for p in paths:
        run = lp.load_run(p, with_data=True)
        ci = lp.contact_data_row(run, contact, require_use_channel)   # offset-proof row
        if ci is None:
            continue
        tt = run["tt"]; win = (tt >= xlim[0]) & (tt <= xlim[1])
        blocks.append(run["data"][ci][win].T)                # (trials x n_time)
        if times is None:
            times = tt[win]
    if not blocks:
        raise ValueError(f"{contact} is not a usable recording row for {patient} "
                         f"{stim_pair} (try require_use_channel=False for bad/neighbor).")
    return np.vstack(blocks), times


def plot_stim_traces(patient, contact, stim_pair, *, dirname=None, xlim=(-0.5, 0.5),
                     require_use_channel=False,
                     trial_color="rgba(70,130,180,0.35)", anonymize=True,
                     show=True) -> go.Figure:
    """Every single-trial voltage trace at ``contact`` for ``stim_pair`` + their mean
    (black), Plotly. ``t = 0`` is stim onset. ``anonymize=True`` drops the subject id
    from the title."""
    V, t = extract_pair_traces(patient, contact, stim_pair, xlim=xlim,
                               dirname=dirname, require_use_channel=require_use_channel)
    fig = go.Figure()
    for i, tr in enumerate(V):
        fig.add_trace(go.Scatter(
            x=t, y=tr, mode="lines", line=dict(color=trial_color, width=1),
            legendgroup="trials", name="trials", showlegend=(i == 0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=t, y=V.mean(0), mode="lines", line=dict(color="black", width=2.5),
        name=f"mean (n={len(V)})"))
    fig.add_vline(x=0, line=dict(color="red", width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color="gray", width=0.5))
    _pt = "" if anonymize else f"{patient} — "
    fig.update_layout(
        title=f"{_pt}{contact} — stim {stim_pair} (n={len(V)} trials)",
        xaxis_title="time (s), 0 = stimulation", yaxis_title="voltage (µV)",
        xaxis=dict(range=list(xlim)), width=850, height=500, template="simple_white")
    if show:
        fig.show()
    return fig


def stim_pair_traces(patient, stim_pair, electrodes=None, tmin=-0.2, tmax=0.5,
                     dirname=None, runs=None, require_use_channel=True) -> dict:
    """Trial-averaged evoked response of each recording electrode to ``stim_pair``.

    Merges trials across runs, averages per electrode, clips to ``[tmin, tmax]``.
    ``electrodes=None`` uses every good recording channel for the pair. Returns
    ``{names, traces (n_elec x n_time), times, stim_pair, runs_used}``.
    """
    paths = _preproc_paths_for_pair(patient, stim_pair, dirname, runs)
    if not paths:
        raise FileNotFoundError(
            f"No preproc file for stim pair {stim_pair!r} in "
            f"{lp.preproc_dir(patient, dirname)} (check the label / dirname).")
    per_elec: dict = {}
    times, runs_used = None, []
    requested = None if electrodes is None else list(dict.fromkeys(electrodes))
    for path in paths:
        run = lp.load_run(path, with_data=True)
        runs_used.append(run["run"])
        tt = run["tt"]; win = (tt >= tmin) & (tt <= tmax)
        if times is None:
            times = tt[win]
        n_win = int(win.sum())
        wanted = requested if requested is not None else list(run["use_channels_names"])
        for name in wanted:
            ci = lp.contact_data_row(run, name, require_use_channel)
            if ci is None:
                continue
            block = run["data"][ci][win]
            if block.shape[0] != n_win:
                continue
            per_elec.setdefault(name, []).append(block.astype(np.float64))
    if not per_elec:
        raise ValueError(f"No usable recording channels for {stim_pair!r} "
                         f"(try require_use_channel=False).")
    names = [n for n in (requested or sorted(per_elec)) if n in per_elec]
    traces = np.array([np.concatenate(per_elec[n], axis=1).mean(axis=1) for n in names])
    return dict(names=names, traces=traces, times=times,
                stim_pair=stim_pair, runs_used=sorted(set(runs_used)))


def plot_stim_pair_traces(patient, stim_pair, electrodes=None, *, tmin=-0.2, tmax=0.5,
                          baseline=(-0.1, -0.01), vmax=None, dirname=None, runs=None,
                          require_use_channel=True, anonymize=False,
                          show=True) -> go.Figure:
    """One stim pair, the trial-averaged (baseline-subtracted) evoked trace of each
    recording electrode — one Plotly line per electrode. This is the companion plot
    the aura animation shows. ``anonymize=True`` drops the subject id from the title."""
    sig = stim_pair_traces(patient, stim_pair, electrodes, tmin, tmax,
                           dirname, runs, require_use_channel)
    names, traces, times = sig["names"], sig["traces"], sig["times"]
    if baseline is not None:
        bwin = (times >= baseline[0]) & (times <= baseline[1])
        if bwin.any():
            traces = traces - traces[:, bwin].mean(axis=1, keepdims=True)
    t_ms = np.asarray(times) * 1e3
    fig = go.Figure()
    if baseline is not None:
        fig.add_vrect(x0=baseline[0] * 1e3, x1=baseline[1] * 1e3,
                      fillcolor="lightgray", opacity=0.4, line_width=0)
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(x=t_ms, y=traces[i], mode="lines",
                                 name=name, line=dict(width=1.4)))
    fig.add_hline(y=0, line=dict(color="gray", width=0.6))
    fig.add_vline(x=0, line=dict(color="black", width=1, dash="dash"))
    if vmax is not None:
        for y in (vmax, -vmax):
            fig.add_hline(y=y, line=dict(color="lightgray", width=0.7, dash="dot"))
    runs_txt = f", runs {sig['runs_used']}" if sig.get("runs_used") else ""
    _pt = "" if anonymize else f"{patient} — "
    fig.update_layout(title=f"{_pt}stim {stim_pair} evoked traces{runs_txt}",
                      xaxis_title="time (ms)", yaxis_title="voltage (a.u.)",
                      template="simple_white", width=900, height=500)
    if show:
        fig.show()
    return fig


# =========================================================================== #
# CCEP aura animations  
# --------------------------------------------------------------------------- #
# One stim pair's evoked response drawn as distance-decaying polarity auras on the
# subject's native FreeSurfer (ACPC mm) surfaces: hue = voltage polarity (red +,
# blue -) via a diverging colormap centered at 0; intensity fades with true
# Euclidean distance from each electrode, so the glow is a physically meaningful
# smear on the cortical mesh. Two render paths share the geometry / decay-kernel
# helpers below:
#   * animate_stim_response_interactive -> Plotly (base env; orbit + Play + scrubber)
#   * animate_stim_response             -> PyVista GIF (needs code/.venv-aura)
# PyVista (and matplotlib, for the GIF colormap) are imported lazily inside the GIF
# path, so importing utils in the base env never pulls them in.
# =========================================================================== #


# --- geometry: native surfaces, cropped around the electrodes --------------- #
def _electrode_coords(patient: str, names: list[str]) -> dict:
    """name -> ACPC xyz (mm) for localized electrodes; also the derived hemi set."""
    elec = _load_electrodes(patient)
    xyz = {str(r["name"]): np.array([r["x"], r["y"], r["z"]], float)
           for _, r in elec.iterrows()}
    hemi = {str(r["name"]): r["_hemi"] for _, r in elec.iterrows()}
    coords = {n: xyz[n] for n in names if n in xyz}
    hemis = sorted({hemi[n] for n in coords})
    return dict(coords=coords, hemis=hemis, hemi_of=hemi)


def _load_surface_pv(patient: str, kind: str, hemis: list[str]):
    """Combined PyVista PolyData of ``kind`` ('pial'|'white') over ``hemis``."""
    import pyvista as pv

    fs = lp.freesurfer_dir(patient)
    meshes = []
    for h in hemis:
        pts, tri = _load_gifti(fs / f"{kind}.{h}.surf.gii")
        if pts is None or tri is None:
            continue
        faces = np.hstack([np.full((tri.shape[0], 1), 3, int), tri]).ravel()
        meshes.append(pv.PolyData(pts, faces))
    if not meshes:
        raise FileNotFoundError(f"No {kind}.{{{','.join(hemis)}}}.surf.gii for {patient}")
    m = meshes[0]
    for extra in meshes[1:]:
        m = m.merge(extra)
    return m


def _crop_to_electrodes(mesh, elec_xyz: np.ndarray, margin_mm: float):
    """Keep the surface patch within ``margin_mm`` of ANY electrode."""
    tree = cKDTree(elec_xyz)
    d, _ = tree.query(mesh.points)
    keep = d <= margin_mm
    if not keep.any():
        return None
    sub = mesh.extract_points(keep, adjacent_cells=True)
    return sub.extract_surface(algorithm="dataset_surface")


def _concat_meshes(meshes: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate ``[(display_pts, anat_pts, tri), ...]`` into one mesh, offsetting
    triangle indices. ``display_pts`` is what gets drawn; ``anat_pts`` is the
    (native) position used for the distance kernel (differs only for inflated)."""
    dparts, aparts, tparts, off = [], [], [], 0
    for dp, ap, tr in meshes:
        dparts.append(dp); aparts.append(ap); tparts.append(tr + off)
        off += dp.shape[0]
    return np.vstack(dparts), np.vstack(aparts), np.vstack(tparts)


def _crop_mesh(display_pts, anat_pts, tri, elec_xyz, margin):
    """Keep vertices whose *anatomical* position is within ``margin`` of any
    electrode, drop faces that lose a vertex, and reindex. Returns the cropped
    (display_pts, anat_pts, tri) or ``None`` if nothing survives."""
    d, _ = cKDTree(elec_xyz).query(anat_pts)
    keep = d <= margin
    if not keep.any():
        return None
    idx = np.where(keep)[0]
    remap = np.full(keep.shape[0], -1, int)
    remap[idx] = np.arange(idx.size)
    face_keep = keep[tri].all(axis=1)
    return display_pts[idx], anat_pts[idx], remap[tri[face_keep]]


# --- aura kernel + per-frame coloring --------------------------------------- #
def _decay_kernel(d: np.ndarray, sigma_mm: float, decay: str, cutoff: float) -> np.ndarray:
    if decay == "gaussian":
        w = np.exp(-(d ** 2) / (2.0 * sigma_mm ** 2))
    elif decay in ("exp", "exponential"):
        w = np.exp(-d / sigma_mm)
    elif decay == "linear":
        w = np.clip(1.0 - d / cutoff, 0.0, 1.0)
    else:
        raise ValueError("decay must be 'gaussian', 'exp', or 'linear'")
    w[d > cutoff] = 0.0
    return w


def _decay_weights(vertices: np.ndarray, elec_xyz: np.ndarray, sigma_mm: float,
                   decay: str, cutoff: float, chunk: int = 20000) -> np.ndarray:
    """(n_vertices x n_elec) spatial weights, zeroed beyond ``cutoff`` mm.

    Distances are computed in vertex chunks so the transient
    ``(n_vert x n_elec x 3)`` array stays bounded even for whole-hemisphere
    crops with many electrodes."""
    n = vertices.shape[0]
    W = np.empty((n, elec_xyz.shape[0]), np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = np.linalg.norm(vertices[s:e, None, :] - elec_xyz[None, :, :], axis=2)
        W[s:e] = _decay_kernel(d, sigma_mm, decay, cutoff)
    return W


def _blend_rgb(field: np.ndarray, coverage: np.ndarray, cmap, vmax: float,
               base_gray: float, aura_gain: float) -> np.ndarray:
    """Diverging aura -> (N x 3) uint8 blended over a gray base.

    ``field`` is the distance-weighted *mean* voltage (real units, signed) so hue
    tracks polarity and magnitude is not inflated by electrode density.
    ``coverage`` in [0, 1] is how close the vertex is to any electrode (fades the
    aura to gray with distance). Opacity = ``coverage`` * ramp(|field|), so a
    vertex is gray both far from electrodes AND at baseline (|field| ~ 0)."""
    norm = np.clip(field / vmax, -1.0, 1.0)
    aura = cmap((norm + 1.0) / 2.0)[:, :3]                     # (N,3) in [0,1]
    alpha = (np.clip(coverage, 0.0, 1.0) *
             np.clip(np.abs(field) / vmax * aura_gain, 0.0, 1.0))[:, None]
    rgb = base_gray * (1.0 - alpha) + aura * alpha
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _diverging_scale(mid_hex: str, neg: str = "#2166ac", pos: str = "#b2182b") -> list:
    """Plotly colorscale blue -> ``mid_hex`` -> red (neutral at the midpoint, so
    zero field shows the surface's own base color)."""
    return [[0.0, neg], [0.5, mid_hex], [1.0, pos]]


# --- interactive Plotly aura (base env; orbit + Play + time scrubber) ------- #
def animate_stim_response_interactive(
    patient: str,
    stim_pair: str,
    electrodes: list[str] | None = None,
    *,
    surface: str = "native",
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: tuple[float, float] | None = (-0.1, -0.01),
    sigma_mm: float = 8.0,
    decay: str = "gaussian",
    cutoff_mm: float | None = None,
    crop_margin_mm: float | None = None,
    vmax: float | None = None,
    n_frames: int = 700,
    frame_ms: int = 80,
    pial_color: str = "#e7c6ad",
    white_color: str = "#dcdcd2",
    cortex_color: str = "#c6c6bd",
    pial_opacity: float = 0.5,
    show_white: bool = True,
    hemi: str | None = None,
    decimate: int = 1,
    marker_size: int = 6,
    html_out: str | Path | None = None,
    show_traces: bool = True,
    traces_out: str | Path | None = None,
    return_traces: bool = False,
    width: int = 950,
    height: int = 760,
    dirname: str | None = None,
    runs: list[str] | None = None,
    require_use_channel: bool = True,
    anonymize: bool = False,
    verbose: bool = True,
) -> go.Figure:
    """Interactive Plotly CCEP aura animation (orbit / zoom / pan, Play button, time
    scrubber), using only the base env. Renders inline in Jupyter and can export a
    self-contained HTML (``html_out=...``).

    The aura is a distance-weighted sum of each electrode's (baseline-subtracted)
    voltage mapped through a diverging colorscale whose *midpoint is the surface's
    own base color*, so zero field (baseline / far from electrodes) shows plain
    cortex and the response blooms red/blue with polarity.

    ``anonymize=True`` drops the subject id from BOTH the 3-D figure title and the
    companion voltage-trace plot.

    Parameters
    ----------
    surface : ``'native'`` (default) draws the **pial and white** surfaces in
        distinct colors (``pial_color`` / ``white_color``), pial translucent.
        ``'inflated'`` draws the single inflated surface (``cortex_color``); the
        aura still decays by *true* anatomical distance (kernel evaluated at the
        mid-thickness position, painted on the inflated vertex). Electrodes are
        snapped pial->inflated for display.
    pial_color, white_color, cortex_color : hex base colors (diverging midpoints).
    n_frames : evenly-spaced time samples turned into animation frames. Each frame
        carries a per-vertex intensity array, so keep this modest for big crops
        (or raise ``decimate`` / shrink ``crop_margin_mm``).
    frame_ms : ms per frame during Play.
    hemi : restrict surfaces to 'L'/'R'; default = the electrodes' hemisphere(s).
    decimate : keep every k-th triangle (lighter figure; geometry only).
    html_out : if given, also write a standalone interactive HTML there.
    show_traces : also build (and show) the per-electrode evoked-trace plot via
        :func:`plot_stim_pair_traces`. Default ``True``.
    traces_out : if given, save that traces plot as standalone HTML.
    return_traces : if ``True``, return ``(fig3d, traces_fig)`` and skip
        auto-displaying the traces plot (the caller controls it).
    Other parameters match :func:`animate_stim_response`.
    """
    cutoff_mm = cutoff_mm if cutoff_mm is not None else 3.0 * sigma_mm
    crop_margin_mm = crop_margin_mm if crop_margin_mm is not None else 8.0
    if surface not in ("native", "inflated"):
        raise ValueError("surface must be 'native' or 'inflated'")

    # 1. signal + baseline
    sig = stim_pair_traces(patient, stim_pair, electrodes, tmin, tmax,
                           dirname, runs, require_use_channel)
    names, traces, times = sig["names"], sig["traces"], sig["times"]
    if baseline is not None:
        bwin = (times >= baseline[0]) & (times <= baseline[1])
        if bwin.any():
            traces = traces - traces[:, bwin].mean(axis=1, keepdims=True)

    # 2. coordinates
    loc = _electrode_coords(patient, names)
    coords, hemi_of = loc["coords"], loc["hemi_of"]
    keep = [n in coords for n in names]
    names = [n for n, k in zip(names, keep) if k]
    traces = traces[np.array(keep)]
    if not names:
        raise ValueError("No electrodes with localized coordinates.")
    elec_xyz = np.array([coords[n] for n in names])
    hemis = [hemi.upper()] if hemi else sorted({hemi_of[n] for n in names})

    if vmax is None:
        vmax = float(np.percentile(np.abs(traces), 97.5)) or 1.0
    fs = lp.freesurfer_dir(patient)

    # 3. surfaces (cropped) + decay kernels
    built = []                       # dicts: kind, color, opacity, dp, tri, W
    if surface == "native":
        specs = ([("white", white_color, 1.0)] if show_white else []) \
              + [("pial", pial_color, pial_opacity)]
        for kind, color, opac in specs:
            per_h = []
            for h in hemis:
                pts, tri = _load_gifti(fs / f"{kind}.{h}.surf.gii")
                per_h.append((pts, pts, tri))          # display == anatomical
            dp, ap, tr = _concat_meshes(per_h)
            crop = _crop_mesh(dp, ap, tr, elec_xyz, crop_margin_mm)
            if crop is None:
                continue
            dp, ap, tr = crop
            if decimate > 1:
                tr = tr[::decimate]
            W = _decay_weights(ap, elec_xyz, sigma_mm, decay, cutoff_mm)
            built.append(dict(kind=kind, color=color, opac=opac, dp=dp, tri=tr, W=W))
        elec_display = elec_xyz
    else:  # inflated
        per_h = []
        for h in hemis:
            ipts, itri = _load_gifti(fs / f"inflated.{h}.surf.gii")
            ppts, _ = _load_gifti(fs / f"pial.{h}.surf.gii")
            wpts, _ = _load_gifti(fs / f"white.{h}.surf.gii")
            per_h.append((ipts, (ppts + wpts) / 2.0, itri))   # display=inflated, anat=mid
        dp, ap, tr = _concat_meshes(per_h)
        crop = _crop_mesh(dp, ap, tr, elec_xyz, crop_margin_mm)
        if crop is None:
            raise ValueError("No inflated surface near the electrodes.")
        dp, ap, tr = crop
        if decimate > 1:
            tr = tr[::decimate]
        W = _decay_weights(ap, elec_xyz, sigma_mm, decay, cutoff_mm)
        built.append(dict(kind="inflated", color=cortex_color, opac=1.0,
                          dp=dp, tri=tr, W=W))
        # snap electrodes pial -> inflated for display
        disp = {}
        for h in hemis:
            names_h = [n for n in names if hemi_of[n] == h]
            if not names_h:
                continue
            ppts, _ = _load_gifti(fs / f"pial.{h}.surf.gii")
            ipts, _ = _load_gifti(fs / f"inflated.{h}.surf.gii")
            snapped = _snap_to_inflated(np.array([coords[n] for n in names_h]), ppts, ipts)
            disp.update({n: c for n, c in zip(names_h, snapped)})
        elec_display = np.array([disp[n] for n in names])

    if not built:
        raise ValueError("No surface vertices near the electrodes — raise crop_margin_mm.")

    frame_idx = np.linspace(0, len(times) - 1,
                            min(n_frames, len(times))).round().astype(int)
    if verbose:
        nv = sum(b["dp"].shape[0] for b in built)
        print(f"[aura] interactive {surface}: {len(names)} elec, hemi {hemis}, "
              f"{nv} vertices, {len(frame_idx)} frames, vmax={vmax:.3g}")

    marker_scale = _diverging_scale("#e0e0e0")

    # 4. base traces (float32 / int32 to keep the base64 payload small)
    def field(W, fi):
        return (W @ traces[:, fi]).astype(np.float32)

    data = []
    for b in built:
        dp = b["dp"].astype(np.float32)
        tr = b["tri"].astype(np.int32)
        data.append(go.Mesh3d(
            x=dp[:, 0], y=dp[:, 1], z=dp[:, 2],
            i=tr[:, 0], j=tr[:, 1], k=tr[:, 2],
            intensity=field(b["W"], frame_idx[0]), intensitymode="vertex",
            colorscale=_diverging_scale(b["color"]), cmin=-vmax, cmax=vmax,
            opacity=b["opac"], showscale=False, name=b["kind"], hoverinfo="skip",
            lighting=dict(ambient=0.55, diffuse=0.6, specular=0.08, roughness=0.9),
            flatshading=False))
    elec_display = elec_display.astype(np.float32)
    data.append(go.Scatter3d(
        x=elec_display[:, 0], y=elec_display[:, 1], z=elec_display[:, 2],
        mode="markers", text=names,
        marker=dict(size=marker_size, color=traces[:, frame_idx[0]].astype(np.float32),
                    colorscale=marker_scale, cmin=-vmax, cmax=vmax,
                    line=dict(color="black", width=1),
                    colorbar=dict(title="voltage (a.u.)", len=0.6, x=0.92)),
        name="electrodes", hovertemplate="%{text}<extra></extra>"))
    anim_traces = list(range(len(built) + 1))

    # 5. frames
    frames = []
    for k, fi in enumerate(frame_idx):
        fdata = [go.Mesh3d(intensity=field(b["W"], fi)) for b in built]
        fdata.append(go.Scatter3d(marker=dict(color=traces[:, fi].astype(np.float32),
                                              colorscale=marker_scale,
                                              cmin=-vmax, cmax=vmax)))
        frames.append(go.Frame(name=str(k), data=fdata, traces=anim_traces))

    # 6. slider + play/pause
    steps = [dict(method="animate", label=f"{times[fi] * 1e3:+.0f}",
                  args=[[str(k)], dict(mode="immediate",
                                       frame=dict(duration=frame_ms, redraw=True),
                                       transition=dict(duration=0))])
             for k, fi in enumerate(frame_idx)]
    play = dict(type="buttons", direction="left", x=0.02, y=0.02,
                xanchor="left", yanchor="bottom", showactive=False, pad=dict(t=0, r=6),
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                          fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(mode="immediate",
                                            frame=dict(duration=0, redraw=False))])])

    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        width=width, height=height,
        title=((f"CCEP aura — stim {stim_pair} " if anonymize else
                f"CCEP aura — {patient}, stim {stim_pair} ")
               + f"({surface}, runs {sig['runs_used']})"),
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        updatemenus=[play],
        sliders=[dict(active=0, x=0.12, y=0.02, len=0.8,
                      pad=dict(t=0, b=0), currentvalue=dict(prefix="t = ", suffix=" ms"),
                      steps=steps)],
        margin=dict(l=0, r=0, t=40, b=0))

    if html_out is not None:
        html_out = Path(html_out)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_out), include_plotlyjs=True, auto_play=False)
        if verbose:
            print(f"[aura] wrote interactive HTML -> {html_out}")

    # 7. companion: per-electrode voltage traces (same anonymize setting)
    fig_tr = None
    if show_traces or return_traces or traces_out is not None:
        fig_tr = plot_stim_pair_traces(
            patient, stim_pair, electrodes=names, tmin=tmin, tmax=tmax,
            baseline=baseline, vmax=vmax, dirname=dirname, runs=runs,
            require_use_channel=require_use_channel, anonymize=anonymize, show=False)
        if traces_out is not None:
            traces_out = Path(traces_out)
            traces_out.parent.mkdir(parents=True, exist_ok=True)
            fig_tr.write_html(str(traces_out))
            if verbose:
                print(f"[aura] wrote voltage-trace HTML -> {traces_out}")

    if return_traces:
        return fig, fig_tr
    if fig_tr is not None and show_traces:
        fig_tr.show()
    return fig

