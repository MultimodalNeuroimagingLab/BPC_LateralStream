"""
spatial_plot.py
===============
General-purpose spatial rendering of sEEG electrodes on a subject's INFLATED
brain, with surface coordinates snapped pial -> inflated
(cf. MATLAB ``ieeg_snap2inflated`` / ``vp_inflated_pial_BRWplusITG1``).

Two public plotters, both patient-id driven:

- ``plot_electrodes_inflated(patient, ...)`` -- the general entry point: render
  **every** electrode for a subject, colored by shaft / hemisphere / seizure
  zone / Destrieux label / any electrodes.tsv column (or a flat single color).
  Optionally highlight a subset and label individual contacts.

- ``plot_bpcs_inflated(patient, contact, ...)`` -- the BPC-specific view used by
  the single-electrode pipeline: stim-pair midpoints colored by BPC, plus the
  recording electrode. Builds on the same surface/snap helpers.

**Coordinate system: subject-native ACPC mm only.** Every file path below
matches ``vp_inflated_pial_BRWplusITG1.m`` with ``mni=0`` (no fsaverage / no
MNI305). Render on THIS subject's own pial/inflated GIFTI.

Path map (matches MATLAB ``mni=0`` branch exactly):

    Pial GIFTI       derivatives/freesurfer/sub-<p>/pial.{L,R}.surf.gii
    Inflated GIFTI   derivatives/freesurfer/sub-<p>/inflated.{L,R}.surf.gii
    Electrodes       sub-<p>/ses-ieeg01/ieeg/sub-<p>_ses-ieeg01_electrodes.tsv
    GM/WM dist       derivatives/gm_wmDistance/sub-<p>/sub-<p>_ses-ieeg01_gm_wm.tsv
    Wang             derivatives/freesurfer/sub-<p>/surf/{lh,rh}.wang15_mplbl.mgz
    Rosenke          derivatives/freesurfer/sub-<p>/surf/{lh,rh}.rosenke18_vcatlas.mgz
    Benson           derivatives/freesurfer/sub-<p>/surf/{lh,rh}.benson14_varea.mgz
    Brainnetome      derivatives/freesurfer/sub-<p>/label/{lh,rh}.BN_Atlas.annot

Two atlas modes for the cortical surface coloring:

- ``atlas='visual'``  (the MATLAB ``vp_inflated_pial_BRWplusITG1`` combo):
  Wang (TO/LO/V3{A,B}/IPS/SPL/FEF, labels 9-22) + Benson (V1-V3, labels 1-3) +
  Rosenke (hOc4v/FG1-4, labels 4-8) + Brainnetome A37 (label 23). Only those
  visual ROIs are colored; the rest of the cortex stays gray. Same 23-row
  pastel-amp cmap as the MATLAB script.

- ``atlas='<fs-annot-name>'`` (e.g. ``'aparc.a2009s'``, ``'aparc'``,
  ``'BN_Atlas'``): every region in the chosen FreeSurfer ``.annot`` is colored
  from its ctab. Useful for a full-cortex parcellation.

- ``atlas=None``: plain gray cortex (no parcellation), good for a clean
  electrode-only view.

``plot_bpcs_inflated`` renders the cortex OPAQUE by default so back-of-brain
stim-pair midpoints are correctly occluded; ``plot_electrodes_inflated`` keeps a
translucent default so deep / subcortical markers stay visible. Adjust with
``surface_opacity``.

Note on ``derivatives/loc_info``: ``loc_info.mat`` is a MATLAB v5 file
containing a MATLAB ``table`` (MatlabOpaque) -- not Python-readable via
``scipy.io`` or ``h5py``. The vp_inflated MATLAB script itself just calls its
electrodes.tsv table ``loc_info``, so we use ``electrodes.tsv`` directly
(same source data) and read the visual-atlas .mgz files alongside.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import cKDTree

import functions.load_preproc as lp


# --------------------------------------------------------------------------- #
# Visual-atlas constants (Wang + Benson + Rosenke + Brainnetome)
# Matches vp_inflated_pial_BRWplusITG1.m
# --------------------------------------------------------------------------- #
VISUAL_REGION_NAMES = [
    "V1", "V2", "V3",                                              #  1- 3 Benson
    "hOc4v", "FG1", "FG2", "FG3", "FG4",                            #  4- 8 Rosenke
    "TO2", "TO1", "LO2", "LO1", "V3B", "V3A",                       #  9-14 Wang
    "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5", "SPL1", "FEF",  # 15-22 Wang
    "A37a",                                                         #    23 BN
]

_VISUAL_CMAP_RAW = np.array([
    [0.9882, 0.5647, 0.5647],  # V1
    [0.9882, 0.7098, 0.5255],  # V2
    [1.0000, 1.0000, 0.6000],  # V3
    [0.0784, 0.9490, 0.9176],  # hOc4v
    [0.1137, 0.9647, 0.0549],  # FG1
    [0.9333, 0.0549, 0.9647],  # FG2
    [0.0863, 0.1059, 0.6824],  # FG3
    [0.4784, 0.0471, 0.0471],  # FG4
    [0.9000, 0.6000, 1.0000],  # TO2
    [0.3438, 0.3910, 0.3668],  # TO1
    [0.6974, 0.5129, 0.8403],  # LO2
    [0.1978, 0.8408, 0.7445],  # LO1
    [0.3955, 0.8227, 0.5828],  # V3B
    [0.9686, 0.3098, 0.7353],  # V3A
    [0.2417, 0.9409, 0.7096],  # IPS0
    [0.2196, 0.0039, 0.7588],  # IPS1
    [0.2000, 0.8000, 0.5000],  # IPS2
    [0.9539, 0.7295, 0.7562],  # IPS3
    [0.2877, 0.6154, 0.4391],  # IPS4
    [0.2206, 0.2239, 0.4094],  # IPS5
    [0.9000, 1.0000, 0.7000],  # SPL1
    [0.2000, 0.5000, 0.2000],  # FEF
    [0.6000, 0.8227, 0.5828],  # A37a
])

# MATLAB "pastel amp" adjustments (lines 210-212 of vp_inflated_pial...)
VISUAL_CMAP = _VISUAL_CMAP_RAW.copy()
VISUAL_CMAP[0:3]  = VISUAL_CMAP[0:3]  ** 0.8
VISUAL_CMAP[3:8]  = VISUAL_CMAP[3:8]  ** 0.15
VISUAL_CMAP[8:22] = VISUAL_CMAP[8:22] ** 0.5


# --------------------------------------------------------------------------- #
# Distinct visual-AREA palette for anatomy bars / grids. Deliberately avoids the
# tab10 cluster/BPC colors so a visual area is never confused with a BPC, and is
# stable per area name so the bar plot and the matrix grid share the same colors.
# --------------------------------------------------------------------------- #
_BPC_RGB = cm.tab10(np.linspace(0, 1, 10))[:4, :3]              # the 4 colors to avoid


def _build_area_pool():
    pool = np.vstack([cm.tab20b(np.linspace(0, 1, 20))[:, :3],
                      cm.tab20c(np.linspace(0, 1, 20))[:, :3],
                      cm.Set2(np.linspace(0, 1, 8))[:, :3],
                      cm.Dark2(np.linspace(0, 1, 8))[:, :3]])
    keep, used = [], []
    for c in pool:
        if min(np.linalg.norm(c - b) for b in _BPC_RGB) < 0.28:   # too close to a BPC color
            continue
        if used and min(np.linalg.norm(c - u) for u in used) < 0.13:  # near a kept color
            continue
        keep.append(mc.to_hex(c)); used.append(c)
    return keep


_AREA_POOL = _build_area_pool()
AREA_ORDER = [
    "V1", "V2", "V3", "V3A", "V3B", "hOc4v", "FG1", "FG2", "FG3", "FG4",
    "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5", "SPL1", "A37",
    "LO1", "LO2", "TO1", "TO2", "Posterior", "Dorsal", "Ventral", "Lateral",
]
_AREA_COLOR = {a: _AREA_POOL[i % len(_AREA_POOL)] for i, a in enumerate(AREA_ORDER)}

# Dedicated, deliberately-distinct colors for the streams/pathways (the pathway
# anatomy bar + the grid's stream panels). All clear of the tab10 BPC colors.
PATHWAY_COLOR = {
    "Posterior": "#7570b3",   # indigo
    "Dorsal":    "#a6761d",   # gold-brown
    "Ventral":   "#e7298a",   # magenta
    "Lateral":   "#1b9e77",   # teal
}


def area_color(name: str) -> str:
    """Distinct, BPC-clash-free hex color for a pathway / visual-area name, stable
    across the anatomy bars and the matrix grid. Pathways get their own dedicated
    colors; blank / 'non-visual area' -> gray."""
    if not name or name == "non-visual area":
        return "#cfcfcf"
    if name in PATHWAY_COLOR:                                  # streams: deliberate colors
        return PATHWAY_COLOR[name]
    canon = "A37" if name.startswith("A37") else name          # A37elv_L/R -> A37 (match the grid)
    if canon in _AREA_COLOR:
        return _AREA_COLOR[canon]
    idx = sum(ord(ch) for ch in canon) % len(_AREA_POOL)       # deterministic fallback
    return _AREA_POOL[idx]


_GRAY = 0.78  # default cortex gray (RGB float)


# --------------------------------------------------------------------------- #
# Surface / annotation loaders
# --------------------------------------------------------------------------- #
def _load_gifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    g = nib.load(str(path))
    pts = tri = None
    for d in g.darrays:
        if d.data.ndim == 2 and d.data.shape[1] == 3:
            if np.issubdtype(d.data.dtype, np.floating):
                pts = d.data.astype(float)
            else:
                tri = d.data.astype(int)
    return pts, tri


def _load_annot(label_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (per-vertex label index, ctab Nx5, region names)."""
    labels, ctab, names = nib.freesurfer.io.read_annot(str(label_path))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return labels, ctab, names


def _load_visual_atlas(fs_dir: Path, h: str) -> tuple[np.ndarray, np.ndarray]:
    """Combined Wang + Benson + Rosenke + BN visual atlas for hemisphere ``h``.

    Returns
    -------
    vert_label : (n_vertices,) int -- 0 outside visual ROIs, 1..23 inside
                 (indices into VISUAL_REGION_NAMES / VISUAL_CMAP).
    vcolors    : (n_vertices, 3) float RGB in [0, 1]. Visual ROIs colored
                 per VISUAL_CMAP; everything else light gray.
    """
    hh = h.lower()
    surf = fs_dir / "surf"
    lab  = fs_dir / "label"

    # Wang: keep labels 9..22 after subtracting 3 (per MATLAB)
    W = nib.load(str(surf / f"{hh}h.wang15_mplbl.mgz")).get_fdata().ravel().astype(int)
    W = W - 3
    W[W < 9] = 0

    # Benson: keep 1..3 (V1, V2, V3)
    B = nib.load(str(surf / f"{hh}h.benson14_varea.mgz")).get_fdata().ravel().astype(int)
    B[B > 3] = 0
    B[B < 0] = 0

    # Rosenke: keep 4..8 (hOc4v, FG1-4)
    R = nib.load(str(surf / f"{hh}h.rosenke18_vcatlas.mgz")).get_fdata().ravel().astype(int)
    R[R < 4] = 0

    # Brainnetome: keep A37elv_{L,R} only, recode to 23
    bn_labels, _, bn_names = _load_annot(lab / f"{hh}h.BN_Atlas.annot")
    target = "A37elv_L" if h.upper() == "L" else "A37elv_R"
    BN = np.zeros_like(W)
    if target in bn_names:
        BN[bn_labels == bn_names.index(target)] = 23
    # MATLAB: suppress BN where Rosenke has a label
    BN[R != 0] = 0

    # Priority: BN > Benson > (Wang + Rosenke)
    vert_label = np.zeros_like(W)
    mask_bn     = BN == 23
    mask_benson = (B != 0) & ~mask_bn
    mask_wr     = ~mask_bn & ~mask_benson
    vert_label[mask_bn]     = 23
    vert_label[mask_benson] = B[mask_benson]
    vert_label[mask_wr]     = (W + R)[mask_wr]

    vcolors = np.full((vert_label.shape[0], 3), _GRAY)
    valid = (vert_label >= 1) & (vert_label <= 23)
    vcolors[valid] = VISUAL_CMAP[vert_label[valid] - 1]
    return vert_label, vcolors


def _vertex_colors_from_annot(labels, ctab) -> np.ndarray:
    """Per-vertex RGB ([0,1]) from a FreeSurfer ctab."""
    n_regions = ctab.shape[0]
    rgb = ctab[:, :3].astype(float) / 255.0
    out = np.full((labels.shape[0], 3), _GRAY)  # light gray fallback
    valid = (labels >= 0) & (labels < n_regions)
    out[valid] = rgb[labels[valid]]
    return out


def _snap_to_inflated(elec_xyz: np.ndarray, pial_pts: np.ndarray,
                      infl_pts: np.ndarray) -> np.ndarray:
    """For each electrode, nearest PIAL vertex -> that vertex's INFLATED coord."""
    tree = cKDTree(pial_pts)
    _, idx = tree.query(elec_xyz)
    return infl_pts[idx]


# --------------------------------------------------------------------------- #
# Shared surface / electrode plumbing
# --------------------------------------------------------------------------- #
def _build_surfaces(fs_dir: Path, label_dir: Path, hemis: list[str],
                    atlas: str | None) -> tuple[dict[str, dict], set[int]]:
    """Load pial + inflated geometry and per-vertex colors for each hemisphere.

    Returns ``(surfaces, present_visual)`` where ``surfaces[h]`` has keys
    ``pial_pts``, ``infl_pts``, ``infl_tri``, ``vcolors`` and ``present_visual``
    is the set of visual-ROI ids found (empty unless ``atlas == 'visual'``).
    """
    surfaces: dict[str, dict] = {}
    present_visual: set[int] = set()
    for h in hemis:
        pial_pts, _        = _load_gifti(fs_dir / f"pial.{h}.surf.gii")
        infl_pts, infl_tri = _load_gifti(fs_dir / f"inflated.{h}.surf.gii")

        if atlas == "visual":
            vert_label, vcolors = _load_visual_atlas(fs_dir, h)
            present_visual |= {int(v) for v in np.unique(vert_label) if v > 0}
        elif atlas is None:
            vcolors = np.full((infl_pts.shape[0], 3), _GRAY)
        else:
            labels, ctab, _ = _load_annot(label_dir / f"{h.lower()}h.{atlas}.annot")
            vcolors = _vertex_colors_from_annot(labels, ctab)

        surfaces[h] = dict(pial_pts=pial_pts, infl_pts=infl_pts,
                           infl_tri=infl_tri, vcolors=vcolors)
    return surfaces, present_visual


def _add_surface_traces(fig: go.Figure, surfaces: dict[str, dict],
                        surface_opacity: float, decimate: int) -> None:
    """Add one translucent inflated-surface Mesh3d per hemisphere."""
    for h, s in surfaces.items():
        pts, tri, vc = s["infl_pts"], s["infl_tri"], s["vcolors"]
        if decimate > 1:
            tri = tri[::decimate]
        vc_hex = ["#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
                  for r, g, b in vc]
        fig.add_trace(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            vertexcolor=vc_hex, opacity=surface_opacity,
            hoverinfo="skip", showscale=False, name=f"inflated {h}",
            showlegend=False,
        ))


def _load_electrodes(patient: str, gm_only: bool = False,
                     gm_thresh: float = -1.0) -> pd.DataFrame:
    """electrodes.tsv -> DataFrame with valid coords + a derived ``_hemi`` column.

    Drops rows with missing x/y/z. ``_hemi`` is taken from the ``hemisphere``
    column when present, else inferred from the sign of x. When ``gm_only`` is
    set, keeps only contacts with gm_wm RelativeDistance >= ``gm_thresh``.
    """
    elec = pd.read_csv(lp.electrodes_tsv(patient), sep="\t",
                       na_values=["n/a", "N/A"])
    elec = elec.dropna(subset=["x", "y", "z"]).copy()
    elec["name"] = elec["name"].astype(str)

    if "hemisphere" in elec.columns:
        hemi_vals = []
        for v, x in zip(elec["hemisphere"], elec["x"]):
            if isinstance(v, str) and v.strip():
                hemi_vals.append(v.strip()[0].upper())
            else:
                hemi_vals.append("L" if x < 0 else "R")
        elec["_hemi"] = hemi_vals
    else:
        elec["_hemi"] = np.where(elec["x"].to_numpy(float) < 0, "L", "R")

    if gm_only:
        gm_file = (lp.DERIV_ROOT / "gm_wmDistance" / f"sub-{patient}"
                   / f"sub-{patient}_ses-{lp.SESSION}_gm_wm.tsv")
        if gm_file.exists():
            gm = pd.read_csv(gm_file, sep="\t", na_values=["n/a", "N/A"])
            # gm_wm.tsv names the contact column 'ElectrodeName' (fallback 'name')
            name_col = "ElectrodeName" if "ElectrodeName" in gm.columns else "name"
            keep = set(gm.loc[gm["RelativeDistance"] >= gm_thresh, name_col].astype(str))
            elec = elec[elec["name"].isin(keep)]

    return elec.reset_index(drop=True)


def _snap_electrodes(elec: pd.DataFrame,
                     surfaces: dict[str, dict]) -> dict[str, np.ndarray]:
    """name -> inflated xyz, snapping each contact via its hemisphere surface."""
    snapped: dict[str, np.ndarray] = {}
    for h, s in surfaces.items():
        sub = elec[elec["_hemi"] == h]
        if sub.empty:
            continue
        coords = sub[["x", "y", "z"]].to_numpy(float)
        infl = _snap_to_inflated(coords, s["pial_pts"], s["infl_pts"])
        for name, c in zip(sub["name"], infl):
            snapped[name] = c
    return snapped


def _add_visual_legend(fig: go.Figure, present_visual: set[int]) -> None:
    """Legend-only square swatches for the visual ROIs present on the surface."""
    for k in sorted(present_visual):
        if not (1 <= k <= len(VISUAL_REGION_NAMES)):
            continue
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=10, color=mc.to_hex(VISUAL_CMAP[k - 1]), symbol="square"),
            name=VISUAL_REGION_NAMES[k - 1], showlegend=True, hoverinfo="skip",
            legendgroup="visual_rois", legendgrouptitle_text="Visual ROIs",
        ))


def _blank_scene_layout(fig: go.Figure, title: str, width: int, height: int,
                        annotations: list[dict] | None = None) -> None:
    scene = dict(aspectmode="data",
                 xaxis=dict(visible=False),
                 yaxis=dict(visible=False),
                 zaxis=dict(visible=False))
    if annotations:
        scene["annotations"] = annotations
    fig.update_layout(
        scene=scene,
        title=title, width=width, height=height,
        legend=dict(groupclick="toggleitem"),
    )


def _label_annotation(coord, text: str, color: str, size: int, offset: int) -> dict:
    """A 3-D scene annotation for one point. Scene annotations live in Plotly's SVG
    overlay, so they always draw IN FRONT of the (possibly opaque) surface meshes —
    unlike ``Scatter3d`` text, which is depth-tested and hidden behind the cortex
    from the wrong angle. A short leader arrow points back to the 3-D location so an
    occluded marker is still locatable. Draggable with ``config={'editable': True}``."""
    return dict(
        x=float(coord[0]), y=float(coord[1]), z=float(coord[2]),
        text=text, showarrow=True, arrowhead=2, arrowwidth=0.6,
        arrowcolor="rgba(80,80,80,0.6)",
        ax=offset, ay=-offset,
        font=dict(size=size, color=color),
        xanchor="left", yanchor="bottom",
    )


def _marker_annotation(coord, glyph: str, color: str, size: int) -> dict:
    """An always-on-top glyph (filled ``●`` dot / ``◆`` diamond) drawn exactly at a
    3-D point (no leader arrow). Like :func:`_label_annotation` it lives in Plotly's
    SVG overlay, so it marks the electrode placement IN FRONT of an opaque surface
    that would otherwise occlude the depth-tested Scatter3d marker."""
    return dict(
        x=float(coord[0]), y=float(coord[1]), z=float(coord[2]),
        text=glyph, showarrow=False,
        font=dict(size=size, color=color),
        xanchor="center", yanchor="middle",
    )


# --------------------------------------------------------------------------- #
# Public plot: all electrodes (general purpose)
# --------------------------------------------------------------------------- #
def plot_electrodes_inflated(
    patient: str,
    atlas: str | None = None,
    hemi: str = "both",
    gm_only: bool = False,
    surface_opacity: float = 0.35,
    marker_size: int = 3,
    marker_color: str = "#222222",
    show_labels: bool = True,
    font_size: int = 4,
    text_color: str = "black",
    label_offset: int = 9,
    highlight: list[str] | None = None,
    decimate: int = 1,
    show_atlas_legend: bool = True,
    anonymize: bool = False,
    width: int = 1100,
    height: int = 750,
) -> go.Figure:
    """Render **all** of a subject's electrodes on the inflated brain.

    ``anonymize=True`` drops the subject id from the figure title (for
    de-identified / shareable output).

    The contacts are drawn as uniform (uncolored) markers; each one is labeled
    with a movable text annotation -- a leader line from the contact to a piece
    of text you can drag. To enable dragging, render with the ``editable``
    config flag::

        fig = plot_electrodes_inflated('subj_id')
        fig.show(config={'editable': True})          # drag labels to reposition

    Parameters
    ----------
    patient : BIDS subject id
    atlas : cortical surface coloring -- ``None`` (plain gray, default),
        ``'visual'`` (Wang+Benson+Rosenke+BN), or any FreeSurfer annot name in
        label/ (e.g. 'aparc.a2009s', 'aparc', 'BN_Atlas').
    hemi : 'L', 'R', or 'both'.
    gm_only : if True, drop electrodes with gm_wm RelativeDistance < -1.
    surface_opacity : 0..1 opacity of the brain surface meshes.
    marker_size : electrode marker size.
    marker_color : single color for all contact markers.
    show_labels : if True, add a draggable text label per contact.
    font_size : label font size.
    text_color : label font color.
    label_offset : initial leader-line length in pixels (how far the text sits
        from its contact before you drag it).
    highlight : optional list of contact names to mark distinctly (red
        diamond), e.g. a recording or stim electrode. Their labels are drawn
        in the highlight color.
    decimate : keep only every k-th surface triangle (lighter plot).
    show_atlas_legend : add a color key for visual ROIs (only when
        ``atlas='visual'``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fs_dir    = lp.freesurfer_dir(patient)
    label_dir = fs_dir / "label"

    hemis = ["L", "R"] if hemi == "both" else [hemi.upper()]
    elec = _load_electrodes(patient, gm_only=gm_only)
    elec = elec[elec["_hemi"].isin(hemis)].reset_index(drop=True)

    surfaces, present_visual = _build_surfaces(fs_dir, label_dir, hemis, atlas)
    snapped = _snap_electrodes(elec, surfaces)

    highlight = set(highlight or [])
    names = [n for n in elec["name"] if n in snapped]

    fig = go.Figure()
    _add_surface_traces(fig, surfaces, surface_opacity, decimate)

    # uniform (uncolored) contact markers
    if names:
        pos = np.array([snapped[n] for n in names])
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=marker_color,
                        line=dict(color="black", width=0.3)),
            text=names, name="electrodes", showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        ))

    # highlighted contacts on top (red diamonds)
    hl_names = [n for n in highlight if n in snapped]
    if hl_names:
        pos = np.array([snapped[n] for n in hl_names])
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker=dict(size=marker_size + 5, color="crimson", symbol="diamond"),
            text=hl_names, name="highlight", showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        ))

    # movable text labels (3D scene annotations; draggable with editable config)
    annotations = []
    if show_labels:
        for n in names:
            c = snapped[n]
            is_hl = n in highlight
            annotations.append(dict(
                x=float(c[0]), y=float(c[1]), z=float(c[2]),
                text=n, showarrow=True, arrowhead=2, arrowwidth=0.6,
                arrowcolor="rgba(80,80,80,0.6)",
                ax=label_offset, ay=-label_offset,
                font=dict(size=font_size,
                          color="crimson" if is_hl else text_color),
                xanchor="left", yanchor="bottom",
            ))

    if atlas == "visual" and show_atlas_legend and present_visual:
        _add_visual_legend(fig, present_visual)

    atlas_lbl = atlas if atlas is not None else "no atlas"
    _sub = "" if anonymize else f", sub-{patient}"
    _blank_scene_layout(
        fig,
        title=(f"All electrodes (n={len(names)}) on subject inflated surface "
               f"({atlas_lbl}, ACPC{_sub})"),
        width=width, height=height, annotations=annotations,
    )
    return fig


# --------------------------------------------------------------------------- #
# Public plot: BPC stim-pair midpoints
# --------------------------------------------------------------------------- #
def plot_bpcs_inflated(
    patient: str,
    contact: str,
    bpc_pairs: np.ndarray,
    pairs: np.ndarray,
    plotweights: np.ndarray,
    atlas: str = "visual",
    hemi: str = "both",
    gm_only: bool = False,
    surface_opacity: float = 1.00,
    decimate: int = 1,
    show_pair_labels: bool = True,
    labels_on_top: bool = False,
    label_offset: int = 10,
    label_unclustered: bool = False,
    markers_on_top: bool = False,
    show_unclustered: bool = True,
    show_atlas_legend: bool = True,
    anonymize: bool = False,
    width: int = 1100,
    height: int = 750,
) -> go.Figure:
    """Render BPC stim-pair midpoints on the inflated brain.

    ``anonymize=True`` drops the subject id from the figure title (for
    de-identified / shareable output).

    Parameters
    ----------
    patient : BIDS subject id
    contact : recording electrode of interest, e.g. 'ROCI11'.
    bpc_pairs : (n_pairs,) BPC index per pair (NaN if excluded).
    pairs : (n_pairs,) ordered pair labels matching `bpc_pairs`.
    plotweights : (n_pairs,) mean-SNR weight per pair.
    atlas : 'visual' (Wang+Benson+Rosenke+BN, default), any FreeSurfer annot
        name present in label/ (e.g. 'aparc.a2009s', 'aparc', 'BN_Atlas'), or
        None for plain gray cortex.
    hemi : 'L', 'R', or 'both'.
    gm_only : if True, drop electrodes with gm_wm RelativeDistance < -1.
    surface_opacity : 0..1 opacity of the brain surface meshes (default 1.0 =
        fully opaque, so midpoints behind the cortex are correctly occluded).
    decimate : keep only every k-th triangle (lighter plot; geometry only).
    show_pair_labels : if True, each stim-pair midpoint shows its name as
        a persistent text label colored to match its BPC.
    labels_on_top : if True, draw the pair / recording-electrode labels as 3-D
        *scene annotations* — an SVG overlay that ALWAYS renders in front of the
        surface — instead of depth-tested inline Scatter3d text. Default False so
        labels of back-of-cortex midpoints stay hidden behind the opaque surface
        (no show-through); set True to force all labels visible. Annotations are
        draggable via ``fig.show(config={'editable': True})``.
    label_offset : leader-line length in pixels for the annotations (only used
        when ``labels_on_top``).
    label_unclustered : if True, also label the unclustered (no-BPC) stim pairs;
        default False — they show as plain grey markers with no text.
    markers_on_top : if True, draw an always-on-top glyph at each point so it
        stays visible IN FRONT of the surface. Default False so the depth-tested
        Scatter3d markers are occluded by the opaque cortex (no show-through).
    show_unclustered : if True (default), stim pairs with no BPC assigned
        (``bpc_pairs`` is NaN) are also drawn, as one small grey group.
    show_atlas_legend : if True, add invisible legend-only entries for each
        visual ROI present so the figure carries a color key.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fs_dir    = lp.freesurfer_dir(patient)
    label_dir = fs_dir / "label"

    hemis = ["L", "R"] if hemi == "both" else [hemi.upper()]
    elec = _load_electrodes(patient, gm_only=gm_only)

    surfaces, present_visual = _build_surfaces(fs_dir, label_dir, hemis, atlas)
    snapped = _snap_electrodes(elec[elec["_hemi"].isin(hemis)], surfaces)

    fig = go.Figure()
    _add_surface_traces(fig, surfaces, surface_opacity, decimate)
    annotations: list[dict] = []          # always-on-top labels (scene annotations)

    # BPC stim-pair midpoints
    bpc_pairs = np.asarray(bpc_pairs)
    pairs = np.asarray(pairs)
    plotweights = np.asarray(plotweights)
    n_components = int(np.nanmax(bpc_pairs)) + 1 if np.any(~np.isnan(bpc_pairs)) else 0
    tab10 = cm.tab10(np.linspace(0, 1, 10))

    def _pair_mid(label: str):
        a, b = label.split("-")
        pa, pb = snapped.get(a), snapped.get(b)
        if pa is None or pb is None:
            return None
        return (pa + pb) / 2

    for q in range(n_components):
        items = [(_pair_mid(pairs[i]), pairs[i], plotweights[i])
                 for i in np.where(bpc_pairs == q)[0]]
        items = [(p, n, w) for p, n, w in items if p is not None]
        if not items:
            continue
        pos = np.array([p for p, _, _ in items])
        sizes = [6 + max(np.nan_to_num(w), 0) * 18 for _, _, w in items]
        names = [n for _, n, _ in items]
        color = mc.to_hex(tab10[q % 10])
        inline = show_pair_labels and not labels_on_top
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode=("markers+text" if inline else "markers"),
            marker=dict(size=sizes, color=color,
                        line=dict(color="black", width=0.5)),
            name=f"BPC {q}",
            legendgroup="bpcs",
            legendgrouptitle_text="BPCs",
            text=names,
            textposition="middle right",
            textfont=dict(color=color, size=11),
            hovertemplate="%{text}<extra>BPC " + str(q) + "</extra>",
        ))
        if show_pair_labels and labels_on_top:      # always-on-top labels instead of inline text
            for coord, nm in zip(pos, names):
                annotations.append(_label_annotation(coord, nm, color, 11, label_offset))
        if markers_on_top:                          # always-on-top marker glyph at each midpoint
            for coord, sz in zip(pos, sizes):
                annotations.append(_marker_annotation(coord, "●", color, int(max(8, sz * 1.3))))

    # unclustered stim pairs (no BPC assigned) -> one grey group
    if show_unclustered:
        un = [(_pair_mid(pairs[i]), pairs[i])
              for i in np.where(np.isnan(bpc_pairs))[0]]
        un = [(p, n) for p, n in un if p is not None]
        if un:
            upos = np.array([p for p, _ in un])
            lbl_un = show_pair_labels and label_unclustered      # unclustered labels off by default
            inline = lbl_un and not labels_on_top
            fig.add_trace(go.Scatter3d(
                x=upos[:, 0], y=upos[:, 1], z=upos[:, 2],
                mode=("markers+text" if inline else "markers"),
                marker=dict(size=7, color="gray", opacity=0.5,
                            line=dict(color="black", width=0.3)),
                name=f"unclustered (n={len(un)})",
                legendgroup="bpcs", legendgrouptitle_text="BPCs",
                text=[n for _, n in un], textposition="middle right",
                textfont=dict(color="gray", size=10),
                hovertemplate="%{text}<extra>unclustered</extra>",
            ))
            if lbl_un and labels_on_top:
                for coord, item in zip(upos, un):
                    annotations.append(_label_annotation(coord, item[1], "gray", 10, label_offset))
            if markers_on_top:                          # placement glyphs (no labels)
                for coord in upos:
                    annotations.append(_marker_annotation(coord, "●", "gray", 9))

    # recording electrode
    if contact in snapped:
        ei = snapped[contact]
        fig.add_trace(go.Scatter3d(
            x=[ei[0]], y=[ei[1]], z=[ei[2]],
            mode=("markers" if labels_on_top else "markers+text"),
            marker=dict(size=9, color="black", symbol="diamond"),
            text=[contact], textposition="top center",
            textfont=dict(color="black", size=12),
            name=contact,
            legendgroup="bpcs",
            legendgrouptitle_text="BPCs",
        ))
        if labels_on_top:                           # recording label always on top
            annotations.append(_label_annotation(ei, contact, "black", 12, label_offset))
        if markers_on_top:                          # recording placement glyph (diamond)
            annotations.append(_marker_annotation(ei, "◆", "black", 15))

    if atlas == "visual" and show_atlas_legend and present_visual:
        _add_visual_legend(fig, present_visual)

    _sub = "" if anonymize else f", sub-{patient}"
    _blank_scene_layout(
        fig,
        title=(f"BPCs for {contact} on subject inflated surface "
               f"({atlas}, ACPC{_sub})"),
        width=width, height=height, annotations=annotations,
    )
    return fig
