"""
load_preproc.py
===============
Load preprocessed SPES/CCEP runs (the `*_preproc_run-*.mat` files produced by
preprocSPES_basic.m) and build the "convergent" trials x time matrix for one
recording channel.

The .mat files are MATLAB v7.3 (HDF5). Each holds a `prep` struct with:
    data                  (channels x time x trials) float32
    tt                    (time,)                    -- time vec (s) re: stim onset
    srate                 scalar Hz
    stim_channels_names   the stimulated pair, e.g. 'LQ1-LQ2'
    reref_channels_names  the contralateral WM reref channel, e.g. 'RC6'
    use_channels          1-based indices of "good" recording channels
    use_channels_names    names of those good channels
    channels / evts       MATLAB tables (not parsed here -- channel names come
                          from raw channels.tsv instead)

Paths are derived from a single PROJECT_ROOT / "data" via patient-id helpers
(no hard-coded subject directories).
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # code/functions -> repo root
DATA_ROOT    = PROJECT_ROOT / "data"
DERIV_ROOT   = DATA_ROOT / "derivatives"
OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "outputs"   # code/outputs
SESSION      = "ieeg01"
TASK         = "ccep"

# Which preproc derivatives folder to read. This is the CARLA re-referenced output
# (produced by external/CARLA_preproc.m) that every saved BPC output was built from.
PREPROC_DIRNAME = "bsep_basic_analysis"

# Filename pattern, 
_FNAME_RE = re.compile(r"^(?P<subject>[^_]+)_(?P<pair>.+)_preproc_run-(?P<run>\w+)\.mat$")


# --------------------------------------------------------------------------- #
# Patient-driven path helpers
# --------------------------------------------------------------------------- #
def patient_dir(patient: str) -> Path:
    return DATA_ROOT / f"sub-{patient}"

def preproc_dir(patient: str, dirname: str | None = None) -> Path:
    """Preproc dir for `patient`. `dirname` overrides the global PREPROC_DIRNAME."""
    return DERIV_ROOT / (dirname or PREPROC_DIRNAME) / f"sub-{patient}" / "preproc"

def freesurfer_dir(patient: str) -> Path:
    return DERIV_ROOT / "freesurfer" / f"sub-{patient}"

def has_freesurfer(patient: str) -> bool:
    """True if this subject has the FreeSurfer surfaces the inflated-brain plots
    need (pial + inflated GIFTIs). Some subjects have no freesurfer derivatives;
    anatomical plots must be skipped for them."""
    fs = freesurfer_dir(patient)
    return fs.is_dir() and any(fs.glob("pial.*.surf.gii")) and any(fs.glob("inflated.*.surf.gii"))

def electrodes_tsv(patient: str, ses: str = SESSION,
                   prefer_sorted: bool = True) -> Path:
    """Path to the electrodes TSV, accepting either naming convention.

    Looks for both ``sub-<p>_ses-<s>_electrodes_sorted.tsv`` and
    ``sub-<p>_ses-<s>_electrodes.tsv``. When ``prefer_sorted`` is True the
    ``_sorted`` variant wins if it exists; otherwise the plain file is used.
    Falls back to the plain path (even if missing) so callers get a sensible
    FileNotFoundError downstream.
    """
    ieeg = patient_dir(patient) / f"ses-{ses}" / "ieeg"
    stem = f"sub-{patient}_ses-{ses}_electrodes"
    plain  = ieeg / f"{stem}.tsv"
    sorted_ = ieeg / f"{stem}_sorted.tsv"

    order = (sorted_, plain) if prefer_sorted else (plain, sorted_)
    for path in order:
        if path.exists():
            return path
    return plain

def channels_tsv(patient: str, run: str,
                 ses: str = SESSION, task: str = TASK) -> Path:
    return (patient_dir(patient) / f"ses-{ses}" / "ieeg"
            / f"sub-{patient}_ses-{ses}_task-{task}_run-{run}_channels.tsv")

def events_tsv(patient: str, run: str,
               ses: str = SESSION, task: str = TASK) -> Path:
    return (patient_dir(patient) / f"ses-{ses}" / "ieeg"
            / f"sub-{patient}_ses-{ses}_task-{task}_run-{run}_events.tsv")


# --------------------------------------------------------------------------- #
# Stimulation-current resolution (Huang et al. rule)
# --------------------------------------------------------------------------- #
_CURRENT_RE = re.compile(r"([0-9]*\.?[0-9]+)")

def _parse_current(s) -> float | None:
    """'6.0 mA' -> 6.0; 'n/a' / unparseable -> None."""
    m = _CURRENT_RE.search(str(s))
    return float(m.group(1)) if m else None


def stim_current_runs(patient: str, min_trials: int = 8,
                      ses: str = SESSION, task: str = TASK) -> dict | None:
    """Which (stim pair, run) files to keep so no pair mixes stimulation currents.

    Implements the rule from Huang et al. (``saveBPCs.m`` lines 113-125): when one
    stim site was delivered at more than one current, keep the HIGHEST current that
    has at least ``min_trials`` trials, else fall back to the next highest. A pair
    delivered at a single current is untouched.

    In this dataset current is constant within a ``(site, run)`` cell (audited: 0
    cells carry more than one current), so the choice reduces to selecting runs —
    which lets ``build_convergent_matrix`` skip files without loading them. Trial
    counts come from ``events.tsv`` (pre-preproc), so they are an upper bound on the
    trials that actually survive; that only matters within one trial of the
    ``min_trials`` boundary.

    Returns
    -------
    ``{pair: {run, ...}}`` restricted to pairs that needed a choice, or ``None`` if
    no events file carried a usable ``electrical_stimulation_current`` column (in
    which case callers should not filter).
    """
    ieeg = patient_dir(patient) / f"ses-{ses}" / "ieeg"
    by_pair: dict[str, dict[str, tuple[float | None, int]]] = {}
    saw_column = False

    for path in sorted(ieeg.glob(f"sub-{patient}_ses-{ses}_task-{task}_run-*_events.tsv")):
        m = re.search(r"run-(\w+)_events\.tsv$", path.name)
        if m is None:
            continue
        run = m.group(1)
        try:                                   #
            ev = pd.read_csv(path, sep="\t", dtype=str)   # 13-col reordered variant
        except Exception:
            continue
        if ("electrical_stimulation_current" not in ev.columns
                or "electrical_stimulation_site" not in ev.columns):
            continue
        saw_column = True
        for site, grp in ev.groupby("electrical_stimulation_site"):
            vals = {_parse_current(v) for v in grp["electrical_stimulation_current"]}
            if len(vals) > 1:
                # This dataset was audited to have exactly one current per (site, run),
                # which is what lets the choice reduce to selecting runs. If that ever
                # stops holding, run-level selection cannot separate the currents.
                print(f"[warn] stim_current_runs: {patient} run-{run} site {site} has "
                      f"currents {sorted(v for v in vals if v is not None)} within one "
                      f"run — run-level selection cannot split these; leaving as-is.")
                continue
            by_pair.setdefault(str(site), {})[run] = (vals.pop(), len(grp))

    if not saw_column:
        return None

    keep: dict[str, set] = {}
    for pair, runs in by_pair.items():
        currents = {c for c, _ in runs.values() if c is not None}
        if len(currents) < 2:
            continue                            # single current -> nothing to resolve
        # trials available per current, across runs
        per_current: dict[float, int] = {}
        for cur, n in runs.values():
            if cur is not None:
                per_current[cur] = per_current.get(cur, 0) + n
        ordered = sorted(per_current, reverse=True)
        chosen = next((c for c in ordered if per_current[c] >= min_trials),
                      max(per_current, key=per_current.get))
        # keep the chosen current, and never drop a run whose current is unknown
        # (n/a) — an unparseable value is not evidence that the block is the wrong one
        keep[pair] = {r for r, (c, _) in runs.items() if c == chosen or c is None}
    return keep

# --------------------------------------------------------------------------- #
# Low-level HDF5 helpers (MATLAB v7.3 quirks)
# --------------------------------------------------------------------------- #
def _h5_to_str(dataset) -> str:
    arr = np.asarray(dataset[()]).flatten()
    return "".join(chr(int(c)) for c in arr)


def _h5_str_cell(f, dataset) -> list[str]:
    out = []
    for ref in np.asarray(dataset[()]).flatten():
        if isinstance(ref, h5py.Reference) and ref:
            out.append(_h5_to_str(f[ref]))
        else:
            out.append("")
    return out


def _h5_to_str_or_cell(f, dataset) -> str:
    """Decode a field that is normally a char array but may be a MATLAB cellstr.

    Single-reference preproc stores ``reref_channels_names`` as a char array
    (e.g. 'LB7', or 'CARLA'). Some CARLA preproc files instead store it as a
    cellstr (the list of channels forming the adaptive CAR), which h5py reads as
    HDF5 object references. Return a plain str in both cases (cellstr is joined
    with ', ').
    """
    arr = np.asarray(dataset[()]).flatten()
    if arr.size and isinstance(arr[0], h5py.Reference):
        return ", ".join(n for n in _h5_str_cell(f, dataset) if n)
    return _h5_to_str(dataset)


def parse_filename(name: str) -> dict | None:
    m = _FNAME_RE.match(name)
    return m.groupdict() if m else None


# --------------------------------------------------------------------------- #
# Channel names: the `channels` MATLAB table does not round-trip cleanly
# through h5py; its row order matches channels.tsv (verified).
# --------------------------------------------------------------------------- #
def channel_names_for_run(patient: str, run: str) -> list[str] | None:
    tsv = channels_tsv(patient, run)
    if not tsv.exists():
        return None
    df = pd.read_csv(tsv, sep="\t")
    return df["name"].astype(str).tolist()


# --------------------------------------------------------------------------- #
# Loading one run
# --------------------------------------------------------------------------- #
def load_run(path: str | Path, with_data: bool = True) -> dict:
    """Load one preprocessed run/stim-pair .mat file.

    Returns a dict with keys:
        subject, stim_pair, stim_channels (list of 2), run, srate,
        tt (1D np.float64), n_channels, n_time, n_trials,
        use_channels (1-based int idx), use_channels_names,
        reref_channels_names, channel_names, [data if with_data].
    """
    path = Path(path)
    meta = parse_filename(path.name) or {}
    out = {
        "subject":   meta.get("subject"),
        "stim_pair": meta.get("pair"),
        "run":       meta.get("run"),
        "path":      str(path),
    }

    with h5py.File(path, "r") as f:
        out["srate"] = float(np.asarray(f["srate"]).squeeze())
        out["tt"]    = np.asarray(f["tt"]).squeeze().astype(np.float64)

        # data is stored transposed in HDF5: (trials, time, channels)
        # -> back to MATLAB order (channels, time, trials)
        dshape = f["data"].shape
        out["n_trials"], out["n_time"], out["n_channels"] = dshape

        out["stim_channels_names"]  = _h5_to_str(f["stim_channels_names"])
        out["stim_channels"]        = out["stim_channels_names"].split("-")
        out["reref_channels_names"] = _h5_to_str_or_cell(f, f["reref_channels_names"])

        out["use_channels"]       = np.asarray(f["use_channels"]).squeeze().astype(int)
        out["use_channels_names"] = _h5_str_cell(f, f["use_channels_names"])

        if with_data:
            out["data"] = np.asarray(f["data"]).transpose(2, 1, 0)  # ch x time x trial

    out["channel_names"] = channel_names_for_run(out["subject"], out["run"])
    return out


def contact_data_row(run: dict, contact: str, require_use_channel: bool = True) -> int | None:
    """0-based data-row index of ``contact`` in a loaded run — offset-proof.

    Maps the contact to its row via the ``.mat``'s own
    ``use_channels_names`` <-> ``use_channels`` pairing, which is aligned to the
    actual ``data`` rows even when channels were dropped at preproc time
    (``exclude_channels``). This avoids the index shift you get from
    ``channel_names.index(contact)`` over the full channels.tsv (the tsv has the
    excluded rows; the data does not).

    Returns None if the contact is not a usable recording channel in this run.

    With ``require_use_channel=False``, falls back to the full channels.tsv index
    for contacts absent from ``use_channels`` (e.g. bad / stim-neighbor). NOTE:
    that fallback is NOT offset-corrected for subjects with excluded channels —
    fully fixing it requires storing the per-data-row channel names in the .mat.
    """
    used_map = dict(zip(run["use_channels_names"], run["use_channels"].tolist()))
    if contact in used_map:
        return int(used_map[contact]) - 1
    if require_use_channel:
        return None
    names = run.get("channel_names")
    if names is None or contact not in names:
        return None
    return names.index(contact)


def list_contacts(patient: str) -> list[str]:
    """All recording-channel names for this patient (from first preproc file)."""
    first = next(iter(sorted(preproc_dir(patient).glob("*_preproc_run-*.mat"))))
    return load_run(first, with_data=False)["channel_names"]


# --------------------------------------------------------------------------- #
# Convergent matrix (no baseline re-referencing)
# --------------------------------------------------------------------------- #
def _stim_hemi_keep(stim_hemi, contact: str):
    """Hemispheres to KEEP for stim pairs, or None for all.

    A stim pair's hemisphere is the first letter of its label ('LK1-LK2' -> 'L');
    the recording contact's hemisphere is ``contact[0]``. ``stim_hemi``:
      None              -> keep all (no filtering)
      'ipsi' / 'same'   -> only stim on the SAME hemisphere as the contact
      'contra'/'opposite' -> only stim on the OPPOSITE hemisphere
      'L' / 'R'         -> only that hemisphere, explicitly
    """
    if stim_hemi is None:
        return None
    s = str(stim_hemi).lower()
    ch = contact[0].upper()
    if s in ("ipsi", "same"):
        return {ch}
    if s in ("contra", "contralateral", "opposite", "opp"):
        return {"L", "R"} - {ch}
    if s in ("l", "r"):
        return {s.upper()}
    raise ValueError(
        "stim_hemi must be None, 'ipsi'/'same', 'contra'/'opposite', or 'L'/'R'")


def build_convergent_matrix(
    patient: str,
    contact: str,
    bpc_tmin: float = 0.015,
    bpc_tmax: float = 1.0,
    runs: list[str] | None = None,
    include_run_in_label: bool = False,
    require_use_channel: bool = True,
    dirname: str | None = None,
    stim_hemi: str | None = "ipsi",
    resolve_current: bool = True,
) -> dict:
    """Build the BPC "convergent" matrix for ONE recording channel.

    Pivots per-stim-pair preprocessed files into a single-channel,
    all-stim-trials layout.

    Parameters
    ----------
    patient : BIDS subject id,
    contact : recording-channel name, e.g. 'RC6' (see `list_contacts`).
    bpc_tmin, bpc_tmax : analysis window in seconds (default 15 ms - 1 s).
    runs : restrict to these run labels (e.g. ['07','09']); None = all.
    include_run_in_label : True -> labels become 'PAIR@run-XX'.
    require_use_channel : True (default) keeps only stim pairs for which
        this contact passed the pipeline's use_channels filter (i.e.
        not stimulated / not a neighbor / not bad).
    dirname : override the derivatives preproc folder. None = the module
        default ``PREPROC_DIRNAME`` ('bsep_basic_analysis', the CARLA output).
    resolve_current : True (default) applies the Huang et al. rule so no stim pair
        mixes stimulation currents — for a site delivered at several currents, keep
        the highest one with >= 8 trials (else the next highest) and drop the rest
        BEFORE loading.
    stim_hemi : keep only stim pairs from a given hemisphere. DEFAULT 'ipsi'
        keeps stim on the contact's own hemisphere (i.e. EXCLUDES
        opposite-hemisphere stimulation); 'contra'/'opposite' keeps only the
        opposite; 'L'/'R' selects explicitly; None = both hemispheres.
        Hemisphere is read from the first letter of the pair label and of
        ``contact`` (a no-op for unilateral subjects, where ipsi = all).

    Note
    ----
    No baseline subtraction is performed. The MATLAB preproc already applies
    a baseline; downstream consumers (e.g. BPC L2 norm) can re-center if needed.

    Returns dict: V, times, stim_sites, pairs, contact, patient,
                  n_trials, n_pairs, n_files_used.
    """
    keep_hemi = _stim_hemi_keep(stim_hemi, contact)

    # Stim-current resolution (Huang et al.): never let one stim pair mix currents.
    # `keep_current[pair]` is the set of runs delivered at the chosen current; pairs
    # absent from the map were delivered at a single current and are untouched.
    keep_current = stim_current_runs(patient) if resolve_current else None
    dropped_current: list[str] = []

    pdir = preproc_dir(patient, dirname)
    paths = sorted(pdir.glob("*_preproc_run-*.mat"))
    if not paths:
        raise FileNotFoundError(f"No preproc files in {pdir}")

    V_blocks, labels = [], []
    times = None
    files_used = 0
    seen_pairs_runs: dict[str, set] = {}

    for path in paths:
        info = parse_filename(path.name)
        if info is None:
            continue
        if runs is not None and info["run"] not in runs:
            continue
        if keep_hemi is not None and info["pair"][0].upper() not in keep_hemi:
            continue                                          # drop opposite-hemisphere stim (no load)
        if keep_current is not None:
            allowed = keep_current.get(info["pair"])
            if allowed is not None and info["run"] not in allowed:
                dropped_current.append(f"{info['pair']}@run-{info['run']}")
                continue                                      # lower-current block (no load)

        run = load_run(path, with_data=True)
        ci = contact_data_row(run, contact, require_use_channel)  # offset-proof row lookup
        if ci is None:
            continue

        tt    = run["tt"]
        trace = run["data"][ci].astype(np.float64)   # (time x trials)

        win = (tt >= bpc_tmin) & (tt <= bpc_tmax)
        Vt  = trace[win].T                           # (trials x n_times)
        if times is None:
            times = tt[win]

        pair  = run["stim_pair"]
        label = f"{pair}@run-{run['run']}" if include_run_in_label else pair
        V_blocks.append(Vt)
        labels.extend([label] * Vt.shape[0])
        seen_pairs_runs.setdefault(pair, set()).add(run["run"])
        files_used += 1

    if not V_blocks:
        hint = (f" (after stim_hemi={stim_hemi!r}, kept hemispheres {sorted(keep_hemi)})"
                if keep_hemi is not None else "")
        raise ValueError(
            f"No usable stim pairs found for contact {contact!r} in patient "
            f"{patient!r}{hint}. Check the name with list_contacts(patient), or set "
            f"require_use_channel=False.")

    V = np.vstack(V_blocks)
    stim_sites = np.array(labels)
    pairs = np.array(sorted(np.unique(stim_sites)))

    if dropped_current:
        print(f"[note] resolve_current: dropped {len(dropped_current)} lower-current "
              f"stim block(s) so no pair mixes currents "
              f"(e.g. {', '.join(sorted(dropped_current)[:3])}"
              f"{', ...' if len(dropped_current) > 3 else ''}). "
              f"Pass resolve_current=False to keep them.")

    if not include_run_in_label:
        multi = {p: rs for p, rs in seen_pairs_runs.items() if len(rs) > 1}
        if multi:
            ex = next(iter(multi))
            print(f"[note] {len(multi)} stim pair(s) appear in >1 run and were "
                  f"MERGED across runs (e.g. {ex}: runs {sorted(multi[ex])}). "
                  f"Pass include_run_in_label=True to keep them separate.")

    return {
        "V":             V,
        "times":         times,
        "stim_sites":    stim_sites,
        "pairs":         pairs,
        "contact":       contact,
        "patient":       patient,
        "n_trials":      V.shape[0],
        "n_pairs":       len(pairs),
        "n_files_used":  files_used,
    }