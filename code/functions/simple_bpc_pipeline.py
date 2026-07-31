"""
simple_bpc_pipeline.py
======================
One-call driver over the single-electrode BPC pipeline (steps 1–11), all figures
rendered with Plotly.

    build_convergent_matrix -> plot_convergent -> exp_decay (optional)
      -> projection -> significance -> nmf -> identify_bpcs
      -> plot_bpcs -> curve_stats/SNR -> spatial_plot -> save_all
"""

from __future__ import annotations

from functions.load_preproc        import build_convergent_matrix, has_freesurfer
from functions.plot_convergent     import plot_convergent_matrix
from functions.exp_decay           import exp_decay_weight, exp_decay_undo
from functions.projection          import projection_matrix
from functions.significance        import significance_matrix
from functions.nmf                 import run_nmf
from functions.bpc_identification  import identify_bpcs
from functions.plot_bpcs           import plot_bpcs
from functions.bpc_stats           import curve_stats, plot_plotweights
from functions.save_outputs        import save_all
from functions.spatial_plot        import plot_bpcs_inflated


def simple_bpc_pipeline(patient, contact, hemisphere, *,
                        bpc_tmin=0.003, bpc_tmax=0.5,
                        exp_weight=True, convergent_kind="heatmap",
                        stim_hemi="ipsi", spatial=True, anonymize=False):
    """Run the full single-electrode BPC pipeline and show every figure.

    Parameters
    ----------
    patient, contact : BIDS subject id and recording-electrode name.
    hemisphere : fallback hemisphere for the inflated-brain plot; normally the plot
        hemisphere is taken from the contact's first letter (L… → left, R… → right),
        and this is used only when it is neither.
    bpc_tmin, bpc_tmax : convergent-matrix analysis window in seconds
        (default 3–500 ms).
    exp_weight : apply the exp(-t/tau) decay weighting (tau=100 ms) before the
        projection/NMF steps and undo it for plotting. On by default.
    convergent_kind : 'heatmap' (default) or 'trace' for the convergent plot.
    stim_hemi : 'ipsi' (default) excludes opposite-hemisphere stimulation;
        'contra'/'opposite' keeps only it; 'L'/'R' explicit; None = both.
    spatial : draw the inflated-brain plot (auto-skipped without FreeSurfer).
    anonymize : drop the subject id from the inflated-brain title / skip message
        (the only patient-identifying spots in this figure set).

    Returns
    -------
    (V, times, stim_sites, pairs, P, tmat, nmf, bpcs, stats)
    """
    # 1. Convergent matrix
    bpc = build_convergent_matrix(patient, contact, bpc_tmin=bpc_tmin,
                                  bpc_tmax=bpc_tmax, stim_hemi=stim_hemi)
    V, times, stim_sites, pairs = bpc["V"], bpc["times"], bpc["stim_sites"], bpc["pairs"]

    # 2. Plot the convergent matrix (heatmap or per-pair voltage traces)
    plot_convergent_matrix(V, times, stim_sites, kind=convergent_kind, show=True)

    # 3. Optional exponential-decay weighting
    if exp_weight:
        V_w, weight = exp_decay_weight(V, times, tau=0.100)
    else:
        V_w, weight = V, None

    # 4. Projection P
    P = projection_matrix(V_w, stim_sites, plot=True)

    # 5. Significance Ξ
    tmat, pairs = significance_matrix(P, stim_sites, pairs=pairs, plot=True)

    # 6. NMF
    nmf = run_nmf(tmat, pairs, penalty="sum", plot=True)
    H = nmf["H"]

    # 7. BPC identification (kPCA PC1)
    bpcs = identify_bpcs(V_w, H, stim_sites, pairs)
    if exp_weight:
        bpcs = exp_decay_undo(bpcs, times, weight=weight)   # Bs -> voltage domain
        Bs_stats = bpcs["Bs_decayed"]                       # stats in the weighted domain
    else:
        Bs_stats = bpcs["Bs"]

    # 8. Plot BPCs (voltage domain)
    plot_bpcs(bpcs, times, xlim=(0.0, 0.5), show=True)

    # 9. Per-pair SNR
    stats = curve_stats(V_w, Bs_stats, stim_sites, pairs, bpcs["bpc_pairs"])
    plot_plotweights(stats, bpcs, pairs, contact=contact, show=True)

    # 10. Inflated-brain plot (needs FreeSurfer surfaces); hemisphere from the contact
    if spatial and has_freesurfer(patient):
        hemi = contact[0].upper() if contact[:1].upper() in ("L", "R") else hemisphere
        plot_bpcs_inflated(
            patient, contact,
            bpc_pairs=bpcs["bpc_pairs"], pairs=pairs,
            plotweights=stats["plotweights"], hemi=hemi, anonymize=anonymize,
        ).show()
    elif spatial:
        _who = "this subject" if anonymize else patient
        print(f"  [skip] no FreeSurfer surfaces for {_who} — skipping inflated-brain plot")

    # 11. Save intermediates
    save_all(patient, contact, dict(
        V=V, times=times, stim_sites=stim_sites, pairs=pairs,
        P=P, tmat=tmat, nmf=nmf, bpcs_pc1=bpcs, stats_pc1=stats))

    return V, times, stim_sites, pairs, P, tmat, nmf, bpcs, stats


def simple_bpc_pipeline_group(electrodes_by_region, hemisphere, *,
                              bpc_tmin=0.003, bpc_tmax=0.5, exp_weight=True,
                              convergent_kind="heatmap", stim_hemi="ipsi",
                              spatial=True, anonymize=False):
    """Run :func:`simple_bpc_pipeline` for every electrode in an
    ``electrodes_by_region`` mapping ``{patient: {sub_label: [contact, ...]}}``.

    Contacts appearing under more than one sub-region are run once per patient.
    Failures are logged and skipped so the batch runs to completion. Returns
    ``{(patient, contact): result_or_exception}``.

    ``anonymize=True`` replaces subject ids in the progress prints with neutral
    ``S1, S2, …`` labels and drops the subject id from every figure.
    """
    todo = []
    for patient, subdict in electrodes_by_region.items():
        seen = set()
        for contacts in subdict.values():
            for contact in contacts:
                if contact not in seen:
                    seen.add(contact)
                    todo.append((patient, contact))

    # neutral per-subject label so anonymized progress lines stay distinguishable
    alias = {p: f"S{i + 1}" for i, p in enumerate(sorted({p for p, _ in todo}))}
    def _who(p):
        return alias[p] if anonymize else p

    results, n = {}, len(todo)
    for i, (patient, contact) in enumerate(todo, 1):
        print(f"\n=== [{i}/{n}] {_who(patient)} / {contact} ===")
        try:
            results[(patient, contact)] = simple_bpc_pipeline(
                patient, contact, hemisphere, bpc_tmin=bpc_tmin, bpc_tmax=bpc_tmax,
                exp_weight=exp_weight, convergent_kind=convergent_kind,
                stim_hemi=stim_hemi, spatial=spatial, anonymize=anonymize)
        except Exception as e:
            print(f"  !! {_who(patient)}/{contact} failed: {e}")
            results[(patient, contact)] = e

    ok = sum(1 for v in results.values() if not isinstance(v, Exception))
    print(f"\nsimple_bpc_pipeline_group: {ok}/{n} electrode(s) completed, {n - ok} failed.")
    return results
