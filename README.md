# BPC_LateralStream
Basis profile curve identification in the lateral stream - summer project Kyunghwan Lim

Pipeline for computing **Basis Profile Curves (BPCs)** from single-pulse electrical
stimulation (SPES / CCEP) intracranial-EEG recordings, pooling them across patients
into **consensus BPCs**, validating those clusters, and characterizing the induced
spectral / broadband responses of each consensus BPC.

Every relevant piece of code is used in bpc_analysis.ipynb from start to finish.

---

## 1 · Preprocessing
**Code:** `preproc_functions/CARLA_preproc.m`, `preproc_functions/CARLA_preproc_pipeline.mlx`
(requires the external `CARLA_JNM-main/functions/CARLA.m` on the MATLAB path).

1. Trials in which a stimulation pair involves a white-matter electrode
   (`gmWM_reldist < −1`) are excluded from the analysis.
2. Voltage time series for all channels were individually inspected. Any channels
   with persistent artifacts / excessive line noise were excluded from the analysis.
3. Per-channel baseline rereferencing is applied by subtracting the mean voltage in
   the `[−100, −10] ms` time frame relative to stimulation.
4. CARLA (Common Average Re-referencing by Least Anticorrelation) is then applied for
   the set of all channels.
5. Line noise at 60, 120, and 180 Hz was removed via linear interpolation in the
   log-amplitude vs log-frequency space (Mewett 2004; as used in Huang et al. 2023).
   A half-bandwidth of 1.5 and a line-anchor bandwidth of 2 were used.

Output → `data/derivatives/bsep_basic_analysis/`, read in Python via
`load_preproc.PREPROC_DIRNAME`.

---

## 2 · Single-electrode BPC pipeline
**Code:** `functions/simple_bpc_pipeline.py` — `simple_bpc_pipeline(patient, contact, hemisphere)`.

1. For a single electrode's recordings, a trial × time matrix is built, with the time
   axis ranging from 3 ms after stimulation to 500 ms after stimulation
   (`load_preproc.build_convergent_matrix`).
2. An exponential weighting function is applied along the time axis, with time
   constant **τ = 100 ms** (`exp_decay.exp_decay_weight`).
3. Projection matrices as well as significance matrices are built according to the
   standard BPC procedure described in the original paper (Miller et al. 2021)
   (`projection.projection_matrix`, `significance.significance_matrix`).
4. Non-negative matrix factorization / dimensionality reduction is applied to the
   significance matrix, with the penalty in the dimensionality-reduction step being
   the sum of the upper-triangle elements of `HHᵀ`, where `WH` is the factorization of
   the significance matrix. Dimensionality reduction iteratively occurs until the
   penalty is below a value of 1 (`nmf.run_nmf`, `penalty='sum'`).
5. Winner-take-all / thresholding on the columns of `H` is applied as outlined in the
   original paper to identify clusters of stimulation pairs, and BPCs are calculated
   via linear kernel PCA (first component) on the clustered responses
   (`bpc_identification.identify_bpcs`).
6. BPCs are unweighted for plotting, and the SNR of each constituent stimulation pair
   is calculated by projecting onto the pair's responses and calculating the ratio
   between the scalar term `α` and the residual error `ε`
   (`bpc_stats.curve_stats`; per-pair SNR is stored as `plotweights`).

Outputs are cached per contact under `outputs/sub-<patient>/<contact>/*.npz`
(`save_outputs.save_all`).

---

## 3 · Cross-patient analysis
**Code:** `functions/bpc_group.py` — `group_bpc_analysis_kmeans(...)`
(this module absorbed the former `cluster_bpc.py`).

1. BPC results (cluster identities, cluster BPCs, and the SNR for each clustered
   stimulation pair) for every electrode labeled as being in the lateral stream in
   `loc_info['visualArea']` were accumulated (`utils.electrodes_by_region('Lateral')`).
   All BPCs with a mean SNR greater than or equal to 1 across their stimulation sites
   were pooled into a BPC × time matrix.
2. Exponential weighting with time constant **τ = 100 ms** was applied along the time
   axis of the matrix.
3. Singular value decomposition was performed on the matrix to get a left
   singular-vector matrix `U`, which was truncated to the first **seven** "principal"
   time components. Each BPC was then projected into this principal-component space and
   clustered with a **K-means** algorithm, with the specified number of clusters being
   **2**. This number of clusters was determined through visually inspecting the
   distribution of BPCs in 3D principal-component space (`bpc_group.plot_pca_space`)
   as well as individual inspection of each BPC shape.
4. The centroid of each cluster in PCA space was found, projected back into signal
   space, and unweighted. Each of these signals now represents the **consensus** BPC.

Call: `group_bpc_analysis_kmeans(lateral, n_dims=7, n_clusters=2, snr_window=(1.0, 10))`.

---

## 4 · Validation of consensus BPCs
**Code:** `functions/ari_validation.py`.

1. Two different sets of partitions were formed from the existing dataset. One set
   consisted of partitions where a single patient was left out of the entire dataset,
   while another set was partitions where random groups of ten electrodes were left
   out.
2. The cross-patient clustering algorithm (§3) was then used for each partition. The
   Adjusted Rand Index (ARI) was calculated between the resulting clusters of every
   pair of partitions in the dataset.
3. The pairwise ARI for each set were plotted in a scatterplot to verify that no one
   partition pair had abnormally low ARI values.

---

## 5 · Spectrogram & broadband by consensus BPC
**Code:** `functions/spectrogram.py`, `functions/broadband.py`, `functions/spectral_common.py`,
`functions/artifact_removal.py`. Driver notebook: `code/spectral_broadband.ipynb`.
This reproduces the Figure-5 analysis of Huang et al. (2023) (`bpcSpectraBB.m`).

For each (recording electrode × stimulation pair) connection assigned to a consensus
BPC (mean SNR ≥ 1), the stimulation artifact is first mitigated: the ±8 ms around
stimulation is blanked and refilled with a cross-fade of the reflected adjacent
windows (`artifact_removal.remove_stim_artifact`; Crowther et al. 2019).

### Spectrograms — `spectrogram.group_bpc_spectrograms`
1. A complex-Morlet continuous wavelet transform (`cmor`, bandwidth 1.5, center 1.0)
   is computed at 67 log-spaced frequencies from 2 to 200 Hz (10 voices per octave);
   power = |coefficient|².
2. Each frequency is normalized to a **decibel baseline** — `log10` power minus its
   mean over the `[−500, −100] ms` window (equivalently, dividing power by the
   geometric-mean baseline power) — and then averaged across trials (geometric mean).
   A geometric-mean / dB baseline is used rather than an arithmetic-mean-power baseline
   (`log10(power / mean(power))`), which would carry a per-frequency negative offset
   that biases the vs-rest test.
3. For each consensus BPC, the connection spectrograms of every assigned stimulation
   site are stacked and a one-sample *t* statistic against 0 is taken at each
   time-frequency bin. The algorithmically excluded connections are treated the same
   way as a negative control.

### Broadband — `broadband.group_bpc_broadband`
1. The signal is band-pass filtered to **70–170 Hz** (zero-phase, 3rd-order
   Butterworth), and its analytic-signal (Hilbert) envelope is taken; broadband
   power = `log10(|Hilbert|²)`.
2. Each trial's mean broadband over the `[−500, −100] ms` baseline is subtracted (a dB
   baseline), then the trials are averaged → the connection's broadband time course.
3. For each consensus BPC, the site traces are averaged (mean ± SEM across sites) and a
   one-sample *t* statistic against 0 is taken at each time sample.

### Significance — `spectral_common.fdr_significant_pooled`
Time-frequency bins (spectrograms) and samples (broadband) significantly different
from rest are identified by the one-sample *t* tests above and controlled for multiple
comparisons using a **5% false discovery rate (FDR) under dependency**, as implemented
by Benjamini and Yekutieli (2001). This method is used because nearby time-frequency
bins / samples often exhibit positive or negative regression dependency with each
other. The correction is applied **pooled across all consensus BPCs and the excluded
control at once** (as in `bpcSpectraBB.m`). For display, the spectrogram shows the
per-BPC *t*-statistic heatmap with the significant regions kept — the non-significant
bins are whited out (or, optionally, the significant regions are outlined via
`sig_style='outline'`); the broadband trace carries a significance bar colored **red**
where power increases and **blue** where it decreases.

---

## References
- Miller, Müller & Hermes (2021). 
- Huang, Gregg, Ojeda Valencia, Brinkmann, Lundstrom, Worrell, Miller & Hermes (2023).
- Benjamini & Yekutieli (2001). 
- Mewett, Reynolds & Nazeran (2004). 
- Crowther et al. (2019). 