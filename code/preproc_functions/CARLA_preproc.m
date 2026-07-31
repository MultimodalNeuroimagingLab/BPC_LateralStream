function CARLA_preproc(localDataPath, cm, cfg)
%CARLA_PREPROC  SPES/CCEP preprocessing for ONE patient using CARLA re-referencing.
%   CARLA_preproc(localDataPath, cm, cfg) runs preprocessing for every
%   (gray-matter) stim pair of one patient and writes preproc .mat files +
%   QC/imagesc/traces PNGs under derivatives/bsep_basic_analysis.
%                                 CARLA -- "CAR by Least Anticorrelation"
%                                 (Huang et al., J. Neurosci. Methods 2024).
%                                 Channels are ranked by increasing cross-trial
%                                 covariance on the response interval; the
%                                 common-average subset is grown channel-by-
%                                 channel and its size is chosen adaptively at
%                                 the point of least anticorrelation (least
%                                 negative zMinMean). The per-trial common
%                                 average over that optimal subset is subtracted
%                                 from ALL channels. No single reference channel
%                                 is needed, so the per-run/per-hemisphere reref
%                                 config of preproc_patient is gone.
%
%   localDataPath : struct from setLocalDataPath(1) (.iEEG is the BIDS root)
%   cm            : colormap from loc_colormap()
%   cfg fields:
%     .sub             subject id, e.g. 
%     .ses             session, e.g. 'ieeg01'
%     .task            task, e.g. 'ccep'
%     .runs            cellstr of run labels, e.g. {'01','02',...}
%     .ens             0/1 ENS amplifier flag (labels only)        (default 0)
%     .exclude_white   drop pairs where either contact is WM       (default true)
%     .grey_thresh     gm_wm_relativeDistance >= this = gray        (default -1)
%     .do_qc           save the 3-panel raw/CARLA/bipolar QC fig    (default true)
%     .do_final        save the median+colormap imagesc fig         (default true)
%     .do_traces       save the stacked-line-trace fig              (default true)
%     .do_carla_qc     save the CARLA diagnostic fig (zMinMean+var)  (default true)
%     .trace_spacing   uV between channel baselines in traces fig    (default 250)
%     -- CARLA parameters --
%     .sens            true  -> first-peak ("sensitive") cutoff: the n just
%                              before the first statistically-significant drop
%                              in zMinMean (bootstrapped, left-tailed 95% CI).
%                      false -> global optimum: the n at the global maximum
%                              (least negative) zMinMean.            (default true)
%     .nboot           # bootstrap samples for zMinMean / cutoff     (default 100)
%     .notch_output    apply a notch filter to the CARLA-referenced output.
%                      DEFAULT false = stock/paper-faithful CARLA: the output is
%                      UNfiltered and line noise is removed only by the adaptive
%                      re-referencing (CARLA notches internally for SUBSET
%                      SELECTION only, never the output). Set true to additionally
%                      notch the saved output at .notch_freqs.        (default false)
%     .notch_freqs     notch frequencies (Hz) when notch_output=true (default [60 120 180])
%     .carla_path      path to CARLA package /functions dir to addpath
%                      (default /Users/kyung/Documents/CARLA_JNM-main/functions)
%     -- line-noise removal (log-log spectrum interpolation, from yanez_preproc) --
%     .do_linenoise    remove 60 Hz (+harmonic) line noise from the CARLA output
%                      by interpolating the amplitude spectrum across each line band
%                      in log|X| vs log f (phase preserved), applied to figures AND
%                      the saved prep.data. Independent of .notch_output (a stock IIR
%                      notch); leave notch_output=false to use interpolation alone.
%                                                                     (default true)
%     .line_freq       fundamental line frequency (Hz)               (default 60)
%     .line_halfbw     half-width (Hz) of the interpolated band       (default 1.5)
%     .line_anchor_bw  width (Hz) of the outside-band anchor windows   (default 2)
%     .line_max_harm   # harmonics to clean (3 = 60/120/180 Hz;
%                      [] = up to Nyquist)                            (default 3)
%
%
% ---------- defaults (fill only missing fields) ----------
def = struct('ens',0,'exclude_white',true,'grey_thresh',-1, ...
             'do_qc',true,'do_final',true,'do_traces',true,'do_carla_qc',true, ...
             'trace_spacing',250, ...
             'sens',true,'nboot',100,'notch_output',false, ...
             'do_linenoise',true,'line_freq',60,'line_halfbw',1.5,'line_anchor_bw',2, ...
             'carla_path','/Users/kyung/Documents/CARLA_JNM-main/functions');
fn = fieldnames(def);
for k = 1:numel(fn)
    if ~isfield(cfg,fn{k}), cfg.(fn{k}) = def.(fn{k}); end
end
if ~isfield(cfg,'notch_freqs'),  cfg.notch_freqs  = [60 120 180]; end
if ~isfield(cfg,'line_max_harm'), cfg.line_max_harm = 3; end   % 60/120/180 Hz
% exclude_channels kept separate ({} can't go in struct() literal):
% {} | cellstr (all runs) | containers.Map(run -> cellstr). Corrupt/unreadable
% contacts to drop before any readMef3 
if ~isfield(cfg,'exclude_channels'), cfg.exclude_channels = {}; end

% ---------- make CARLA available on the path ----------
if ~isempty(cfg.carla_path) && exist(cfg.carla_path,'dir')
    addpath(genpath(cfg.carla_path));
end
assert(exist('CARLA','file')==2, ...
    'CARLA.m not found on path. Set cfg.carla_path to the CARLA package /functions dir.');

% ---------- output dirs ----------
clinicDir  = fullfile(localDataPath.iEEG,'derivatives','bsep_basic_analysis',['sub-' cfg.sub]);
preprocDir = fullfile(clinicDir,'preproc');
if ~exist(clinicDir,'dir'),  mkdir(clinicDir);  end
if ~exist(preprocDir,'dir'), mkdir(preprocDir); end

% ---------- get stim info ----------
stim = get_stim(localDataPath.iEEG, cfg.sub, cfg.runs, cfg.ses, cfg.task);

% ---------- gray-matter-only stim-pair filter ----------
if cfg.exclude_white
    loc_info_path = fullfile(localDataPath.iEEG,'derivatives','loc_info',['sub-' cfg.sub],'loc_info.mat');
    if exist(loc_info_path,'file')
        S = load(loc_info_path);                       % variable 'loc_info' (a table)
        grey_elec = S.loc_info.name(S.loc_info.gm_wm_relativeDistance >= cfg.grey_thresh);
        % get_stim returns pair names with intra-name hyphens already stripped
        grey_elec = erase(grey_elec, '-');
        keep = false(numel(stim.pairs),1);
        for ii = 1:numel(stim.pairs)
            e = split(stim.pairs{ii},'-');
            keep(ii) = ismember(e{1},grey_elec) && ismember(e{2},grey_elec);
        end
        ff = {'pairs','events','channels','mef','run_name'};
        for f = 1:numel(ff), stim.(ff{f}) = stim.(ff{f})(keep); end
        fprintf('  %s: kept %d/%d gray-matter stim pairs\n', cfg.sub, sum(keep), numel(keep));
    else
        warning('exclude_white=true but %s not found; keeping all pairs.', loc_info_path);
    end
end

% ---------- save stim info ----------
save(fullfile(clinicDir,[cfg.sub '_stimInfo']), 'stim');

% ---------- preprocess each stim pair (uses existing parallel pool if open) ----------
parfor ss = 1:numel(stim.pairs)
    preproc_one_pair_carla(ss, stim, cfg, cm, clinicDir, preprocDir);
end
end


% =====================================================================
function preproc_one_pair_carla(ss, stim, cfg, cm, clinicDir, preprocDir)
disp(['  stim pair ' int2str(ss) ' of ' int2str(numel(stim.pairs)) ': ' stim.pairs{ss}])

% ensure CARLA is on the worker path too (parfor workers may not inherit it)
if exist('CARLA','file')~=2 && ~isempty(cfg.carla_path) && exist(cfg.carla_path,'dir')
    addpath(genpath(cfg.carla_path));
end

channelsInfo = readtable(stim.channels{ss},'FileType','text','Delimiter','\t','TreatAsEmpty',{'N/A','n/a'});

% drop corrupt/unreadable channels BEFORE any read (uncatchable readMef3 bus error)
ex = resolve_exclude(cfg.exclude_channels, stim.run_name{ss});
if ~isempty(ex)
    drop = ismember(channelsInfo.name, ex);
    if any(drop)
        warning('%s run-%s: excluding channel(s) %s', stim.pairs{ss}, stim.run_name{ss}, strjoin(channelsInfo.name(drop)',', '));
        channelsInfo(drop, :) = [];
    end
end

stim_pair = stim.pairs{ss};
el1 = extractBefore(stim_pair,'-');
el2 = extractAfter(stim_pair,'-');

% events for this stim site, good only
evts = ieeg_readtableRmHyphens(stim.events{ss},'electrical_stimulation_site',1);
evts(~ismember(evts.electrical_stimulation_site, stim_pair), :) = [];
evts(ismember(evts.status,'bad'), :) = [];

% load data (trange seconds around each stim onset)
metadata = readMef3(stim.mef{ss});
srate = metadata.time_series_metadata.section_2.sampling_frequency;
trange = [-3 3];
tt = (0:(trange(end)-trange(1))*srate-1)/srate + trange(1);
ranges = round([evts.onset*srate + tt(1)*srate, ...
                evts.onset*srate + tt(1)*srate + length(tt)]);
[~, data] = readMef3(stim.mef{ss}, [], channelsInfo.name, 'samples', ranges);

% strip hyphens from channel names to match stim-pair names
for ii = 1:numel(channelsInfo.name)
    channelsInfo.name(ii) = join(split(channelsInfo.name(ii),'-'),'');
end

% good recording channels, excluding the stim pair and its neighbors (+/- 2)
good_channels = find(ismember(channelsInfo.status,'good') & ...
    ismember(channelsInfo.type,{'ECOG','SEEG'}) & ~ismember(channelsInfo.name,{el1 el2}));
neighborStim_channels = [min(find(ismember(channelsInfo.name,{el1 el2})))-[2 1] ...
                         max(find(ismember(channelsInfo.name,{el1 el2})))+[1 2]];
neighborStim_channels(neighborStim_channels<1) = [];
plot_channels      = setdiff(good_channels, neighborStim_channels);
plot_channel_names = {channelsInfo.name{plot_channels}}';

% baseline-correct DC (per-trial pre-stim offset; preserved from preproc_patient)
baseRange = [-0.100, -0.010];
these_data1 = data - mean(data(:, tt>=baseRange(1) & tt<=baseRange(end), :), 2);

% amplifier label + (Natus) headbox boundary for the QC figure only
if srate == 4800
    amplifier = 'gtec';
else
    if cfg.ens==1, amplifier = 'ens'; else, amplifier = 'natus'; end
end
idx_headbox = 0;
if srate ~= 4800
    headboxes = channelsInfo.headbox(plot_channels);
    idx_headbox = find(diff(headboxes)~=0)+0.5;
    if isempty(idx_headbox), idx_headbox = 0; end
end

% ================== CARLA re-referencing ==================
% Candidate channels for the adaptive common average = the good (plot) channels.
% Guard against channels that would break covariance/correlation (all-NaN or flat
% on the response interval, e.g. a zero-filled corrupt contact).
respwin = tt >= 0.01 & tt <= 0.3;                              % CARLA response interval
Vcand   = these_data1(plot_channels, :, :);
isNanCh = any(isnan(Vcand), [2 3]);
isFlat  = max(var(Vcand(:, respwin, :), 0, 2), [], 3) == 0;
okCarla = ~(isNanCh | isFlat);
carla_channels = plot_channels(okCarla);                       % channel numbers fed to CARLA
if any(~okCarla)
    warning('%s run-%s: %d candidate channel(s) dropped from CARLA (NaN/flat).', ...
            stim_pair, stim.run_name{ss}, sum(~okCarla));
end

% Run CARLA on the candidate channels. CARLA notch-filters internally only to
% pick the subset; the returned CAR is built from the UNfiltered input.
[~, CAR, cstats] = CARLA(tt, these_data1(carla_channels, :, :), srate, cfg.sens, cfg.nboot);

% Subtract the per-trial adaptive common average from ALL channels.
CARrs      = reshape(CAR, 1, numel(tt), []);                   % 1 x time x trials (broadcasts over channels)
these_data = these_data1 - CARrs;

% channels (by original channel number / name) that formed the optimal CAR
car_used_channels = carla_channels(cstats.chsUsed);
car_used_names    = channelsInfo.name(car_used_channels);
nCAR              = numel(cstats.chsUsed);

% Optional notch of the CARLA-referenced output (off => paper-faithful unfiltered output)
if cfg.notch_output
    these_data = notch_trials(these_data, srate, cfg.notch_freqs);
end
% ==========================================================

% --- final: line-noise removal by log-log spectrum interpolation (from yanez_preproc) ---
% Replaces the 60 Hz (+harmonic) line peaks in the amplitude spectrum with a
% straight line in log|X| vs log f (anchored just outside each band), keeping the
% original phase. Applied here so every downstream output (figures + saved
% prep.data) is line-noise-cleaned. Independent of notch_output above.
if cfg.do_linenoise
    these_data = spectrum_interp_loglog(these_data, srate, cfg.line_freq, ...
                     cfg.line_halfbw, cfg.line_anchor_bw, cfg.line_max_harm);
end

% --- QC figure: raw / CARLA / bipolar (mean across trials) ---
if cfg.do_qc
    figure('Position',[0 0 1200 1000])
    subplot(1,3,1)
    imagesc(tt,1:numel(plot_channels),mean(data(plot_channels,:,:),3),[-300 300])
    xlim([-.1 1]); xlabel('time(s)'); yline(idx_headbox,'LineWidth',1.5);
    set(gca,'YTick',1:numel(plot_channels),'YTickLabel',plot_channel_names,'FontSize',6)
    title(['stim ' stim_pair '  no ref - ' amplifier])
    subplot(1,3,2)
    imagesc(tt,1:numel(plot_channels),mean(these_data(plot_channels,:,:),3),[-300 300])
    set(gca,'YTick',1:numel(plot_channels),'YTickLabel',plot_channel_names,'FontSize',6)
    xlim([-.1 1]); xlabel('time(s)'); yline(idx_headbox,'LineWidth',1.5);
    title(sprintf('CARLA (adaptive CAR, %d/%d channels)', nCAR, numel(carla_channels)))
    bipolar = diff(these_data1(plot_channels,:,:));
    subplot(1,3,3)
    imagesc(tt,1:numel(plot_channels)-1,mean(bipolar,3),[-300 300])
    set(gca,'YTick',1:numel(plot_channels)-1,'YTickLabel',plot_channel_names(1:end-1),'FontSize',6)
    xlim([-.1 1]); xlabel('time(s)'); title('Bipolar')
    set(gcf,'PaperPositionMode','auto')
    print('-dpng','-r300', fullfile(clinicDir,[cfg.sub '_' stim_pair '_run-' stim.run_name{ss} '_qc']))
end

% --- CARLA diagnostic figure: zMinMean curve + sorted (co)variance ---
if cfg.do_carla_qc
    cmSens = [1, 165/255, 0];                                  % orange marker for chosen n
    zmm = cstats.zMinMean;                                     % (nCand x nboot) if >1 trial, else (1 x nCand)
    figure('Position',[200 200 900 320])
    subplot(1,2,1); hold on
    if size(evts,1) > 1 && size(zmm,2) > 1                     % bootstrapped: show mean +/- SD
        errorbar(mean(zmm,2), std(zmm,0,2), 'k.-', 'MarkerSize',10, 'CapSize',1);
        plot(nCAR, mean(zmm(nCAR,:),2), '*', 'Color', cmSens, 'MarkerSize',10);
    else                                                       % single trial: zmm is 1 x nCand
        plot(zmm(:), 'k.-', 'MarkerSize',10);
        plot(nCAR, zmm(nCAR), '*', 'Color', cmSens, 'MarkerSize',10);
    end
    yline(0,'Color','k'); xlim([0 numel(carla_channels)+1]);
    xlabel('CAR subset size n'); ylabel('zMinMean');
    title(sprintf('%s run %s  (n_{opt}=%d, sens=%d)', stim_pair, stim.run_name{ss}, nCAR, cfg.sens))
    subplot(1,2,2); hold on
    plot(cstats.vars(cstats.order), 'k.-', 'MarkerSize',10);   % (co)variance, increasing order
    xline(nCAR + 0.5, 'Color', cmSens);
    xlim([0 numel(carla_channels)+1]);
    xlabel('channel (sorted)'); ylabel('cross-trial covariance');
    title('CARLA channel ranking')
    set(gcf,'PaperPositionMode','auto')
    print('-dpng','-r300', fullfile(clinicDir,[cfg.sub '_' stim_pair '_run-' stim.run_name{ss} '_carla']))
end

% --- final figure: median across trials + CCEP colormap ---
if cfg.do_final
    these_data_bs = these_data - mean(these_data(:, tt>=baseRange(1) & tt<=baseRange(end), :), 2);
    split_num = min(numel(plot_channels),128);
    if numel(plot_channels) > 128
        figure('Position',[0 0 800 900])
        subplot(1,2,1)
        imagesc(tt,1:split_num,median(these_data_bs(plot_channels(1:split_num),:,:),3),[-300 300])
        xlim([-.1 1]); colormap(cm)
        set(gca,'YTick',1:split_num,'YTickLabel',plot_channel_names(1:split_num))
        title(['preprocessed ' stim_pair ' run ' stim.run_name{ss}])
        subplot(1,2,2)
        imagesc(tt,split_num+1:numel(plot_channels),median(these_data_bs(plot_channels(split_num+1:end),:,:),3),[-300 300])
        xlim([-.1 1]); colormap(cm)
        set(gca,'YTick',split_num+1:numel(plot_channels),'YTickLabel',plot_channel_names(split_num+1:end))
    else
        figure('Position',[0 0 400 900])
        imagesc(tt,1:split_num,median(these_data_bs(plot_channels(1:split_num),:,:),3),[-300 300])
        xlim([-.1 1]); colormap(cm)
        set(gca,'YTick',1:split_num,'YTickLabel',plot_channel_names(1:split_num))
        title(['preprocessed ' stim_pair ' run ' stim.run_name{ss}])
    end
    set(gcf,'PaperPositionMode','auto')
    print('-dpng','-r300', fullfile(clinicDir,[cfg.sub '_' stim_pair '_run-' stim.run_name{ss} '_imagesc']))
end

% --- final figure as stacked LINE traces (line-version of the imagesc) ---
if cfg.do_traces
    bs  = these_data - mean(these_data(:, tt>=baseRange(1) & tt<=baseRange(end), :), 2);
    med = median(bs, 3);                          % ch x time, trial-median (matches imagesc)
    sp  = cfg.trace_spacing;                       % uV between channel baselines
    win = tt >= -0.1 & tt <= 1;                    % display window (matches imagesc xlim)
    split_num = min(numel(plot_channels), 128);
    nseg = ceil(numel(plot_channels)/split_num);
    figure('Position',[0 0 450*nseg 1000])
    for seg = 1:nseg
        idx = (seg-1)*split_num + (1:split_num);
        idx = idx(idx <= numel(plot_channels));    % channel positions in plot_channels
        n   = numel(idx);
        subplot(1,nseg,seg); hold on
        for k = 1:n
            off = (n-k)*sp;                        % k=1 at top, k=n at bottom
            plot(tt(win), off + med(plot_channels(idx(k)), win), 'r', 'LineWidth',0.4);
        end
        xline(0,'r-');
        set(gca,'YTick',(0:n-1)*sp,'YTickLabel',flip(plot_channel_names(idx)),'FontSize',5)
        ylim([-sp, n*sp]); xlim([-.1 1]); xlabel('time (s)')
        title(sprintf('%s run %s  (median, %d uV/div)', stim_pair, stim.run_name{ss}, sp))
    end
    set(gcf,'PaperPositionMode','auto')
    print('-dpng','-r300', fullfile(clinicDir,[cfg.sub '_' stim_pair '_run-' stim.run_name{ss} '_traces']))
end

% --- save preprocessed data ---
prep = struct();
prep.data                 = single(these_data);
prep.tt                   = tt;
prep.srate                = srate;
prep.evts                 = evts;
prep.use_channels         = plot_channels;
prep.use_channels_names   = plot_channel_names;
prep.channels             = channelsInfo;
prep.amplifier            = amplifier;
prep.stim_channels_names  = stim_pair;
% --- CARLA metadata ---
% reref_channels_names is kept a SCALAR CHAR for back-compat: the Python loader
% (load_preproc.load_run) parses this field with _h5_to_str, which only handles a
% char array. The actual CARLA channel list lives in car_channels_names (cellstr).
prep.reref_method         = 'CARLA';
prep.reref_channels_names = 'CARLA';                 % scalar char (drop-in for _h5_to_str)
prep.car_channels_names   = car_used_names;          % cellstr: channels forming the adaptive CAR
prep.car_channels         = car_used_channels;       % their original channel numbers
prep.carla_nOptimum       = nCAR;                    % optimal CAR subset size
prep.carla_sens           = cfg.sens;               % cutoff mode used
prep.carla_order          = carla_channels(cstats.order); % CARLA ranking (channel numbers)
prep.carla_vars           = cstats.vars;            % cross-trial covariance per candidate
prep.carla_zMinMean       = cstats.zMinMean;        % anticorrelation curve
if cfg.do_linenoise
    prep.linenoise_method = sprintf('loglog_spectrum_interp_%gHz_pm%gHz', cfg.line_freq, cfg.line_halfbw);
else
    prep.linenoise_method = 'none';
end
save(fullfile(preprocDir,[cfg.sub '_' stim_pair '_preproc_run-' stim.run_name{ss}]), ...
     '-fromstruct', prep, '-v7.3')

close all
end


% =====================================================================
function V = notch_trials(V, srate, freqs)
% Apply zero-phase IIR notch filters (matches CARLA's internal designfilt) per trial.
for f = freqs(:)'
    dNotch = designfilt('bandstopiir','FilterOrder',4,'DesignMethod','butter', ...
                        'HalfPowerFrequency1',f-2,'HalfPowerFrequency2',f+2, ...
                        'SampleRate',srate);
    for tr = 1:size(V,3)
        V(:,:,tr) = filtfilt(dNotch, V(:,:,tr)')';
    end
end
end


% =====================================================================
function ex = resolve_exclude(exc, run)
% Resolve the per-run channel-exclusion list.
%   exc : {} | char | cellstr (applies to all runs) | containers.Map(run -> cellstr)
if isempty(exc)
    ex = {};
elseif isa(exc, 'containers.Map')
    if isKey(exc, run), ex = exc(run); else, ex = {}; end
else
    ex = exc;                      % cellstr/char applies to every run
end
if ischar(ex), ex = {ex}; end
end


% =====================================================================
function out = spectrum_interp_loglog(data, srate, line_freq, halfbw, anchor_bw, max_harm)
%SPECTRUM_INTERP_LOGLOG  Line-noise removal by log-log spectrum interpolation.
%   Removes 60 Hz (+harmonic) line noise from each channel/trial by replacing the amplitude
%   spectrum inside a narrow band around each harmonic with a straight line fit in
%   LOG-amplitude vs LOG-frequency space, while preserving the original phase, then
%   inverse-FFT. Interpolating on the log-log scale follows the ~1/f background so
%   only the narrow line peak is removed (cf. Mewett 2004 / Leske & Dalal 2014
%   spectrum interpolation, here done in log-log not linear).
%
%   data      : ch x time x trial (FFT is taken along the time dim 2)
%   srate     : sampling rate (Hz)
%   line_freq : fundamental line frequency (Hz), e.g. 60
%   halfbw    : half-width (Hz) of the band that gets interpolated
%   anchor_bw : width (Hz) of the windows just outside the band used to anchor
%               the log-log line (median amplitude there sets each endpoint)
%   max_harm  : number of harmonics to clean; [] = all up to Nyquist
if isempty(data), out = data; return; end
N   = size(data, 2);
nyq = srate / 2;
f   = (0:N-1) * (srate / N);          % two-sided bin frequencies (Hz)

X   = fft(data, [], 2);               % ch x N x trial (complex)
amp = abs(X);
ph  = angle(X);

% bins strictly between DC and Nyquist (leave DC and the Nyquist bin untouched)
posIdx = 2:floor((N-1)/2) + 1;
fpos   = f(posIdx);

% harmonics that fully fit (band + anchor window) below Nyquist
if isempty(max_harm), max_harm = floor(nyq / line_freq); end
harms = line_freq * (1:max_harm);
harms = harms(harms + halfbw + anchor_bw < nyq);

for h = harms
    band = posIdx(fpos >= h - halfbw       & fpos <= h + halfbw);
    lo   = posIdx(fpos >= h - halfbw - anchor_bw & fpos <  h - halfbw);
    hi   = posIdx(fpos >  h + halfbw       & fpos <= h + halfbw + anchor_bw);
    if isempty(band) || isempty(lo) || isempty(hi), continue; end

    fL = f(lo(end));   fR = f(hi(1));                 % anchor (band-edge) freqs
    AL = max(median(amp(:, lo, :), 2), realmin);      % ch x 1 x trial endpoints
    AR = max(median(amp(:, hi, :), 2), realmin);

    w  = (log(f(band)) - log(fL)) / (log(fR) - log(fL));   % 1 x nb in [0,1]
    logA = log(AL) + reshape(w, 1, [], 1) .* (log(AR) - log(AL));  % ch x nb x trial
    amp(:, band, :) = exp(logA);
end

% rebuild with original phase, restore Hermitian symmetry, invert to real signal
Xnew = amp .* exp(1i * ph);
neg  = N - posIdx + 2;                 % conjugate-symmetric partners of posIdx
Xnew(:, neg, :) = conj(Xnew(:, posIdx, :));
out  = real(ifft(Xnew, [], 2));
end
