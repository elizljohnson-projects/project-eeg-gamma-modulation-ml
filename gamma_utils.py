import glob
import os
import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from scipy import signal as sp_signal
from specparam import SpectralModel
from bycycle.features import compute_features

def load_eeg_files(data_dir = '.', pattern = '*/data_clean.mat'):
    """
    Load all MATLAB EEG files from directory and convert to MNE Epochs objects.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing .mat files (default: current directory)
    pattern : str
        Glob pattern for finding files (default: '*/data_clean.mat')
    
    Returns:
    --------
    dict : Dictionary where keys are subject IDs and values are mne.Epochs objects
           (epoch x channel x time)
    """
    # find all .mat files
    mat_files = glob.glob(os.path.join(data_dir, pattern), recursive = True)
    print(f'\nFound {len(mat_files)} files\n')
    
    all_data = {}
    
    # load all files
    for filepath in mat_files:
        # extract subject ID from parent directory
        sid = os.path.basename(os.path.dirname(filepath))
        
        print(f'Loading {sid}...')
        
        # load file
        mat_contents = loadmat(filepath, squeeze_me = True, struct_as_record = False)

        # extract data structure
        struct = mat_contents['data']

        # convert to mne.Epochs (n_epochs × n_channels × n_timepoints)
        epochs_data = np.array([t for t in struct.trial])

        # get time vector from 1st trial
        tmin = float(struct.time[0][0])

        # get channel list
        ch_names = [str(ch) for ch in struct.label]

        # create mne.Info object
        info = mne.create_info(
            ch_names = ch_names,
            sfreq = float(struct.fsample),
            ch_types = 'eeg'
        )

        # create mne.EpochsArray
        epochs = mne.EpochsArray(
            data = epochs_data,
            info = info,
            tmin = tmin,
            verbose = False
        )

        all_data[sid] = epochs
    
    return all_data

def compute_psd(all_data, channel = 'Cz'):
    """
    Compute PSD (1-55 Hz) for the specified channel.
    
    Parameters:
    ----------
    all_data : dict
        Dictionary of MNE Epochs objects from load_eeg_files()
    channel : str
        Channel to analyze (default: 'Cz')
    
    Returns:
    -------
    psd_results : dict
        Dictionary with keys like 'KS' containing PSD, channel, and frequencies
    """
    # set frequencies
    fmin, fmax = 1, 55
    bandwidth = 2
    
    psd_results = {}
    
    # Loop through subjects
    for sid in all_data.keys():
        print(f'Computing PSD for {sid}...')
        
        # get data
        epochs = all_data[sid]
        
        # cut to 0-500 ms stimulus window
        epochs_stim = epochs.copy().crop(tmin = 0, tmax = 0.5)
        
        # get channel index
        try:
            ch_idx = epochs_stim.ch_names.index(channel)
        except ValueError:
            print(f'  Channel {channel} not found, skipping {sid}')
            continue
        
        # get PSD for this channel
        data = epochs_stim.get_data()
        n_times_orig = data.shape[2]
        n_fft = 2048
        
        # zero-pad on time axis
        pad_length = n_fft - n_times_orig
        data_padded = np.pad(data[:, ch_idx:ch_idx + 1, :], ((0, 0), (0, 0), (0, pad_length)), mode = 'constant')
        
        # compute PSD
        psd, freqs = mne.time_frequency.psd_array_multitaper(
            data_padded,
            sfreq = epochs_stim.info['sfreq'],
            fmin = fmin,
            fmax = fmax,
            bandwidth = bandwidth,
            adaptive = False,  # equal taper weights
            low_bias = True,  # Attempt to reduce leakage
            normalization = 'full',  # PSD
            verbose = False
        )
        
        # store results
        psd_results[sid] = {
            'psd': psd,
            'channel': channel,
            'freqs': freqs
        }
    
    return psd_results

def fit_specparam(psd_results):
    """
    Fit specparam models to PSD per trial.
    
    Parameters:
    ----------
    psd_results : dict
        Dictionary from compute_psd() with PSD data
    
    Returns:
    -------
    specparam_results : dict
        Dictionary with specparam fits and peak parameters per trial for each subject
        - Peak frequency (Hz): Center frequency of the detected gamma peak
        - Peak power: Height of the oscillatory component above the aperiodic background
        - Peak bandwidth (Hz): Width of the peak (1-10 Hz)
        - Peak SNR: Ratio of oscillatory power to aperiodic power at the peak frequency
        - Aperiodic offset: Y-intercept of the aperiodic background, reflecting overall power
        - Aperiodic slope: Rate of power decay across frequencies, reflecting neural noise
    """
    specparam_results = {}
    
    # Loop through subjects
    for sid in psd_results.keys():
        print(f'Fitting {sid}...')
        
        # get data
        psd = psd_results[sid]['psd']  # (n_epochs, 1, n_freqs)
        freqs = psd_results[sid]['freqs']

        # set specparam frequency range
        freq_range = [freqs[0], freqs[-1]]
        
        # get PSD
        psd_ch = psd[:, 0, :]  # (n_epochs, n_freqs)
        
        # store fits, peak params, and aperiodic params per trial
        trial_fits = []
        peak_params = []
        ap_offsets = []
        ap_slopes = []
        
        for trial_idx in range(psd_ch.shape[0]):
            # initialize model
            fm = SpectralModel(peak_width_limits = [1, 10], min_peak_height = 0.1)
            
            # fit model
            fm.fit(freqs, psd_ch[trial_idx, :], freq_range)
            
            # store fit
            trial_fits.append(fm)
            
            # extract aperiodic parameters
            offset = fm.aperiodic_params_[0]
            slope = -fm.aperiodic_params_[1]  # convert exponent to actual slope
            
            ap_offsets.append(offset)
            ap_slopes.append(slope)
            
            # extract peak parameters
            if fm.n_peaks_ > 0:
                # find peaks in the 30-50 Hz range
                peak_cfs = fm.peak_params_[:, 0]
                in_range = (peak_cfs > 30) & (peak_cfs < 50)
                
                if np.any(in_range):
                    # find peak closest to 40 Hz
                    valid_peaks = fm.peak_params_[in_range, :]
                    valid_cfs = valid_peaks[:, 0]
                    closest_idx = np.argmin(np.abs(valid_cfs - 40))
                    peak_cf, peak_pw, peak_bw = valid_peaks[closest_idx, :]
                    
                    # extract aperiodic power at peak frequency
                    ap_power = fm._ap_fit[np.argmin(np.abs(freqs - peak_cf))]
                    
                    # compute SNR at peak frequency as oscillatory power / aperiodic power
                    snr = peak_pw / ap_power
                    
                    peak_params.append((peak_cf, peak_pw, peak_bw, snr))
                else:
                    peak_params.append((np.nan, np.nan, np.nan, np.nan))
            else:
                peak_params.append((np.nan, np.nan, np.nan, np.nan))
        
        # store results
        specparam_results[sid] = {
            'trial_fits': trial_fits,
            'peak_params': np.array(peak_params),  # center_freq, power, bandwidth, SNR
            'ap_params': np.array([ap_offsets, ap_slopes]),  # aperiodic offset, slope
            'channel': psd_results[sid]['channel'],
            'freqs': freqs
        }
    
    return specparam_results

def extract_ssep(all_data, specparam_results, channel = 'Cz'):
    """
    Extract bandpass-filtered signals at each trial's center frequency and bandwidth 
    for specified channel.
    
    Parameters:
    -----------
    all_data : dict
        Dictionary of MNE Epochs objects from load_eeg_files()
    specparam_results : dict
        Dictionary of specparam results from fit_specparam()
    channel : str
        Channel to analyze (default: 'Cz')
    
    Returns:
    --------
    ssep_results : dict
        Dictionary with bandpass-filtered signals per trial for each subject
    """    
    ssep_results = {}
    
    # loop through subjects
    for sid in all_data.keys():
        print(f'Extracting SSEPs for {sid}...')
        
        # get epochs
        epochs = all_data[sid]
        
        # crop to 0-500 ms window
        epochs_stim = epochs.copy().crop(tmin = 0, tmax = 0.5)
        
        # get channel index
        try:
            ch_idx = epochs_stim.ch_names.index(channel)
        except ValueError:
            print(f'  Channel {channel} not found, skipping {sid}')
            continue
        
        # get sampling rate
        fs = epochs_stim.info['sfreq']
        
        # get peak parameters for this subject
        peak_params = specparam_results[sid]['peak_params']
        
        # store filtered signals per trial
        filtered_signals = []
        filter_params = []
        
        for trial_idx in range(len(epochs_stim)):
            # get raw signal for this trial and channel
            signal = epochs_stim.get_data()[trial_idx, ch_idx, :]
            
            # get peak center frequency and bandwidth for this trial
            peak_cf = peak_params[trial_idx, 0]
            peak_bw = peak_params[trial_idx, 2]
            
            # skip if no peak detected
            if np.isnan(peak_cf):
                filtered_signals.append(np.full_like(signal, np.nan))
                filter_params.append([np.nan, np.nan, np.nan, np.nan])
                continue
            
            # define bandpass filter range using specparam outputs
            f_low = peak_cf - peak_bw / 2
            f_high = peak_cf + peak_bw / 2
            
            try:
                # design butterworth bandpass filter with lower order for stability
                order = 2
                sos = butter(order, [f_low, f_high], btype = 'bandpass', fs = fs, output = 'sos')
                
                # apply zero-phase filter using sos
                signal_filtered = sosfiltfilt(sos, signal)
                
                # check for invalid output
                if np.any(np.isnan(signal_filtered)) or np.any(np.isinf(signal_filtered)):
                    filtered_signals.append(np.full_like(signal, np.nan))
                else:
                    filtered_signals.append(signal_filtered)
                    
            except:
                filtered_signals.append(np.full_like(signal, np.nan))
            
            filter_params.append([peak_cf, peak_bw, f_low, f_high])
        
        # store results
        ssep_results[sid] = {
            'filtered_signals': np.array(filtered_signals),
            'filter_params': np.array(filter_params), 
            'channel': channel,
            'fs': fs,
            'time': epochs_stim.times
        }
    
    return ssep_results

def analyze_ssep(ssep_results):
    """
    Extract temporal dynamics features from filtered SSEP signals.
    
    Parameters:
    -----------
    ssep_results : dict
        Dictionary from extract_bandpass_signals() with filtered SSEP signals
    
    Returns:
    --------
    temporal_results : dict
        Dictionary with temporal features per trial for each subject:
        - Response latency (ms): Time from stimulus onset until SSEP response reaches 50% max amplitude
        - Amplitude slope (μV/s): Rate of change in SSEP amplitude, indicating response stability
        - Autocorrelation (1 cycle): Correlation of the signal with itself shifted by 1 cycle, 
          measuring oscillation consistency
    """    
    temporal_results = {}
    
    # loop through subjects
    for sid in ssep_results.keys():
        print(f'Computing temporal features for {sid}...')
        
        # get filtered signals and parameters
        filtered_signals = ssep_results[sid]['filtered_signals']
        filter_params = ssep_results[sid]['filter_params']
        fs = ssep_results[sid]['fs']
        
        # time vector for the 500 ms window
        n_times = filtered_signals.shape[1]
        time_vec = np.arange(n_times) / fs  # in sec
        
        # store features per trial
        trial_features = {
            'response_latency': [],
            'amplitude_slope': [],
            'autocorr_1cycle': []
        }
        
        for trial_idx in range(filtered_signals.shape[0]):
            signal = filtered_signals[trial_idx, :]
            peak_cf = filter_params[trial_idx, 0]
            
            # skip if no valid signal (nan from missing peak)
            if np.any(np.isnan(signal)) or np.isnan(peak_cf):
                trial_features['response_latency'].append(np.nan)
                trial_features['amplitude_slope'].append(np.nan)
                trial_features['autocorr_1cycle'].append(np.nan)
                continue
            
            # 1. response latency
            # compute envelope using Hilbert transform
            analytic_signal = sp_signal.hilbert(signal)
            envelope = np.abs(analytic_signal)
            
            # smooth envelope with 50 ms window
            window_size = int(0.05 * fs)
            envelope_smooth = np.convolve(envelope, np.ones(window_size) / window_size, mode = 'same')
            
            # threshold: 50% of max envelope
            threshold = 0.5 * np.max(envelope_smooth)
            
            # find first time point exceeding threshold
            above_thresh = np.where(envelope_smooth > threshold)[0]
            if len(above_thresh) > 0:
                latency_samples = above_thresh[0]
                latency_ms = (latency_samples / fs) * 1000  # convert to ms
            else:
                latency_ms = np.nan
            
            trial_features['response_latency'].append(latency_ms)
            
            # 2. amplitude slope
            # fit linear trend to envelope
            slope, intercept = np.polyfit(time_vec, envelope_smooth, 1)
            trial_features['amplitude_slope'].append(slope)
            
            # 3. autocorrelation at 1-cycle lag
            # compute lag based on trial's peak frequency
            lag_seconds = 1 / peak_cf
            lag_samples = int(lag_seconds * fs)
            
            # ensure lag doesn't exceed signal length
            if lag_samples >= len(signal):
                trial_features['autocorr_1cycle'].append(np.nan)
            else:
                # compute autocorrelation using Pearson correlation
                sig1 = signal[:-lag_samples]
                sig2 = signal[lag_samples:]
                
                autocorr = np.corrcoef(sig1, sig2)[0, 1]
                trial_features['autocorr_1cycle'].append(autocorr)
        
        # convert to arrays
        for key in trial_features.keys():
            trial_features[key] = np.array(trial_features[key])
        
        # store results
        temporal_results[sid] = {
            **trial_features,
            'channel': ssep_results[sid]['channel']
        }
    
    return temporal_results

def analyze_bycycle(all_data, specparam_results, channel = 'Cz'):
    """
    Compute bycycle features for SSEPs per trial using raw signals for specified channel.
    
    Parameters:
    -----------
    all_data : dict
        Dictionary of MNE Epochs objects from load_eeg_files()
    specparam_results : dict
        Dictionary of specparam results from fit_specparam()
    channel : str
        Channel to analyze (default: 'Cz')
    
    Returns:
    --------
    bycycle_results : dict
        Dictionary with mean bycycle features per trial for each subject:
        - Amplitude: Peak-to-trough voltage difference, indicating oscillation strength
        - Period (ms): Duration of one oscillation cycle
        - Rise time (ms): Duration from trough to peak, indicating ascending phase speed
        - Decay time (ms): Duration from peak to trough, indicating descending phase speed
        - Rise-decay symmetry: Ratio comparing rise and decay times, where values near 0.5 indicate symmetric waveforms
        - Peak voltage: Voltage at the peak of each cycle
        - Trough voltage: Voltage at the trough of each cycle
        - Burst rate: Proportion of detected cycles meeting burst criteria, indicating sustained oscillatory activity
    """
    
    # gamma frequency range
    f_range = (30, 50)
    
    # burst detection thresholds for 40-Hz SSEPs
    threshold_kwargs = {
        'amp_fraction_threshold': 0.2,  # SSEPs have consistent amplitudes, so cycles should be >20% of median
        'amp_consistency_threshold': 0.6,  # allow variability for burst cycles
        'period_consistency_threshold': 0.6,  # 40 Hz is stable, so periods should be consistent w/ some jitter
        'monotonicity_threshold': 0.6,  # SSEPs should have clean rise/decay, but not too strict b/c there is noise
        'min_n_cycles': 2  # 2 SB window means ~20 cycles at 40 Hz, so 3 consecutive cycles is reasonable
    }
    
    bycycle_results = {}
    
    # loop through subjects
    for sid in all_data.keys():
        print(f'Computing bycycle features for {sid}...')
        
        # get epochs
        epochs = all_data[sid]
        
        # crop to 0-500 ms window
        epochs_stim = epochs.copy().crop(tmin = 0, tmax = 0.5)
        
        # get channel index
        try:
            ch_idx = epochs_stim.ch_names.index(channel)
        except ValueError:
            print(f'  Channel {channel} not found, skipping {sid}')
            continue
        
        # get sampling rate
        fs = epochs_stim.info['sfreq']
        
        # store features per trial
        trial_features = {
            'amplitude': [],
            'period': [],
            'rise_time': [],
            'decay_time': [],
            'rise_decay_symmetry': [],
            'peak_voltage': [],
            'trough_voltage': [],
            'burst_rate': []
        }
        
        for trial_idx in range(len(epochs_stim)):
            # get raw signal for this trial and channel
            signal = epochs_stim.get_data()[trial_idx, ch_idx, :]
            
            # get peak frequency from specparam for this trial
            peak_cf = specparam_results[sid]['peak_params'][trial_idx, 0]
            
            # skip if no peak detected (matches specparam behavior)
            if np.isnan(peak_cf):
                for key in trial_features.keys():
                    trial_features[key].append(np.nan)
                continue
            
            # compute bycycle features on raw signal
            df = compute_features(
                signal,
                fs,
                f_range = f_range,
                center_extrema = 'peak',
                threshold_kwargs = threshold_kwargs
            )
            
            # check if any cycles were detected
            if len(df) == 0:
                for key in trial_features.keys():
                    trial_features[key].append(np.nan)
                continue
            
            # compute mean across cycles for each feature
            trial_features['amplitude'].append(df['volt_amp'].mean())
            trial_features['period'].append((df['period'].mean() / fs) * 1000)  # convert to ms
            trial_features['rise_time'].append((df['time_rise'].mean() / fs) * 1000)  # convert to ms
            trial_features['decay_time'].append((df['time_decay'].mean() / fs) * 1000)  # convert to ms
            trial_features['rise_decay_symmetry'].append(df['time_rdsym'].mean())
            trial_features['peak_voltage'].append(df['volt_peak'].mean())
            trial_features['trough_voltage'].append(df['volt_trough'].mean())
            
            # burst rate
            burst_rate = df['is_burst'].mean()
            trial_features['burst_rate'].append(burst_rate)
        
        # convert to arrays
        for key in trial_features.keys():
            trial_features[key] = np.array(trial_features[key])
        
        # store results
        bycycle_results[sid] = {
            **trial_features,
            'channel': channel
        }
    
    return bycycle_results

def create_paired_df(specparam_baseline, specparam_sham, specparam_tacs, specparam_tdcs,
                    temporal_baseline, temporal_sham, temporal_tacs, temporal_tdcs,
                    bycycle_baseline, bycycle_sham, bycycle_tacs, bycycle_tdcs):
    """
    Create df with all conditions paired with baseline.
    
    Parameters:
    -----------
    specparam_baseline, specparam_sham, specparam_tacs, specparam_tdcs : dict
        Specparam results from fit_specparam()
    temporal_baseline, temporal_sham, temporal_tacs, temporal_tdcs : dict
        Temporal results from analyze_ssep()
    bycycle_baseline, bycycle_sham, bycycle_tacs, bycycle_tdcs : dict
        Bycycle results from analyze_bycycle()
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with one row per trial, columns for subject ID, trial ID, condition, group, and all 17 features
    """    
    rows = []
    
    # loop through conditions
    for group_name, specparam_condition, temporal_condition, bycycle_condition in [
        ('tACS', specparam_tacs, temporal_tacs, bycycle_tacs),
        ('tDCS', specparam_tdcs, temporal_tdcs, bycycle_tdcs),
        ('Sham', specparam_sham, temporal_sham, bycycle_sham)
    ]:
        
        condition_name = group_name
        
        # loop through subjects in condition
        for sid in specparam_condition.keys():
            # skip if subject not in baseline
            if sid not in specparam_baseline:
                continue
            
            # process baseline trials
            n_trials_baseline = specparam_baseline[sid]['peak_params'].shape[0]
            for tid in range(n_trials_baseline):
                # skip if any feature is NaN (peak detection failed)
                if np.isnan(specparam_baseline[sid]['peak_params'][tid, 0]):
                    continue
                
                row_baseline = {
                    'sid': sid,
                    'tid': tid,
                    'condition': 'Baseline',
                    'group': group_name,
                    # specparam features
                    'peak_frequency': specparam_baseline[sid]['peak_params'][tid, 0],
                    'peak_power': specparam_baseline[sid]['peak_params'][tid, 1],
                    'peak_bandwidth': specparam_baseline[sid]['peak_params'][tid, 2],
                    'peak_snr': specparam_baseline[sid]['peak_params'][tid, 3],
                    'aperiodic_offset': specparam_baseline[sid]['ap_params'][0, tid],
                    'aperiodic_slope': specparam_baseline[sid]['ap_params'][1, tid],
                    # temporal features
                    'response_latency': temporal_baseline[sid]['response_latency'][tid],
                    'amplitude_slope': temporal_baseline[sid]['amplitude_slope'][tid],
                    'autocorr_1cycle': temporal_baseline[sid]['autocorr_1cycle'][tid],
                    # bycycle features
                    'amplitude': bycycle_baseline[sid]['amplitude'][tid],
                    'period': bycycle_baseline[sid]['period'][tid],
                    'rise_time': bycycle_baseline[sid]['rise_time'][tid],
                    'decay_time': bycycle_baseline[sid]['decay_time'][tid],
                    'rise_decay_symmetry': bycycle_baseline[sid]['rise_decay_symmetry'][tid],
                    'peak_voltage': bycycle_baseline[sid]['peak_voltage'][tid],
                    'trough_voltage': bycycle_baseline[sid]['trough_voltage'][tid],
                    'burst_rate': bycycle_baseline[sid]['burst_rate'][tid]
                }
                rows.append(row_baseline)
            
            # process condition trials
            n_trials_condition = specparam_condition[sid]['peak_params'].shape[0]
            for tid in range(n_trials_condition):
                # skip if any feature is NaN (peak detection failed)
                if np.isnan(specparam_condition[sid]['peak_params'][tid, 0]):
                    continue
                
                row_condition = {
                    'sid': sid,
                    'tid': tid,
                    'condition': condition_name,
                    'group': group_name,
                    # specparam features
                    'peak_frequency': specparam_condition[sid]['peak_params'][tid, 0],
                    'peak_power': specparam_condition[sid]['peak_params'][tid, 1],
                    'peak_bandwidth': specparam_condition[sid]['peak_params'][tid, 2],
                    'peak_snr': specparam_condition[sid]['peak_params'][tid, 3],
                    'aperiodic_offset': specparam_condition[sid]['ap_params'][0, tid],
                    'aperiodic_slope': specparam_condition[sid]['ap_params'][1, tid],
                    # temporal features
                    'response_latency': temporal_condition[sid]['response_latency'][tid],
                    'amplitude_slope': temporal_condition[sid]['amplitude_slope'][tid],
                    'autocorr_1cycle': temporal_condition[sid]['autocorr_1cycle'][tid],
                    # bycycle features
                    'amplitude': bycycle_condition[sid]['amplitude'][tid],
                    'period': bycycle_condition[sid]['period'][tid],
                    'rise_time': bycycle_condition[sid]['rise_time'][tid],
                    'decay_time': bycycle_condition[sid]['decay_time'][tid],
                    'rise_decay_symmetry': bycycle_condition[sid]['rise_decay_symmetry'][tid],
                    'peak_voltage': bycycle_condition[sid]['peak_voltage'][tid],
                    'trough_voltage': bycycle_condition[sid]['trough_voltage'][tid],
                    'burst_rate': bycycle_condition[sid]['burst_rate'][tid]
                }
                rows.append(row_condition)
    
    # create df
    df = pd.DataFrame(rows)
    
    return df
