import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_psd(psd_baseline, psd_sham, psd_tacs, psd_tdcs):
    """
    Plot PSD across subjects for each condition vs. baseline.
    
    Parameters:
    ----------
    psd_baseline : dict
        Baseline PSD data from compute_psd()
    psd_sham, psd_tacs, psd_tdcs : dict
        Dictionaries from compute_psd()
    """
    fig, axes = plt.subplots(1, 3, figsize = (9, 3), sharey = True)
    
    # colors for each condition
    colors = {
        'tACS': 'orange',
        'tDCS': 'green',
        'Sham': 'dodgerblue',
        'Baseline': 'dimgray'
    }
    
    # process each condition
    for ax_idx, (condition_name, psd_data, color) in enumerate([
        ('tACS', psd_tacs, colors['tACS']),
        ('tDCS', psd_tdcs, colors['tDCS']),
        ('Sham', psd_sham, colors['Sham'])
    ]):
        ax = axes[ax_idx]
        
        # collect data for this channel across subjects
        psds_baseline = []
        psds_condition = []
        freqs = None
      
        for sid in psd_data.keys():
            # skip subject if not present in both condition and baseline
            if sid not in psd_baseline:
                continue
           
            # average across trials
            psd = psd_data[sid]['psd'][:, 0, :]  # (n_epochs, n_freqs)
            psd_mean = psd.mean(axis = 0)
            psds_condition.append(psd_mean)
            
            psd_base = psd_baseline[sid]['psd'][:, 0, :]  # (n_epochs, n_freqs)
            psd_base_mean = psd_base.mean(axis = 0)
            psds_baseline.append(psd_base_mean)
            
            # get frequencies
            if freqs is None:
                freqs = psd_data[sid]['freqs']
        
        # convert to array
        psds_condition = np.array(psds_condition)
        psds_baseline = np.array(psds_baseline)  # (n_subjects, n_freqs)
        
        # compute mean and SEM across subjects
        condition_mean = psds_condition.mean(axis = 0)
        condition_sem = psds_condition.std(axis = 0) / np.sqrt(psds_condition.shape[0])
        
        baseline_mean = psds_baseline.mean(axis = 0)
        baseline_sem = psds_baseline.std(axis = 0) / np.sqrt(psds_baseline.shape[0])
        
        # convert to log scale (dB)
        condition_mean_log = 10 * np.log10(condition_mean)
        condition_sem_log = 10 * np.log10(condition_sem + condition_mean) - condition_mean_log
        
        baseline_mean_log = 10 * np.log10(baseline_mean)
        baseline_sem_log = 10 * np.log10(baseline_sem + baseline_mean) - baseline_mean_log
        
        # plot baseline
        ax.plot(freqs, baseline_mean_log, color = colors['Baseline'],
                linewidth = 1.5, label = 'Baseline', linestyle = '--')
        ax.fill_between(freqs,
                        baseline_mean_log - baseline_sem_log,
                        baseline_mean_log + baseline_sem_log,
                        color = colors['Baseline'], alpha = 0.2)
        
        # plot condition
        ax.plot(freqs, condition_mean_log, color = color, linewidth = 1.5, label = condition_name)
        ax.fill_between(freqs,
                        condition_mean_log - condition_sem_log,
                        condition_mean_log + condition_sem_log,
                        color = color, alpha = 0.2)
        
        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(f'{condition_name} vs. baseline')
        ax.legend()
    
    axes[0].set_ylabel('Power (dB)')
    plt.tight_layout()
    plt.show()

def plot_specparam(specparam_baseline, specparam_sham, specparam_tacs, specparam_tdcs):
    """
    Plot histograms and summary statistics for specparam parameters for each condition vs. baseline.
    
    Parameters:
    -----------
    specparam_baseline : dict
        Baseline specparam results from fit_specparam()
    specparam_sham, specparam_tacs, specparam_tdcs : dict
        Dictionaries from fit_specparam()
    """    
    # parameters to plot
    params = {
        'peak_cf': 'Peak frequency (Hz)',
        'peak_pw': 'Peak power',
        'peak_bw': 'Peak bandwidth (Hz)',
        'peak_snr': 'Peak SNR',
        'ap_offset': 'Aperiodic offset',
        'ap_slope': 'Aperiodic slope'
    }
    
    # colors for each condition
    colors = {
        'tACS': 'orange', 
        'tDCS': 'green',
        'Sham': 'dodgerblue',
        'Baseline': 'dimgray'
    }
    
    # collect data for each condition paired with baseline
    def collect_paired_data(specparam_results, specparam_baseline):
        data_condition = {
            'peak_cf': [],
            'peak_pw': [],
            'peak_bw': [],
            'peak_snr': [],
            'ap_offset': [],
            'ap_slope': []
        }
        
        data_baseline = {
            'peak_cf': [],
            'peak_pw': [],
            'peak_bw': [],
            'peak_snr': [],
            'ap_offset': [],
            'ap_slope': []
        }
        
        n_subjects = 0
        
        for sid in specparam_results.keys():
            # skip if subject not in baseline
            if sid not in specparam_baseline:
                continue
            
            n_subjects += 1
            
            # get condition data
            peak_params = specparam_results[sid]['peak_params']
            ap_params = specparam_results[sid]['ap_params']
            
            # get baseline data
            peak_params_base = specparam_baseline[sid]['peak_params']
            ap_params_base = specparam_baseline[sid]['ap_params']
            
            # collect parameters (excluding NaNs)
            valid_trials = ~np.isnan(peak_params[:, 0])
            data_condition['peak_cf'].extend(peak_params[valid_trials, 0])
            data_condition['peak_pw'].extend(peak_params[valid_trials, 1])
            data_condition['peak_bw'].extend(peak_params[valid_trials, 2])
            data_condition['peak_snr'].extend(peak_params[valid_trials, 3])
            data_condition['ap_offset'].extend(ap_params[0, valid_trials])
            data_condition['ap_slope'].extend(ap_params[1, valid_trials])
            
            # collect baseline parameters
            valid_trials_base = ~np.isnan(peak_params_base[:, 0])
            data_baseline['peak_cf'].extend(peak_params_base[valid_trials_base, 0])
            data_baseline['peak_pw'].extend(peak_params_base[valid_trials_base, 1])
            data_baseline['peak_bw'].extend(peak_params_base[valid_trials_base, 2])
            data_baseline['peak_snr'].extend(peak_params_base[valid_trials_base, 3])
            data_baseline['ap_offset'].extend(ap_params_base[0, valid_trials_base])
            data_baseline['ap_slope'].extend(ap_params_base[1, valid_trials_base])
        
        # convert to arrays
        for key in data_condition.keys():
            data_condition[key] = np.array(data_condition[key])
            data_baseline[key] = np.array(data_baseline[key])
        
        return data_condition, data_baseline, n_subjects
    
    # collect paired data for each condition
    data_tacs, data_baseline_tacs, n_tacs = collect_paired_data(specparam_tacs, specparam_baseline)
    data_tdcs, data_baseline_tdcs, n_tdcs = collect_paired_data(specparam_tdcs, specparam_baseline)
    data_sham, data_baseline_sham, n_sham = collect_paired_data(specparam_sham, specparam_baseline)
    
    # create figure with subplots (3 columns for conditions)
    fig = plt.figure(figsize = (9, 18))
    gs = fig.add_gridspec(6, 3, hspace = 0.35, wspace = 0.15)
    
    # plot histograms
    for row_idx, (param_key, param_label) in enumerate(params.items()):
        
        # collect all data for this parameter to determine shared x-limits
        all_data = []
        for data_condition, data_baseline in [
            (data_tacs, data_baseline_tacs),
            (data_tdcs, data_baseline_tdcs),
            (data_sham, data_baseline_sham)
        ]:
            all_data.extend(data_condition[param_key])
            all_data.extend(data_baseline[param_key])
        
        # calculate x-limits using percentiles to exclude outliers
        x_min = np.percentile(all_data, 1)
        x_max = np.percentile(all_data, 99)
        x_range = x_max - x_min
        x_margin = x_range * 0.05
        
        # create axes with shared y for this row
        axes_row = []
        for col_idx in range(3):
            if col_idx == 0:
                ax = fig.add_subplot(gs[row_idx, col_idx])
            else:
                ax = fig.add_subplot(gs[row_idx, col_idx], sharey = axes_row[0])
            axes_row.append(ax)
        
        for col_idx, (condition_name, data_condition, data_baseline, color) in enumerate([
            ('tACS', data_tacs, data_baseline_tacs, colors['tACS']),
            ('tDCS', data_tdcs, data_baseline_tdcs, colors['tDCS']),
            ('Sham', data_sham, data_baseline_sham, colors['Sham'])
        ]):
            ax = axes_row[col_idx]
            
            # plot baseline
            ax.hist(data_baseline[param_key], bins = 30, alpha = 0.3, color = colors['Baseline'], 
                   label = 'Baseline', density = True, range = (x_min - x_margin, x_max + x_margin))
            
            # plot condition
            ax.hist(data_condition[param_key], bins = 30, alpha = 0.3, color = color, 
                   label = condition_name, density = True, range = (x_min - x_margin, x_max + x_margin))
            
            # set shared x-limits
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            
            if row_idx == 0:
                ax.set_title(f'{condition_name} vs. Baseline')
                ax.legend(loc = 'upper right', bbox_to_anchor = (1.12, 1))
            
            if col_idx == 0:
                ax.set_ylabel('Density')
            else:
                ax.tick_params(labelleft = False)
            
            ax.set_xlabel(param_label)            
    
    plt.show()
    
    # print summary statistics
    print('Specparam summary statistics')
    
    for condition_name, data_condition, data_baseline, n_subj in [
        ('tACS', data_tacs, data_baseline_tacs, n_tacs),
        ('tDCS', data_tdcs, data_baseline_tdcs, n_tdcs),
        ('Sham', data_sham, data_baseline_sham, n_sham)
    ]:
        print(f'\n{condition_name} vs. Baseline (N = {n_subj} subjects):')
        
        for param_key, param_label in params.items():
            values_condition = data_condition[param_key]
            values_baseline = data_baseline[param_key]
            
            print(f'{param_label:19s}: '
                  f'Baseline median = {np.median(values_baseline):6.3f} | '
                  f'{condition_name} median = {np.median(values_condition):6.3f}')

def plot_ssep(ssep_baseline, ssep_sham, ssep_tacs, ssep_tdcs, trial_idx = 100):
    """
    Plot example SSEP traces from each condition vs. baseline.
    
    Parameters:
    -----------
    ssep_baseline : dict
        Baseline SSEP data from extract_ssep()
    ssep_sham, ssep_tacs, ssep_tdcs : dict
        Dictionaries from extract_ssep()
    trial_idx : int
        Trial index to plot (default: 100)
    """
    
    fig, axes = plt.subplots(3, 1, figsize = (9, 8), sharey = True)
    
    # colors for each condition
    colors = {
        'tACS': 'orange', 
        'tDCS': 'green',
        'Sham': 'dodgerblue',
        'Baseline': 'dimgray'
    }
    
    # plot an example trial from each condition
    for idx, (condition_name, ssep_data, color, ax) in enumerate([
        ('tACS', ssep_tacs, colors['tACS'], axes[0]),
        ('tDCS', ssep_tdcs, colors['tDCS'], axes[1]),
        ('Sham', ssep_sham, colors['Sham'], axes[2])
    ]):
        
        # get first subject with valid data at trial_idx
        found = False
        for sid in ssep_data.keys():
            # skip if subject not in baseline
            if sid not in ssep_baseline:
                continue
            
            filtered_signals = ssep_data[sid]['filtered_signals']
            filter_params = ssep_data[sid]['filter_params']
            time = ssep_data[sid]['time']
            
            # get baseline data
            filtered_signals_base = ssep_baseline[sid]['filtered_signals']
            filter_params_base = ssep_baseline[sid]['filter_params']
            
            # check if trial_idx exists and is valid for both condition and baseline
            if trial_idx <=  len(filtered_signals) and trial_idx <=  len(filtered_signals_base):
                signal = filtered_signals[trial_idx, :]
                signal_base = filtered_signals_base[trial_idx, :]
                
                if not np.isnan(signal[0]) and not np.isnan(signal_base[0]):  # check if valid
                    peak_cf = filter_params[trial_idx, 0]
                    peak_bw = filter_params[trial_idx, 1]
                    f_low = filter_params[trial_idx, 2]
                    f_high = filter_params[trial_idx, 3]
                    
                    peak_cf_base = filter_params_base[trial_idx, 0]
                    peak_bw_base = filter_params_base[trial_idx, 1]
                    
                    # convert time to ms
                    time_ms = time * 1000
                    
                    # plot baseline
                    ax.plot(time_ms, signal_base, color = colors['Baseline'], 
                           linewidth = 1.5, label = 'Baseline', linestyle = '--', alpha = 0.7)
                    
                    # plot condition
                    ax.plot(time_ms, signal, color = color, linewidth = 1.5, label = condition_name)
                    
                    ax.axhline(0, color = 'k', linewidth = 0.5, alpha = 0.5)
                    ax.set_xlim(time_ms[0], time_ms[-1])
                    ax.set_ylabel('Amplitude (μV)')
                    ax.set_title(f'{condition_name} (peak {peak_cf:.1f} Hz, bandwidth {peak_bw:.1f} Hz) vs. '
                                f'baseline (peak {peak_cf_base:.1f} Hz, bandwidth {peak_bw_base:.1f} Hz)')
                    ax.legend(loc = 'upper right')
                    
                    found = True
                    break
        
        if not found:
            ax.text(0.5, 0.5, f'No valid data for trial {trial_idx}', 
                   ha = 'center', va = 'center', transform = ax.transAxes)
    
    # common x-label
    axes[-1].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.show()

def plot_ssep_temporal(temporal_baseline, temporal_sham, temporal_tacs, temporal_tdcs):
    """
    Plot histograms and summary statistics for SSEP temporal features for each condition vs. baseline.
    
    Parameters:
    -----------
    temporal_baseline : dict
        Baseline temporal results from analyze_ssep()
    temporal_sham, temporal_tacs, temporal_tdcs : dict
        Dictionaries from analyze_ssep()
    """    
    # parameters to plot
    params = {
        'response_latency': 'Response latency (ms)',
        'amplitude_slope': 'Amplitude slope (μV/s)',
        'autocorr_1cycle': 'Autocorrelation (1 cycle)'
    }
    
    # colors for each condition
    colors = {
        'tACS': 'orange', 
        'tDCS': 'green',
        'Sham': 'dodgerblue',
        'Baseline': 'dimgray'
    }
    
    # collect data across subjects
    def collect_paired_data(temporal_results, temporal_baseline):
        data_condition = {
            'response_latency': [],
            'amplitude_slope': [],
            'autocorr_1cycle': []
        }
        
        data_baseline = {
            'response_latency': [],
            'amplitude_slope': [],
            'autocorr_1cycle': []
        }
        
        n_subjects = 0
        
        for sid in temporal_results.keys():
            # skip if subject not in baseline
            if sid not in temporal_baseline:
                continue
            
            n_subjects +=  1
            
            # get condition data
            response_latency = temporal_results[sid]['response_latency']
            amplitude_slope = temporal_results[sid]['amplitude_slope']
            autocorr_1cycle = temporal_results[sid]['autocorr_1cycle']
            
            # get baseline data
            response_latency_base = temporal_baseline[sid]['response_latency']
            amplitude_slope_base = temporal_baseline[sid]['amplitude_slope']
            autocorr_1cycle_base = temporal_baseline[sid]['autocorr_1cycle']
            
            # collect parameters (excluding NaNs)
            data_condition['response_latency'].extend(response_latency[~np.isnan(response_latency)])
            data_condition['amplitude_slope'].extend(amplitude_slope[~np.isnan(amplitude_slope)])
            data_condition['autocorr_1cycle'].extend(autocorr_1cycle[~np.isnan(autocorr_1cycle)])
            
            data_baseline['response_latency'].extend(response_latency_base[~np.isnan(response_latency_base)])
            data_baseline['amplitude_slope'].extend(amplitude_slope_base[~np.isnan(amplitude_slope_base)])
            data_baseline['autocorr_1cycle'].extend(autocorr_1cycle_base[~np.isnan(autocorr_1cycle_base)])
        
        # convert to arrays
        for key in data_condition.keys():
            data_condition[key] = np.array(data_condition[key])
            data_baseline[key] = np.array(data_baseline[key])
        
        return data_condition, data_baseline, n_subjects
    
    # collect paired data for each condition
    data_tacs, data_baseline_tacs, n_tacs = collect_paired_data(temporal_tacs, temporal_baseline)
    data_tdcs, data_baseline_tdcs, n_tdcs = collect_paired_data(temporal_tdcs, temporal_baseline)
    data_sham, data_baseline_sham, n_sham = collect_paired_data(temporal_sham, temporal_baseline)
    
    # create figure with subplots (3 columns for conditions)
    fig = plt.figure(figsize = (9, 9))
    gs = fig.add_gridspec(3, 3, hspace = 0.35, wspace = 0.15)
    
    # plot histograms
    for row_idx, (param_key, param_label) in enumerate(params.items()):
        
        # collect all data for this parameter to determine shared x-limits
        all_data = []
        for data_condition, data_baseline in [
            (data_tacs, data_baseline_tacs),
            (data_tdcs, data_baseline_tdcs),
            (data_sham, data_baseline_sham)
        ]:
            all_data.extend(data_condition[param_key])
            all_data.extend(data_baseline[param_key])
        
        # calculate x-limits using percentiles to exclude outliers
        x_min = np.percentile(all_data, 1)
        x_max = np.percentile(all_data, 99)
        x_range = x_max - x_min
        x_margin = x_range * 0.05
        
        # create axes with shared y for this row
        axes_row = []
        for col_idx in range(3):
            if col_idx ==  0:
                ax = fig.add_subplot(gs[row_idx, col_idx])
            else:
                ax = fig.add_subplot(gs[row_idx, col_idx], sharey = axes_row[0])
            axes_row.append(ax)
        
        for col_idx, (condition_name, data_condition, data_baseline, color) in enumerate([
            ('tACS', data_tacs, data_baseline_tacs, colors['tACS']),
            ('tDCS', data_tdcs, data_baseline_tdcs, colors['tDCS']),
            ('Sham', data_sham, data_baseline_sham, colors['Sham'])
        ]):
            ax = axes_row[col_idx]
            
            # plot baseline
            ax.hist(data_baseline[param_key], bins = 30, alpha = 0.3, color = colors['Baseline'], 
                   label = 'Baseline', density = True, range = (x_min - x_margin, x_max + x_margin))
            
            # plot condition
            ax.hist(data_condition[param_key], bins = 30, alpha = 0.3, color = color, 
                   label = condition_name, density = True, range = (x_min - x_margin, x_max + x_margin))
            
            # set shared x-limits
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            
            if row_idx ==  0:
                ax.set_title(f'{condition_name} vs. Baseline')
                ax.legend()
            
            if col_idx ==  0:
                ax.set_ylabel('Density')
            else:
                ax.tick_params(labelleft = False)
            
            ax.set_xlabel(param_label)
    
    plt.show()
    
    # print summary statistics
    print('SSEP temporal features summary statistics')
    
    for condition_name, data_condition, data_baseline, n_subj in [
        ('tACS', data_tacs, data_baseline_tacs, n_tacs),
        ('tDCS', data_tdcs, data_baseline_tdcs, n_tdcs),
        ('Sham', data_sham, data_baseline_sham, n_sham)
    ]:
        print(f'\n{condition_name} vs. Baseline (N = {n_subj} subjects):')
        
        for param_key, param_label in params.items():
            values_condition = data_condition[param_key]
            values_baseline = data_baseline[param_key]
            
            print(f'{param_label:25s}: '
                  f'Baseline median = {np.median(values_baseline):6.3f} | '
                  f'{condition_name} median = {np.median(values_condition):6.3f}')

def plot_bycycle(bycycle_baseline, bycycle_sham, bycycle_tacs, bycycle_tdcs):
    """
    Plot histograms and summary statistics for bycycle features for each condition vs. baseline.
    
    Parameters:
    -----------
    bycycle_baseline : dict
        Baseline bycycle results from analyze_bycycle()
    bycycle_sham, bycycle_tacs, bycycle_tdcs : dict
        Dictionaries from analyze_bycycle()
    """    
    # parameters to plot
    params = {
        'amplitude': 'Amplitude',
        'period': 'Period (ms)',
        'rise_time': 'Rise time (ms)',
        'decay_time': 'Decay time (ms)',
        'rise_decay_symmetry': 'Rise-decay symmetry',
        'peak_voltage': 'Peak voltage',
        'trough_voltage': 'Trough voltage',
        'burst_rate': 'Burst rate'
    }
        
    # colors for each condition
    colors = {
        'tACS': 'orange', 
        'tDCS': 'green',
        'Sham': 'dodgerblue',
        'Baseline': 'dimgray'
    }
    
    # collect data across subjects
    def collect_paired_data(bycycle_results, bycycle_baseline):
        data_condition = {key: [] for key in params.keys()}
        data_baseline = {key: [] for key in params.keys()}
        
        n_subjects = 0
        
        for sid in bycycle_results.keys():
            # skip if subject not in baseline
            if sid not in bycycle_baseline:
                continue
            
            n_subjects +=  1
            
            # collect parameters (excluding NaNs)
            for key in params.keys():
                values = np.array(bycycle_results[sid][key])
                values_base = np.array(bycycle_baseline[sid][key])
                
                data_condition[key].extend(values[~np.isnan(values)])
                data_baseline[key].extend(values_base[~np.isnan(values_base)])
        
        # convert to arrays
        for key in data_condition.keys():
            data_condition[key] = np.array(data_condition[key])
            data_baseline[key] = np.array(data_baseline[key])
        
        return data_condition, data_baseline, n_subjects
    
    # collect paired data for each condition
    data_tacs, data_baseline_tacs, n_tacs = collect_paired_data(bycycle_tacs, bycycle_baseline)
    data_tdcs, data_baseline_tdcs, n_tdcs = collect_paired_data(bycycle_tdcs, bycycle_baseline)
    data_sham, data_baseline_sham, n_sham = collect_paired_data(bycycle_sham, bycycle_baseline)
    
    # create figure with subplots (3 columns for conditions)
    fig = plt.figure(figsize = (9, 24))
    gs = fig.add_gridspec(8, 3, hspace = 0.35, wspace = 0.15)
    
    # plot histograms
    for row_idx, (param_key, param_label) in enumerate(params.items()):
        
        # collect all data for this parameter to determine shared x-limits
        all_data = []
        for data_condition, data_baseline in [
            (data_tacs, data_baseline_tacs),
            (data_tdcs, data_baseline_tdcs),
            (data_sham, data_baseline_sham)
        ]:
            all_data.extend(data_condition[param_key])
            all_data.extend(data_baseline[param_key])
        
        # calculate x-limits using percentiles to exclude outliers
        x_min = np.percentile(all_data, 1)
        x_max = np.percentile(all_data, 99)
        x_range = x_max - x_min
        x_margin = x_range * 0.05
        
        # create axes with shared y for this row
        axes_row = []
        for col_idx in range(3):
            if col_idx ==  0:
                ax = fig.add_subplot(gs[row_idx, col_idx])
            else:
                ax = fig.add_subplot(gs[row_idx, col_idx], sharey = axes_row[0])
            axes_row.append(ax)
        
        for col_idx, (condition_name, data_condition, data_baseline, color) in enumerate([
            ('tACS', data_tacs, data_baseline_tacs, colors['tACS']),
            ('tDCS', data_tdcs, data_baseline_tdcs, colors['tDCS']),
            ('Sham', data_sham, data_baseline_sham, colors['Sham'])
        ]):
            ax = axes_row[col_idx]
            
            # plot baseline
            ax.hist(data_baseline[param_key], bins = 30, alpha = 0.3, color = colors['Baseline'], 
                   label = 'Baseline', density = True, range = (x_min - x_margin, x_max + x_margin))
            
            # plot condition
            ax.hist(data_condition[param_key], bins = 30, alpha = 0.3, color = color, 
                   label = condition_name, density = True, range = (x_min - x_margin, x_max + x_margin))
            
            # set shared x-limits
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            
            if row_idx ==  0:
                ax.set_title(f'{condition_name} vs. Baseline')
                ax.legend()
            
            if col_idx ==  0:
                ax.set_ylabel('Density')
            else:
                ax.tick_params(labelleft = False)
            
            ax.set_xlabel(param_label)
    
    plt.show()
    
    # print summary statistics
    print('Bycycle features summary statistics')
    
    for condition_name, data_condition, data_baseline, n_subj in [
        ('tACS', data_tacs, data_baseline_tacs, n_tacs),
        ('tDCS', data_tdcs, data_baseline_tdcs, n_tdcs),
        ('Sham', data_sham, data_baseline_sham, n_sham)
    ]:
        print(f'\n{condition_name} vs. Baseline (N = {n_subj} subjects):')
        
        for param_key, param_label in params.items():
            values_condition = data_condition[param_key]
            values_baseline = data_baseline[param_key]
            
            print(f'{param_label:19s}: '
                  f'Baseline median = {np.median(values_baseline):6.3f} | '
                  f'{condition_name} median = {np.median(values_condition):6.3f}')
