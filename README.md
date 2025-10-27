## Modulation of auditory gamma-band responses using transcranial electrical stimulation: Part 2 (2025)
### Follow up to [Modulation of auditory gamma-band responses using transcranial electrical stimulation](https://github.com/elizljohnson-projects/project-eeg-gamma-modulation.git)

Noninvasive brain stimulation is used across clinical, commercial, and research applications to change the brain in hopes of changing behavior, yet we don't fully understand how it changes brain activity. Using scalp EEG, we demonstrated that gamma frequency-tuned transcranial alternating current stimulation (tACS), but not transcranial direct current stimulation (tDCS), enhances auditory gamma responses in healthy adults ([Jones et al., 2020](https://doi.org/10.1152/jn.00003.2020)).

Our research aims to uncover mechanistic explanations of the neural basis of human behavior, that is, move from where to how. Our goals are multifaceted: (1) advance fundamental science by discovering new knowledge using rigorous, reproducible methods; and (2) advance translational applications in neurotechnology, precision medicine, and product development that are grounded in rigorous science. This project is a re-analysis of Jones et al., 2020, aiming to classify stimulation conditions from single-trial gamma-band steady-state evoked potentials (SSEPs):

Goal 1: Feature extraction and engineering
- Characterize gamma-band SSEPs using spectral decomposition, time-domain analysis, and cycle-by-cycle waveform analysis
- Extract 14 features spanning frequency-domain, temporal, and waveform morphology characteristics
- Engineer baseline-relative features to control for baseline individual differences

Goal 2: Machine learning classification
- Compare classification algorithms
- Optimize hyperparameters using 2-stage randomized search
- Identify optimal features through importance ranking
- Validate model performance with permutation testing (shuffled subject-condition mappings)

Results demonstrate that machine learning with baseline-relative features can classify stimulation conditions (tACS, tDCS, Sham) at the single-trial level using data from a single EEG channel (Cz), achieving above-chance accuracy.

Publications or other papers using these scripts and/or data should cite the original publication:
- Jones, KT, Johnson, EL, Tauxe, ZS, Rojas, DC. Modulation of auditory gamma-band responses using transcranial electrical stimulation. _Journal of Neurophysiology_ 123 (2020). [DOI](https://doi.org/10.1152/jn.00003.2020)

Software:
- Python 3.12.12
- Environment: Google Colab
- Package versions:
  - NumPy 2.0.2
  - Pandas 2.2.2
  - SciPy 1.16.2
  - MNE 1.10.2
  - Matplotlib 3.10.0
  - Scikit-learn 1.6.1
  - Specparam 2.0.0rc3
  - Bycycle 1.2.0

Notes:
- Run using the notebook `gamma_mod_ml.ipynb` with `gamma_utils.py`, `plotting_utils.py`, and `ml_utils.py` uploaded to the Colab File panel.
- This notebook may take several hours to run due to hyperparameter tuning and permutation testing.
