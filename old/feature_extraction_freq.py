import mne
import numpy as np


def extract_psd_features(epochs_data):
    """
    Extracts features based on Power Spectral Density (PSD) from an MNE Epochs object.

    For each epoch, it calculates the average power in predefined frequency bands
    (Theta, Alpha, Beta) for each EEG channel.

    Args:
        epochs_data (mne.Epochs): The epoched data.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The feature matrix (X), with shape (n_epochs, n_channels * n_bands).
            - np.ndarray: The labels vector (y).
    """
    # 1. Define frequency bands of interest
    FREQ_BANDS = {
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30]
    }

    # 2. Calculate PSD for each epoch
    # This computes the power spectral density using the Welch method.
    spectrum = epochs_data.compute_psd(method='welch', fmin=4., fmax=30., picks='eeg', verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # psds shape is (n_epochs, n_channels, n_freqs)

    # 3. Calculate average power in each band for each channel
    X = []
    for epoch_psds in psds:
        epoch_features = []
        for ch_psds in epoch_psds:
            for band, f_range in FREQ_BANDS.items():
                # Find the indices of the frequencies that fall within the current band
                band_indices = np.where((freqs >= f_range[0]) & (freqs < f_range[1]))[0]
                # Calculate the average power for this band
                avg_power = np.mean(ch_psds[band_indices])
                epoch_features.append(avg_power)
        X.append(epoch_features)

    X = np.array(X)
    y = epochs_data.events[:, -1]

    print(f"PSD Feature matrix X created with shape: {X.shape}")  # Np. (107, 48)

    return X, y