import mne
import numpy as np
from src.preprocessing import preprocessing
from src.epoching import create_epochs
import os


def extract_psd_features(epochs_data):
    """
    Extracts Power Spectral Density (PSD) features from an MNE Epochs object.

    This function calculates the average power in predefined frequency bands
    (Theta, Alpha, Beta) for each EEG channel and each epoch.

    Args:
        epochs_data (mne.Epochs): An MNE Epochs object.

    Returns:
        tuple (np.ndarray, np.ndarray): A tuple containing:
            - features (np.ndarray): The feature matrix, shape (n_epochs, n_channels * n_bands).
            - labels (np.ndarray): The labels vector, shape (n_epochs,).
    """

    freq_bands = {
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30],
        "gamma": [30, 40]
    }

    spectrum = epochs_data.compute_psd(method='welch', fmin=4.0, fmax=40.0, picks='eeg', verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)

    features_list = []
    for epoch_power_data in psds:
        epoch_features = []
        for channel_power_data in epoch_power_data:
            for band, f_range in freq_bands.items():
                band_indices = np.where((freqs >= f_range[0]) & (freqs < f_range[1]))[0]

                if len(band_indices) == 0:
                    avg_power = 0.0
                else:
                    avg_power = np.mean(channel_power_data[band_indices])
                epoch_features.append(avg_power)
        features_list.append(epoch_features)

    features = np.array(features_list)
    labels = epochs_data.events[:, -1]

    #print(f"PSD Feature matrix X created with shape: {features.shape}")

    return features, labels


# Sandbox for testing only this file
if __name__ == '__main__':
    DATA_FOLDER = "../dataset"
    EXAMPLE_USER = "2D663E30"
    EXAMPLE_EEG_FILENAME = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
    EXAMPLE_EEG_PATH = os.path.join(DATA_FOLDER, EXAMPLE_USER, EXAMPLE_EEG_FILENAME)

    raw = mne.io.read_raw_fif(EXAMPLE_EEG_PATH, preload=True, verbose=False)
    raw.apply_function(lambda x: x * 1e-6)
    preprocessed_raw = preprocessing(raw)
    epochs = create_epochs(preprocessed_raw)

    if epochs:
        print("\n------------------------Running Feature Extraction------------------------")
        X, y = extract_psd_features(epochs)

        print("\n------------------------Feature Extraction Results------------------------")
        print("Shape of our feature matrix X:", X.shape)
        print("Example of the first row of X (first 10 features):", X[0, :10])
        print("\nShape of our labels vector y:", y.shape)
        print("Unique labels found in y:", np.unique(y))
        print("Example of the first 5 labels:", y[:5])