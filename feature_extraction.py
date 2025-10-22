import mne
import numpy as np
from preprocessing import preprocessing
from epoching import create_epochs
import os


def extract_psd_features(epochs_data):

    # For each epoch it calculates the average power in predefined frequency bands (Theta, Alpha, Beta) for each EEG channel.

    FREQ_BANDS = {
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30]
    }

    spectrum = epochs_data.compute_psd(method='welch', fmin=4.0, fmax=30.0, picks='eeg', verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)

    X = []
    for epoch_psds in psds:
        epoch_features = []
        for ch_psds in epoch_psds:
            for band, f_range in FREQ_BANDS.items():
                band_indices = np.where((freqs >= f_range[0]) & (freqs < f_range[1]))[0]
                avg_power = np.mean(ch_psds[band_indices])
                epoch_features.append(avg_power)
        X.append(epoch_features)

    X = np.array(X)
    y = epochs_data.events[:, -1]

    print(f"PSD Feature matrix X created with shape: {X.shape}")

    return X, y


# Test for one file
DATA_FOLDER = "dataset"
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