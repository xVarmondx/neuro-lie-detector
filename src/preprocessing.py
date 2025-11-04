import mne
import os
import matplotlib.pyplot as plt


def preprocessing(raw_data):
    """
    Cleans a raw MNE Raw object.

    Selects only 'eeg' type channels.
    Applies a 1-40 Hz band-pass filter.

    Args:
        raw_data (mne.io.Raw): The raw MNE Raw object to be processed.

    Returns:
        mne.io.Raw: The processed (cleaned) MNE Raw object.
    """

    raw_copy = raw_data.copy()
    raw_copy.pick_types(eeg=True)
    #print(f"Number of channels after picking EEG types: {len(raw_copy.ch_names)}")

    raw_copy.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw_copy.set_montage(montage, on_missing='warn')
    return raw_copy


# Sandbox for testing only this file
if __name__ == '__main__':

    print("--- Running preprocessing.py in test mode ---")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FOLDER = os.path.join(BASE_DIR, "dataset")

    EXAMPLE_USER = "2D663E30"
    EXAMPLE_EEG_FILENAME = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
    EXAMPLE_EEG_PATH = os.path.join(DATA_FOLDER, EXAMPLE_USER, EXAMPLE_EEG_FILENAME)

    if not os.path.exists(EXAMPLE_EEG_PATH):
        print(f"Test file not found at: {EXAMPLE_EEG_PATH}")
        print("Please check the path. The script expects to be in an 'src' folder next to 'dataset'.")
    else:
        raw = mne.io.read_raw_fif(EXAMPLE_EEG_PATH, preload=True, verbose=False)
        raw.apply_function(lambda x: x*1e-6)

        print(f"\nOriginal channel count: {len(raw.ch_names)}")

        print("\n--- Running preprocessing function ---")
        preprocessed_raw = preprocessing(raw)

        print("\n--- Preprocessing complete ---")
        print(f"Processed channel count: {len(preprocessed_raw.ch_names)}")

        print("Displaying test plot...")
        preprocessed_raw.plot(title="Preprocessed Data (Test Run)")
        plt.show()