import mne
import os
import matplotlib.pyplot as plt


def preprocessing(raw_data):

    # Cleans the raw MNE data object.

    raw_copy = raw_data.copy()
    raw_copy.pick_types(eeg=True)
    print(f"Number of channels after picking EEG types: {len(raw_copy.ch_names)}")

    raw_copy.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)
    print("Applied a 1-40 Hz band-pass filter")

    montage = mne.channels.make_standard_montage('standard_1020')
    raw_copy.set_montage(montage, on_missing='warn')
    print("Set standard_1020 montage.")
    return raw_copy


DATA_FOLDER = "dataset"
EXAMPLE_USER = "2D663E30"
EXAMPLE_EEG_FILENAME = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
EXAMPLE_EEG_PATH = os.path.join(DATA_FOLDER, EXAMPLE_USER, EXAMPLE_EEG_FILENAME)

raw = mne.io.read_raw_fif(EXAMPLE_EEG_PATH, preload=True, verbose=False)

print("\n------------------------Data BEFORE preprocessing------------------------")
print(f"Number of channels: {len(raw.ch_names)}")

print("\n------------------------Running preprocessing------------------------")
preprocessed_raw = preprocessing(raw)

print("\n------------------------Data AFTER preprocessing------------------------")
print(f"Number of channels: {len(preprocessed_raw.ch_names)}")

preprocessed_raw.plot(title="Preprocessed Data")

plt.show()