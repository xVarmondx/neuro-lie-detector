import mne
import os
import matplotlib.pyplot as plt
from preprocessing import preprocessing

# Cuts the continuous preprocessed EEG data into epochs around stimulus events
def create_epochs(preprocessed_raw_data, tmin=-0.2, tmax=0.8):

    # Find events from annotations selecting only stimulus-related ones
    events, event_id = mne.events_from_annotations(preprocessed_raw_data, verbose=False)
    stimulus_event_ids = {key: value for key, value in event_id.items() if 'PersonalDataField' in key}

    if not stimulus_event_ids:
        print("No stimulus events found")
        return None

    #print(f"Found {len(events)} total events")
    #print(f"Using {len(stimulus_event_ids)} stimulus types for epoching")

    # Create the epochs object
    epochs = mne.Epochs(
        preprocessed_raw_data,
        events=events,
        event_id=stimulus_event_ids,
        tmin=tmin,
        tmax=tmax,
        preload=True,
        baseline=(-0.2, 0),  # The time before the stimulus (-0.2s to 0s) is used for baseline correction
        reject=None,
        verbose=False
    )

    print(f"Created {len(epochs)} epochs from the data")
    return epochs

DATA_FOLDER = "dataset"
EXAMPLE_USER = "2D663E30"
EXAMPLE_EEG_FILENAME = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
EXAMPLE_EEG_PATH = os.path.join(DATA_FOLDER, EXAMPLE_USER, EXAMPLE_EEG_FILENAME)

raw = mne.io.read_raw_fif(EXAMPLE_EEG_PATH, preload=True, verbose=False)
raw.apply_function(lambda x: x*1e-6)

# Clean raw data
preprocessed_raw = preprocessing(raw)

# Create epochs
epochs = create_epochs(preprocessed_raw)

if epochs:
    print("\n------------------------Epochs Information------------------------")
    print(epochs)

    epochs.plot(n_epochs=10, n_channels=16, title="Created Epochs", show=False)
    plt.show()