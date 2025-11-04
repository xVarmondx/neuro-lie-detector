import mne
import os
import matplotlib.pyplot as plt
from src.preprocessing import preprocessing
import warnings


def create_epochs(preprocessed_raw_data, tmin=-0.2, tmax=0.8, master_event_id=None):
    """
    cuts the continuous preprocessed EEG data into epochs around stimulus events.

    Args:
        preprocessed_raw_data (mne.io.Raw): The cleaned MNE Raw object.
        tmin (float): Start time of the epoch before the event (in seconds).
        tmax (float): End time of the epoch after the event (in seconds).
        master_event_id (dict | None): A predefined event_id map to ensure
            consistency across files. If None, one will be generated.

    Returns:
        mne.Epochs | None: The created MNE Epochs object, or None if no
        stimulus events are found.
    """
    try:
        events, event_id = mne.events_from_annotations(preprocessed_raw_data, event_id=master_event_id, verbose=False)
    except ValueError:
        warnings.warn(f"Could not find any of the events specified in master_event_id for file.")
        return None

    stimulus_event_ids = {key: value for key, value in event_id.items() if 'PersonalDataField' in key}

    if not stimulus_event_ids:
        warnings.warn("No 'PersonalDataField' stimulus events found in this file.")
        return None

    epochs = mne.Epochs(
        preprocessed_raw_data,
        events=events,
        event_id=stimulus_event_ids,
        tmin=tmin,
        tmax=tmax,
        preload=True,
        baseline=(-0.2, 0),
        reject=None,
        verbose=False
    )

    return epochs


# Sandbox for testing only this file
if __name__ == '__main__':

    print("--- Running epoching.py in test mode ---")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FOLDER = os.path.join(BASE_DIR, "dataset")

    EXAMPLE_USER = "2D663E30"
    EXAMPLE_EEG_FILENAME = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
    EXAMPLE_EEG_PATH = os.path.join(DATA_FOLDER, EXAMPLE_USER, EXAMPLE_EEG_FILENAME)

    if not os.path.exists(EXAMPLE_EEG_PATH):
        print(f"Test file not found at: {EXAMPLE_EEG_PATH}")
    else:
        raw = mne.io.read_raw_fif(EXAMPLE_EEG_PATH, preload=True, verbose=False)
        raw.apply_function(lambda x: x * 1e-6)

        print("Running preprocessing...")
        preprocessed_raw = preprocessing(raw)

        print("Generating a test master_event_id map...")
        try:
            _, test_master_id = mne.events_from_annotations(preprocessed_raw, verbose=False)
            print("Test map created successfully.")
        except Exception as e:
            test_master_id = None
            print(f"Could not create test map: {e}")

        if test_master_id:
            print("Running create_epochs...")
            test_epochs = create_epochs(preprocessed_raw, master_event_id=test_master_id)

            if test_epochs:
                print(f"Successfully created {len(test_epochs)} epochs.")
                print("\n------------------------Epochs Information------------------------")
                print(test_epochs)

                print("Displaying test plot...")
                test_epochs.plot(n_epochs=10, n_channels=16, title="Created Epochs (Test Run)", show=False)
                plt.show()
            else:
                print("create_epochs function returned None.")
        else:
            print("Skipping epoch creation, test map generation failed.")