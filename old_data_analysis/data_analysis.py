"""
Script for exploratory data analysis (EDA) of the entire participant dataset.

Steps:

1. Identifies all participants (excluding "corrupted" and "old" data).
2. Builds a global, consistent event map (master_event_id).
3. In a loop, processes each participant's data:
    - Loads 'HONEST' and 'DECEITFUL' files.
    - Applies preprocessing (from preprocessing.py).
    - Creates epochs (from epoching.py) using the master_event_id map.
    - Splits epochs into two global lists: all_honest_epochs and all_deceitful_epochs.
4. Combines and averages epochs from both groups, creating Evoked (ERP) objects.
5. Generates and displays two plots:
    -ERP comparison (Global Field Power) for 'Honest' vs. 'Deceitful'.
    - Topographic maps showing differences in brain activity over time.
"""

import mne
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
from src.preprocessing import preprocessing
from src.epoching import create_epochs

warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More events than default colors available")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")

DATA_FOLDER = "../dataset"

participant_dirs = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))
                    if "popsute" not in d.lower() and "stare" not in d.lower()]
print(f"Found {len(participant_dirs)} participant folders.")

all_descriptions = set()

for participant_uuid in participant_dirs:
    participant_path = os.path.join(DATA_FOLDER, participant_uuid)
    files = glob.glob(os.path.join(participant_path, "*_raw.fif"))

    for f in files:
        try:
            raw = mne.io.read_raw_fif(f, preload=False, verbose=False)
            descriptions = np.unique(raw.annotations.description)
            all_descriptions.update(descriptions)
        except Exception as e:
            print(f"Skipping file {os.path.basename(f)}: {e}")

master_event_id = {desc: i+1 for i, desc in enumerate(sorted(list(all_descriptions)))}

all_honest_epochs = []
all_deceitful_epochs = []

for participant_uuid in participant_dirs:
    participant_path = os.path.join(DATA_FOLDER, participant_uuid)
    print(f"\n------------------------Processing participant: {participant_uuid}------------------------")

    honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
    deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

    try:
        for h_file in honest_files:
            raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
            prep_honest = preprocessing(raw_honest)
            epochs_honest = create_epochs(prep_honest, master_event_id=master_event_id)

            if epochs_honest and len(epochs_honest) > 0:
                all_honest_epochs.append(epochs_honest)

        for d_file in deceitful_files:
            raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
            prep_deceitful = preprocessing(raw_deceitful)
            epochs_deceitful = create_epochs(prep_deceitful, master_event_id=master_event_id)
            if epochs_deceitful and len(epochs_deceitful) > 0:
                all_deceitful_epochs.append(epochs_deceitful)

    except Exception as e:
        print(f"Error processing {participant_uuid}: {e}")


if all_honest_epochs and all_deceitful_epochs:
    combined_honest = mne.concatenate_epochs(all_honest_epochs)
    combined_deceitful = mne.concatenate_epochs(all_deceitful_epochs)

    print(f"Total honest epochs collected: {len(combined_honest)}")
    print(f"Total deceitful epochs collected: {len(combined_deceitful)}")

    evoked_honest = combined_honest.average()
    evoked_deceitful = combined_deceitful.average()

    evokeds = {"Honest": evoked_honest, "Deceitful": evoked_deceitful}

    mne.viz.plot_compare_evokeds(evokeds,
                                 picks='eeg',
                                 legend='upper left',
                                 title="ERP: Honest vs Deceitful all participants")

    diff_erp = mne.combine_evoked([evoked_honest, evoked_deceitful], weights=[1, -1])

    topo_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    diff_erp.plot_topomap(times=topo_times)
    plt.show()

else:
    print("No epochs were loaded")