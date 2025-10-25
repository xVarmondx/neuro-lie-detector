import sys

import mne
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import warnings
from preprocessing import preprocessing
from epoching import create_epochs

warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More events than default colors available")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")

DATA_FOLDER = "dataset"

# We findind all subdirectories in the dataset folder. Subdirectories is the particinants
participant_dirs = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]
print(f"Found {len(participant_dirs)} participant folders.")

# Building event_id map. We need this map so that event IDs like PersonalDataField are consistent across all files.
all_descriptions = set() # Using a set automatically handles duplicates

# Loop through each participant folder
for participant_uuid in participant_dirs:
    participant_path = os.path.join(DATA_FOLDER, participant_uuid)

    # Find all raw .fif files for this participant
    files = glob.glob(os.path.join(participant_path, "*_raw.fif"))

    # Loop through the files found
    for f in files:
        try:
            raw = mne.io.read_raw_fif(f, preload=False, verbose=False)

            # Get unique annotation descriptions from this file
            descriptions = np.unique(raw.annotations.description)
            #print("descriptionnnnnn",descriptions)

            # Add these descriptions to our master set
            all_descriptions.update(descriptions)
        except Exception as e:
            print(f"Skipping file {os.path.basename(f)}: {e}")

# Create the final map: {description_string: integer_id}
master_event_id = {desc: i+1 for i, desc in enumerate(sorted(list(all_descriptions)))}
#print("maaaaster",master_event_id)

#print(f"Master map created with {len(master_event_id)} unique event types.")

# Create lists to store epochs for each condition
all_honest_epochs = []
all_deceitful_epochs = []

# Loop through each participant folder
for participant_uuid in participant_dirs:
    participant_path = os.path.join(DATA_FOLDER, participant_uuid)
    print(f"\n------------------------Processing participant: {participant_uuid}------------------------")

    honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
    deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

    try:
        # Processing honest files
        for h_file in honest_files:
            raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
            prep_honest = preprocessing(raw_honest) # we applying preprocessing like filtering
            epochs_honest = create_epochs(prep_honest, master_event_id=master_event_id)

            if epochs_honest and len(epochs_honest) > 0:
                all_honest_epochs.append(epochs_honest)

        # Processing deceitful files
        for d_file in deceitful_files:
            raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
            prep_deceitful = preprocessing(raw_deceitful)
            epochs_deceitful = create_epochs(prep_deceitful, master_event_id=master_event_id)
            if epochs_deceitful and len(epochs_deceitful) > 0:
                all_deceitful_epochs.append(epochs_deceitful)

    except Exception as e:
        print(f"Error processing {participant_uuid}: {e}")


# Check if we actually loaded any data
if all_honest_epochs and all_deceitful_epochs:
    # Combine all the small Epochs objects into two big ones
    combined_honest = mne.concatenate_epochs(all_honest_epochs)
    combined_deceitful = mne.concatenate_epochs(all_deceitful_epochs)

    print(f"Total honest epochs collected: {len(combined_honest)}")
    print(f"Total deceitful epochs collected: {len(combined_deceitful)}")

    # Calculate the average brain wave across all epochs for each condition
    evoked_honest = combined_honest.average()
    evoked_deceitful = combined_deceitful.average()

    #print("hoooneest", evoked_honest)

    # Dictionary to hold the average ERPs
    evokeds = {"Honest": evoked_honest, "Deceitful": evoked_deceitful}

    # Plot 1 - compare ERP waveforms
    mne.viz.plot_compare_evokeds(evokeds,
                                 picks='eeg', # we are using only EEG channels
                                 legend='upper left',
                                 title="ERP: Honest vs Deceitful all participants")

    # Plot 2 - Calculate and plot the difference topomaps

    # Calculate the difference wave: Honest minus Deceitful
    diff_erp = mne.combine_evoked([evoked_honest, evoked_deceitful], weights=[1, -1])

    # Define specific time points to show the scalp map for in seconds
    topo_times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # Generate the topomap plot
    diff_erp.plot_topomap(times=topo_times)
    plt.show()

else:
    print("No epochs were loaded")