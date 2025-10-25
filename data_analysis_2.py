import pandas as pd
import mne
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import warnings
from preprocessing import preprocessing
from epoching import create_epochs

warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More events than default colors available")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")

DATA_FOLDER = "dataset"
SURVEY_FILE = os.path.join(DATA_FOLDER, "Ankiety.xlsx")
UUID_COLUMN = 'UUID'
GENDER_COLUMN = 'Płeć'

# Find all participant directories
participant_dirs = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]
print(f"Found {len(participant_dirs)} participant folders.")

# Building event_id map. We need this map so that event IDs like PersonalDataField are consistent across all files.
all_descriptions = set() # Set to store unique event descriptions

# # Loop through each participant folder - collecting event descriptions
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

            # Add these descriptions to our master set
            all_descriptions.update(descriptions)
        except Exception as e:
            print(f"Skipping file {os.path.basename(f)}: {e}")

# Create the final map {description: id} AFTER the collection loop
# Sorting ensures consistency across runs
master_event_id = {desc: i+1 for i, desc in enumerate(sorted(list(all_descriptions)))}
#print(f"Master map created with {len(master_event_id)} unique events.")

try:
    df_survey = pd.read_excel(SURVEY_FILE)
    # Converting UUID and gender to lowercase strings for easier comparison

    df_survey[UUID_COLUMN] = df_survey[UUID_COLUMN].astype(str).str.lower()
    df_survey[GENDER_COLUMN] = df_survey[GENDER_COLUMN].astype(str).str.lower()

except Exception as e:
    print(f"ERROR: Could not load or process {SURVEY_FILE}. {e}")
    raise e

# Create lists of UUIDs for males and females
try:
    male_uuids = df_survey[df_survey[GENDER_COLUMN] == 'm'][UUID_COLUMN].tolist()
    female_uuids = df_survey[df_survey[GENDER_COLUMN] == 'k'][UUID_COLUMN].tolist()
    print(f"Found {len(male_uuids)} male and {len(female_uuids)} female participants in survey")
except KeyError:
    print(f"ERROR: Column '{GENDER_COLUMN}' or '{UUID_COLUMN}' not foundin survey ")
    raise

print("\n------------------------Processing data with gender segregation------------------------")
male_honest_epochs = []
male_deceitful_epochs = []
female_honest_epochs = []
female_deceitful_epochs = []

# Main processing loop
for participant_folder_name in participant_dirs:
    print(f"--- Processing: {participant_folder_name} ---")

    participant_folder_lower = participant_folder_name.lower()

    # Check if a known UUID is part of the folder name
    is_male = any(uuid in participant_folder_lower for uuid in male_uuids)
    is_female = any(uuid in participant_folder_lower for uuid in female_uuids)

    # Skip folder if it doesn't match any gender/UUID from the survey
    if not (is_male or is_female):
        print(f"Skipped: Folder {participant_folder_name} does not match any UUID from survey.")
        continue

    print(f"Matched folder to gender: {'Male' if is_male else 'Female'}")
    participant_path = os.path.join(DATA_FOLDER, participant_folder_name)

    # Find files for honest and deceitful conditions
    honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
    deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

    try:
        # Processing honest files
        for h_file in honest_files:
            raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
            prep_honest = preprocessing(raw_honest)  # we applying preprocessing like filtering
            epochs_h = create_epochs(prep_honest, master_event_id=master_event_id)

            if epochs_h and len(epochs_h) > 0:
                # Add to the appropriate list based on gender
                if is_male: male_honest_epochs.append(epochs_h)
                elif is_female: female_honest_epochs.append(epochs_h)

        # Processing deceitful files
        for d_file in deceitful_files:
            raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
            prep_deceitful = preprocessing(raw_deceitful)
            epochs_d = create_epochs(prep_deceitful, master_event_id=master_event_id)

            if epochs_d and len(epochs_d) > 0:
                if is_male: male_deceitful_epochs.append(epochs_d)
                elif is_female: female_deceitful_epochs.append(epochs_d)

    except Exception as e:
        print(f"Error processing {participant_folder_name}: {e}")

if male_honest_epochs and male_deceitful_epochs:
    # Concatenate epochs and calculate average
    evoked_male_honest = mne.concatenate_epochs(male_honest_epochs).average()
    evoked_male_deceitful = mne.concatenate_epochs(male_deceitful_epochs).average()

    # Prepare data for plotting
    evokeds_male = {"Honest (Males)": evoked_male_honest, "Deceitful (Males)": evoked_male_deceitful}

    # Draw the plot
    mne.viz.plot_compare_evokeds(evokeds_male,
                                 picks='eeg',
                                 legend='upper left',
                                 title="ERP Males: Honest vs Deceitful",
                                 show=False)
else:
    print("Not enough data to generate plot for males")

if female_honest_epochs and female_deceitful_epochs:
    # Concatenate epochs and calculate average
    evoked_female_honest = mne.concatenate_epochs(female_honest_epochs).average()
    evoked_female_deceitful = mne.concatenate_epochs(female_deceitful_epochs).average()

    # Prepare data for plotting
    evokeds_female = {"Honest (Females)": evoked_female_honest, "Deceitful (Females)": evoked_female_deceitful}

    # Draw the plot
    mne.viz.plot_compare_evokeds(evokeds_female,
                                 picks='eeg',
                                 legend='upper left',
                                 title="ERP Females: Honest vs Deceitful",
                                 show=False)
else:
    print("Not enough data to generate plot for females.")

plt.show()