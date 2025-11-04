"""
Script for exploratory data analysis (EDA), comparing ERPs by gender.

Steps:
1. Finds all valid participant folders (excluding "corrupted" and "old" data).
2. Builds a global, master_event_id map for consistency.
3. Loads demographic data from "Ankiety.xlsx" using pandas.
4. Identifies male and female participants based on their UUIDs.
5. Loops through all participant folders, processes their data using
   preprocessing.py and epoching.py, and segregates the resulting
   Epochs objects into four lists based on condition (Honest/Deceitful)
   and gender (Male/Female).
6. Concatenates and averages the epochs for each of the four groups.
7. Generates and displays two ERP comparison plots:
    - Male: Honest vs. Deceitful
    - Female: Honest vs. Deceitful
"""

import pandas as pd
import mne
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import warnings
from src.preprocessing import preprocessing
from src.epoching import create_epochs

warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More events than default colors available")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")

DATA_FOLDER = "../dataset"
SURVEY_FILE = os.path.join(DATA_FOLDER, "Ankiety.xlsx")
UUID_COLUMN = 'UUID'
GENDER_COLUMN = 'Płeć'

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

try:
    df_survey = pd.read_excel(SURVEY_FILE)
    df_survey[UUID_COLUMN] = df_survey[UUID_COLUMN].astype(str).str.lower()
    df_survey[GENDER_COLUMN] = df_survey[GENDER_COLUMN].astype(str).str.lower()
except Exception as e:
    print(f"ERROR: Could not load or process {SURVEY_FILE}. {e}")
    raise e

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

for participant_folder_name in participant_dirs:
    print(f"--- Processing: {participant_folder_name} ---")

    participant_folder_lower = participant_folder_name.lower()
    is_male = any(uuid in participant_folder_lower for uuid in male_uuids)
    is_female = any(uuid in participant_folder_lower for uuid in female_uuids)

    if not (is_male or is_female):
        print(f"Skipped: Folder {participant_folder_name} does not match any UUID from survey.")
        continue

    print(f"Matched folder to gender: {'Male' if is_male else 'Female'}")
    participant_path = os.path.join(DATA_FOLDER, participant_folder_name)
    honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
    deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

    try:
        for h_file in honest_files:
            raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
            prep_honest = preprocessing(raw_honest)
            epochs_h = create_epochs(prep_honest, master_event_id=master_event_id)

            if epochs_h and len(epochs_h) > 0:
                if is_male: male_honest_epochs.append(epochs_h)
                elif is_female: female_honest_epochs.append(epochs_h)

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
    evoked_male_honest = mne.concatenate_epochs(male_honest_epochs).average()
    evoked_male_deceitful = mne.concatenate_epochs(male_deceitful_epochs).average()
    evokeds_male = {"Honest (Males)": evoked_male_honest, "Deceitful (Males)": evoked_male_deceitful}
    mne.viz.plot_compare_evokeds(evokeds_male,
                                 picks='eeg',
                                 legend='upper left',
                                 title="ERP Males: Honest vs Deceitful",
                                 show=False)
else:
    print("Not enough data to generate plot for males")

if female_honest_epochs and female_deceitful_epochs:
    evoked_female_honest = mne.concatenate_epochs(female_honest_epochs).average()
    evoked_female_deceitful = mne.concatenate_epochs(female_deceitful_epochs).average()
    evokeds_female = {"Honest (Females)": evoked_female_honest, "Deceitful (Females)": evoked_female_deceitful}
    mne.viz.plot_compare_evokeds(evokeds_female,
                                 picks='eeg',
                                 legend='upper left',
                                 title="ERP Females: Honest vs Deceitful",
                                 show=False)
else:
    print("Not enough data to generate plot for females.")

plt.show()