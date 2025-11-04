"""
Main script for training and evaluating the final AI model.

This script executes the complete ML pipeline, incorporating insights
from the EDA (Exploratory Data Analysis) which showed distinct
patterns for female participants.
"""

import os
import glob
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from src.preprocessing import preprocessing
from src.epoching import create_epochs
from src.feature_extraction import extract_psd_features

warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More events than default colors available")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")

DATA_FOLDER = "dataset"
SURVEY_FILE = os.path.join(DATA_FOLDER, "Ankiety.xlsx")
UUID_COLUMN = "UUID"
GENDER_COLUMN = "Płeć"

all_descriptions = set()

participant_dirs = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))
                    if "popsute" not in d.lower() and "stare" not in d.lower()]
print(f"Found {len(participant_dirs)} participant folders")


for participant_uuid in participant_dirs:
    participant_path = os.path.join(DATA_FOLDER, participant_uuid)
    files = glob.glob(os.path.join(participant_path, "*_raw.fif"))

    for f in files:
        try:
            raw = mne.io.read_raw_fif(f, preload=False, verbose=False)
            descriptions = np.unique(raw.annotations.description)
            all_descriptions.update(descriptions)
        except Exception as e:
            print(f"Skipped file {os.path.basename(f)}: {e}")

master_event_id = {desc: i+1 for i, desc in enumerate(sorted(list(all_descriptions)))}
print(f"Global map created with {len(master_event_id)} unique events")


try:
    df_survey = pd.read_excel(SURVEY_FILE)

    df_survey[UUID_COLUMN] = df_survey[UUID_COLUMN].astype(str).str.lower()
    df_survey[GENDER_COLUMN] = df_survey[GENDER_COLUMN].astype(str).str.lower()
except Exception as e:
    print(f"ERROR: Could not load or process {SURVEY_FILE}: {e}")
    raise e


try:
    female_uuids = df_survey[df_survey[GENDER_COLUMN] == 'k'][UUID_COLUMN].tolist()
    print(f"Found {len(female_uuids)} female participants in the survey")

    if not female_uuids:
        print("WARNING: No female participants found matching the criteria")

except KeyError:
    print(f"ERROR: Column '{GENDER_COLUMN}' or '{UUID_COLUMN}' not found in the survey file")
    raise


participant_dirs_filtered = [
    p_folder for p_folder in participant_dirs
    if any(uuid in p_folder.lower() for uuid in female_uuids)
]
print(f"Running the model on {len(participant_dirs_filtered)} female participants.")

print("\n------------------------Processing data for female participants only------------------------")
all_X = []
all_y = []

for participant_folder_name in participant_dirs_filtered:
    print(f"--- Processing folder: {participant_folder_name} ---")

    participant_path = os.path.join(DATA_FOLDER, participant_folder_name)

    honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
    deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

    try:
        for h_file in honest_files:
            raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
            prep_honest = preprocessing(raw_honest)

            epochs_h = create_epochs(prep_honest, master_event_id=master_event_id)

            if epochs_h and len(epochs_h) > 0:
                X, _ = extract_psd_features(epochs_h)
                all_X.append(X)
                all_y.append(np.zeros(X.shape[0]))

        for d_file in deceitful_files:
            raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
            prep_deceitful = preprocessing(raw_deceitful)

            epochs_d = create_epochs(prep_deceitful, master_event_id=master_event_id)

            if epochs_d and len(epochs_d) > 0:
                X, _ = extract_psd_features(epochs_d)
                all_X.append(X)
                all_y.append(np.ones(X.shape[0]))

    except Exception as e:
        print(f"Error processing {participant_folder_name}: {e}")

if not all_X:
     print("CRITICAL ERROR: No data was loaded")
     exit()

X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)
print(f"Final X shape: {X_final.shape}, Final y shape: {y_final.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.3, random_state=42, stratify=y_final
)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")


model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1,random_state=42, class_weight='balanced')
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal accuracy (females only): {accuracy:.3f} ({accuracy*100:.1f}%)")

print(classification_report(y_test, y_pred, target_names=['honest (0)', 'deceitful (1)']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['honest', 'deceitful'])
disp.plot()
plt.title("Model females only")
plt.show()