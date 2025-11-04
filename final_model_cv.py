"""
Script to train and evaluate the final classification model on ALL participants
using cross-validation.

This script performs the full pipeline:
1. Finds all valid participant folders (excluding "popsute" and "stare").
2. Builds a global master_event_id map for event consistency.
3. Loops through all participants, processes their 'HONEST' and 'DECEITFUL'
   files using the imported preprocessing, epoching, and feature_extraction modules.
4. Uses PSD (Power Spectral Density) features.
5. Aggregates data from all participants into a single dataset (X_final, y_final).
6. Defines a tuned RandomForestClassifier pipeline.
7. Evaluates the model using 5-fold stratified cross-validation (cross_val_score).
8. Generates and displays a classification report and confusion matrix based on
   cross-validated predictions (cross_val_predict).
"""

import os
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from src.preprocessing import preprocessing
from src.epoching import create_epochs
from src.feature_extraction import extract_psd_features

warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More events than default colors available")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")

DATA_FOLDER = "dataset"

all_descriptions = set()

participant_dirs = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))
                    if "popsute" not in d.lower() and "stare" not in d.lower()]
print(f"Found {len(participant_dirs)} participant folders (after initial filtering).")

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
print(f"Global map created with {len(master_event_id)} unique events.")

all_X = []
all_y = []

for participant_folder_name in participant_dirs:
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

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced'
    )
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_final, y_final, cv=cv_strategy, scoring='accuracy', n_jobs=-1)

print(f"Mean Accuracy cv all participants: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

y_pred_cv = cross_val_predict(model, X_final, y_final, cv=cv_strategy, n_jobs=-1)

print(classification_report(y_final, y_pred_cv, target_names=['honest (0)', 'deceitful (1)']))

cm = confusion_matrix(y_final, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['honest', 'deceitful'])
disp.plot()
plt.show()