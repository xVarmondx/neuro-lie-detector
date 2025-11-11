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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
import tqdm
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.preprocessing import preprocessing
from src.epoching import create_epochs
from src.feature_extraction import extract_psd_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level('ERROR')
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
all_groups = []

for i, participant_folder_name in enumerate(tqdm.tqdm(participant_dirs, desc="Processing Participants")):
    #print(f"--- Processing folder: {participant_folder_name} ---")

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
                all_groups.append(np.full(X.shape[0], i))

        for d_file in deceitful_files:
            raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
            prep_deceitful = preprocessing(raw_deceitful)
            epochs_d = create_epochs(prep_deceitful, master_event_id=master_event_id)

            if epochs_d and len(epochs_d) > 0:
                X, _ = extract_psd_features(epochs_d)
                all_X.append(X)
                all_y.append(np.ones(X.shape[0]))
                all_groups.append(np.full(X.shape[0], i))

    except Exception as e:
        print(f"Error processing {participant_folder_name}: {e}")

if not all_X:
     print("CRITICAL ERROR: No data was loaded")
     exit()

X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)
groups = np.concatenate(all_groups, axis=0)

print(f"Final X shape: {X_final.shape}, Final y shape: {y_final.shape}")

models_to_test = {
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    ),
    "RandomForest": make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
    ),
    "SVM (RBF Kernel)": make_pipeline(
        StandardScaler(),
        SVC(class_weight='balanced', random_state=42, C=1.0)
    ),
    "XGBoost": make_pipeline(
        StandardScaler(),
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=100)
    )
}

n_subjets = len(np.unique(groups))
print(f"Found {n_subjets} unique groups for validation")

cv_strategy = LeaveOneGroupOut()

for model_name, model in models_to_test.items():
    print(f"Testing {model_name}")
    scores = cross_val_score(model, X_final, y_final, groups=groups ,cv=cv_strategy, scoring='accuracy', n_jobs=-1)

    print(f"Average accuracy: {np.mean(scores)*100:.2f}%")
    print(f"Scores per subject: {scores}")
    y_pred_cv = cross_val_predict(model, X_final, y_final, groups=groups, cv=cv_strategy, n_jobs=-1)
    print(classification_report(y_final, y_pred_cv, target_names=['honest (0)', 'deceitful (1)']))

    cm = confusion_matrix(y_final, y_pred_cv)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['honest', 'deceitful'])
    disp.plot()
    plt.title("Confusion matrix")
    plt.show()