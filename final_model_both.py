"""
Main script for training and evaluating the final AI model.

Script to train and evaluate a classification model on ALL participants
(both male and female), using a fixed Train/Validation/Test split.
"""
import os
import glob
import numpy as np
import mne
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

DATA_FOLDER = "../dataset"

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


print(f"\n------------------------Processing data for ALL {len(participant_dirs)} participants------------------------")
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

X_temp, X_test, y_temp, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

val_size_fraction = 0.25
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_fraction, random_state=42, stratify=y_temp)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"Total samples: {len(X_final)}")
print(f"Training set size:   {len(X_train)} ({len(X_train)/len(X_final)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X_final)*100:.1f}%)")
print(f"Test set size:       {len(X_test)} ({len(X_test)/len(X_final)*100:.1f}%)")

model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1,random_state=42, class_weight='balanced')
)

model.fit(X_train, y_train)

y_pred_val = model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print(f"Accuracy on Validation Set: {accuracy_val:.3f} ({accuracy_val*100:.1f}%)")
print(classification_report(y_val, y_pred_val, target_names=['honest (0)', 'deceitful (1)']))
cm_val = confusion_matrix(y_val, y_pred_val)
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['honest', 'deceitful'])

y_pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"\nFINAL ACCURACY (on Test Set): {accuracy_test:.3f} ({accuracy_test*100:.1f}%)")
print(classification_report(y_test, y_pred_test, target_names=['honest (0)', 'deceitful (1)']))
cm_test = confusion_matrix(y_test, y_pred_test)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['honest', 'deceitful'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

disp_val.plot(ax=axes[0])
axes[0].set_title("Validation Set Confusion Matrix")

disp_test.plot(ax=axes[1])
axes[1].set_title("Test Set Confusion Matrix")

plt.tight_layout()
plt.show()