import os
import glob
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
# === MODIFIED IMPORT: Need train_test_split twice ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # We are using Random Forest
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from preprocessing import preprocessing
from epoching import create_epochs
from feature_extraction import extract_psd_features

# --- Suppress known warnings ---
warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More events than default colors available")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")

# --- Configuration ---
DATA_FOLDER = "dataset"
# SURVEY_FILE = os.path.join(DATA_FOLDER, "Ankiety.xlsx") # Not needed
# UUID_COLUMN = "UUID"
# GENDER_COLUMN = "Płeć"

# =========================================================================
# === STEP 1: BUILD GLOBAL (MASTER) event_id MAP ===
# =========================================================================
print("--- Building global event_id map ---")
all_descriptions = set()
participant_dirs = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))
                    if "popsute" not in d.lower() and "stare" not in d.lower()]
print(f"Found {len(participant_dirs)} participant folders (after initial filtering)")

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
print("--- Map building complete ---")

# =========================================================================
# === STEP 2: PROCESSING DATA FOR ALL VALID PARTICIPANTS ===
# =========================================================================
print(f"\n--- Processing data for ALL {len(participant_dirs)} valid participants ---")
all_X = []
all_y = []

for participant_folder_name in participant_dirs:
    print(f"--- Processing folder: {participant_folder_name} ---")
    participant_path = os.path.join(DATA_FOLDER, participant_folder_name)
    honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
    deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

    try:
        # Processing honest files
        for h_file in honest_files:
            raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
            prep_honest = preprocessing(raw_honest)
            epochs_h = create_epochs(prep_honest, master_event_id=master_event_id)
            if epochs_h and len(epochs_h) > 0:
                X, _ = extract_psd_features(epochs_h)
                all_X.append(X)
                all_y.append(np.zeros(X.shape[0])) # Label 0

        # Processing deceitful files
        for d_file in deceitful_files:
            raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
            prep_deceitful = preprocessing(raw_deceitful)
            epochs_d = create_epochs(prep_deceitful, master_event_id=master_event_id)
            if epochs_d and len(epochs_d) > 0:
                X, _ = extract_psd_features(epochs_d)
                all_X.append(X)
                all_y.append(np.ones(X.shape[0])) # Label 1

    except Exception as e:
        print(f"Error processing {participant_folder_name}: {e}")

if not all_X:
     print("CRITICAL ERROR: No data was loaded")
     exit()

X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)
print(f"Final X shape: {X_final.shape}, Final y shape: {y_final.shape}")

# =========================================================================
# === STEP 3: SPLIT DATA INTO TRAIN, VALIDATION, AND TEST SETS ===
# =========================================================================
print("\n--- Splitting data into Train, Validation, and Test sets ---")

# First split: Separate out the Test set (e.g., 20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_final, y_final,
    test_size=0.2, # Reserve 20% for the final test set
    random_state=42,
    stratify=y_final
)

# Second split: Split the remaining data (X_temp) into Train and Validation
# We want validation to be 20% OF THE ORIGINAL DATASET.
# Since X_temp is 80% of the original, we need validation to be 0.2 / 0.8 = 0.25 of X_temp.
val_size_fraction = 0.25 # (20% validation / 80% remaining = 25%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_size_fraction,
    random_state=42, # Use the same random state for consistency if desired, or different if not
    stratify=y_temp # Stratify based on the temporary labels
)

print(f"Total samples: {len(X_final)}")
print(f"Training set size:   {len(X_train)} ({len(X_train)/len(X_final)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X_final)*100:.1f}%)")
print(f"Test set size:       {len(X_test)} ({len(X_test)/len(X_final)*100:.1f}%)")


# =========================================================================
# === STEP 4: DEFINE AND TRAIN THE AI MODEL (ON TRAINING SET) ===
# =========================================================================
print("\n--- Starting AI model training (on Training set) ---")

# Define the model pipeline (Scaler + Random Forest with tuned params)
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, random_state=42, class_weight='balanced')
)

# Training model ONLY on the training data
print("Training model...")
model.fit(X_train, y_train)
print("Training complete.")

# =========================================================================
# === STEP 5: EVALUATE ON VALIDATION AND TEST SETS ===
# =========================================================================
print("\n--- Evaluating model performance ---")

# --- Evaluate on Validation Set ---
print("\n--- Validation Set Evaluation ---")
y_pred_val = model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print(f"Accuracy on Validation Set: {accuracy_val:.3f} ({accuracy_val*100:.1f}%)")
print("\nClassification Report (Validation Set):")
print(classification_report(y_val, y_pred_val, target_names=['honest (0)', 'deceitful (1)']))
print("\nConfusion Matrix (Validation Set):")
cm_val = confusion_matrix(y_val, y_pred_val)
disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=['honest', 'deceitful'])
# We'll display plots together at the end

# --- Evaluate on Test Set ---
print("\n--- Test Set Evaluation ---")
y_pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"\nFINAL ACCURACY (on Test Set): {accuracy_test:.3f} ({accuracy_test*100:.1f}%)")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['honest (0)', 'deceitful (1)']))
print("\nConfusion Matrix (Test Set):")
cm_test = confusion_matrix(y_test, y_pred_test)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['honest', 'deceitful'])
# We'll display plots together at the end

# =========================================================================
# === STEP 6: DISPLAY PLOTS ===
# =========================================================================
print("\n--- Displaying Confusion Matrices ---")
# Create subplots to show both matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot validation matrix on the first subplot
disp_val.plot(ax=axes[0])
axes[0].set_title("Validation Set Confusion Matrix")

# Plot test matrix on the second subplot
disp_test.plot(ax=axes[1])
axes[1].set_title("Test Set Confusion Matrix")

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()

print("\n--- Finished final AI pipeline with Train/Validation/Test split ---")