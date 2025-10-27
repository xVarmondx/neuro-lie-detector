import os
import glob
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # We are using Random Forest
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from preprocessing import preprocessing
from epoching import create_epochs
from feature_extraction import extract_psd_features

warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="More events than default colors available")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown")
# warnings.filterwarnings("ignore", message="pick_types() is a legacy function...")

DATA_FOLDER = "dataset"

# Building event_id map. We need this map so that event IDs like PersonalDataField are consistent across all files.
all_descriptions = set() # Set to store unique event descriptions

# Find all valid participant directories excluding 'popsute' or 'stare'
participant_dirs = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))
                    if "popsute" not in d.lower() and "stare" not in d.lower()]
print(f"Found {len(participant_dirs)} participant folders")

# Loop through each participant folder - collecting event descriptions
for participant_uuid in participant_dirs:
    participant_path = os.path.join(DATA_FOLDER, participant_uuid)

    # Find all raw .fif files for this participant
    files = glob.glob(os.path.join(participant_path, "*_raw.fif"))

    # Loop through the files found
    for f in files:
        try:
            # Load file info only to read annotations
            raw = mne.io.read_raw_fif(f, preload=False, verbose=False)

            # Get unique annotation descriptions from this file
            descriptions = np.unique(raw.annotations.description)

            # Add descriptions found in this file to the global set
            all_descriptions.update(descriptions)
        except Exception as e:
            print(f"Skipped file {os.path.basename(f)}: {e}")

# Created the final map {description: id} after collecting all descriptions
master_event_id = {desc: i+1 for i, desc in enumerate(sorted(list(all_descriptions)))}
print(f"Global map created with {len(master_event_id)} unique events")


print(f"\n------------------------Processing data for ALL {len(participant_dirs)} participants------------------------")
all_X = [] # List to store feature matrices
all_y = [] # List to store corresponding labels

for participant_folder_name in participant_dirs:
    print(f"--- Processing folder: {participant_folder_name} ---")

    participant_path = os.path.join(DATA_FOLDER, participant_folder_name)

    # Find files for honest and deceitful conditions
    honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
    deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

    try:
        # Processing honest files
        for h_file in honest_files:
            raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
            prep_honest = preprocessing(raw_honest)  # We applying preprocessing like filtering

            # Create epochs using the global event map
            epochs_h = create_epochs(prep_honest, master_event_id=master_event_id)

            if epochs_h and len(epochs_h) > 0:
                # Extract features using psd features
                X, _ = extract_psd_features(epochs_h)
                all_X.append(X)
                all_y.append(np.zeros(X.shape[0])) # Label 0 for honest

        # Processing deceitful files
        for d_file in deceitful_files:
            raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
            prep_deceitful = preprocessing(raw_deceitful) # We applying preprocessing like filtering

            # Create epochs using the global event map
            epochs_d = create_epochs(prep_deceitful, master_event_id=master_event_id)

            if epochs_d and len(epochs_d) > 0:
                # Extract features using psd features
                X, _ = extract_psd_features(epochs_d)
                all_X.append(X)
                all_y.append(np.ones(X.shape[0])) # Label 1 for deceitful

    except Exception as e:
        print(f"Error processing {participant_folder_name}: {e}")

# Check if any data was loaded
if not all_X:
     print("CRITICAL ERROR: No data was loaded")
     exit() # Exit if no data is available

# Concatenate features and labels from all participants into single arrays
X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)
print(f"Final X shape: {X_final.shape}, Final y shape: {y_final.shape}")

# Split data into training and testing sets and test set(20%)
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



# Define the model pipeline (Scaler + Random Forest)
model = make_pipeline(
    StandardScaler(), # Scale features before training
    RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1,random_state=42, class_weight='balanced')
)

# Training model
model.fit(X_train, y_train)

# Make predictions on the unseen test data
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

# Plot validation matrix
disp_val.plot(ax=axes[0])
axes[0].set_title("Validation Set Confusion Matrix")

# Plot test matrix
disp_test.plot(ax=axes[1])
axes[1].set_title("Test Set Confusion Matrix")

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()