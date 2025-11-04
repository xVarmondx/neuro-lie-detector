import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocessing
from epoching import create_epochs
from feature_extraction import extract_psd_features
import mne
import os

def load_and_process_file(file_path):
    # A helper function to run the full pipeline for one file

    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    preprocessed_raw = preprocessing(raw)
    epochs = create_epochs(preprocessed_raw)
    X, _ = extract_psd_features(epochs)

    return X

# Test for one file
DATA_FOLDER = "dataset"
EXAMPLE_USER = "2D663E30"
EXAMPLE_EEG_FILENAME = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
EXAMPLE_EEG_PATH = os.path.join(DATA_FOLDER, EXAMPLE_USER, EXAMPLE_EEG_FILENAME)


print("--- Step 1: Loading and processing data ---")
HONEST_FILE_PATH = "../dataset/2D663E30/EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
DECEITFUL_FILE_PATH = "../dataset/2D663E30/EEG_ExperimentBlock.DECEITFUL_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
X_honest = load_and_process_file(HONEST_FILE_PATH)
X_deceitful = load_and_process_file(DECEITFUL_FILE_PATH)
y_honest = np.zeros(X_honest.shape[0])
y_deceitful = np.ones(X_deceitful.shape[0])
X = np.concatenate((X_honest, X_deceitful), axis=0)
y = np.concatenate((y_honest, y_deceitful), axis=0)
print(f"Final dataset created. X shape: {X.shape}, y shape: {y.shape}\n")

print("--- Step 2: Splitting data into train and test sets ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples\n")

print("--- Step 3: Training the classification model (Logistic Regression) ---")
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42, solver='liblinear', C=0.1)
)
model.fit(X_train, y_train)
print("Model training complete.\n")

# Krok 4: Ocena modelu
print("--- Step 4: Evaluating model performance ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f} ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['honest (0)', 'deceitful (1)']))

# --- NOWA SEKCJA: WIZUALIZACJA MACIERZY POMYŁEK ---
print("\n--- Displaying Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['honest', 'deceitful'])
disp.plot()
plt.title("Confusion Matrix")
plt.show()  # Pokaż wykres