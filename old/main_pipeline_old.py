import os
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Importujemy nasze sprawdzone funkcje z poprzednich plików
from preprocessing import preprocessing
from epoching import create_epochs
from feature_extraction_freq import extract_psd_features


class EEGDataLoader:
    """
    A class to load, preprocess, and extract features from the entire EEG dataset.
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.participants = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"Found {len(self.participants)} participants in the dataset.")

    def get_all_data(self):
        """
        Processes data for all participants and returns a single feature matrix and labels vector.
        """
        all_X = []
        all_y = []

        for participant_uuid in self.participants:
            print(f"\n--- Processing participant: {participant_uuid} ---")
            participant_path = os.path.join(self.dataset_path, participant_uuid)

            # Define paths for honest and deceitful responses
            # We use glob to find the files without hardcoding the full name
            try:
                honest_file = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))[0]
                deceitful_file = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))[0]
            except IndexError:
                print(f"Warning: Missing data files for participant {participant_uuid}. Skipping.")
                continue

            # Process honest file
            X_honest = self._process_single_file(honest_file)
            if X_honest is not None:
                all_X.append(X_honest)
                all_y.append(np.zeros(X_honest.shape[0]))  # Label 0 for honest

            # Process deceitful file
            X_deceitful = self._process_single_file(deceitful_file)
            if X_deceitful is not None:
                all_X.append(X_deceitful)
                all_y.append(np.ones(X_deceitful.shape[0]))  # Label 1 for deceitful

        # Concatenate data from all participants into single numpy arrays
        X_final = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)

        return X_final, y_final

    def _process_single_file(self, file_path):
        """Helper function to run the full pipeline for one file."""
        try:
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            preprocessed_raw = preprocessing(raw)
            epochs = create_epochs(preprocessed_raw)
            if epochs and len(epochs) > 0:
                X, _ = extract_psd_features(epochs)
                return X
            return None
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            return None


if __name__ == '__main__':
    DATA_FOLDER = "dataset"

    # Krok 1: Użycie naszej nowej klasy do wczytania wszystkich danych
    print("--- Step 1: Loading data from all participants using EEGDataLoader ---")
    data_loader = EEGDataLoader(DATA_FOLDER)
    X, y = data_loader.get_all_data()

    if X.shape[0] == 0:
        print("\nNo data was loaded. Exiting.")
    else:
        print(f"\n--- Total dataset loaded. X shape: {X.shape}, y shape: {y.shape} ---")

        # Krok 2: Podział danych na zbiór treningowy i testowy (jak poprzednio)
        print("\n--- Step 2: Splitting data into train and test sets ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")

        # Krok 3: Trening finalnego modelu
        print("\n--- Step 3: Training the final model (Random Forest) ---")
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        )
        model.fit(X_train, y_train)
        print("Model training complete.")

        # Krok 4: Ocena finalnego modelu
        print("\n--- Step 4: Final model evaluation ---")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy on Test Set: {accuracy:.2f} ({accuracy * 100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['honest (0)', 'deceitful (1)']))

        # Wizualizacja Macierzy Pomyłek
        print("\n--- Displaying Confusion Matrix for the final model ---")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['honest', 'deceitful'])
        disp.plot()
        plt.title("Final Confusion Matrix (All Participants)")
        plt.show()