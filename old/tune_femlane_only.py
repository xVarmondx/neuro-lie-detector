import os
import glob
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV  # <-- NOWY IMPORT
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline  # Pipeline jest potrzebny do siatki
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from preprocessing import preprocessing
from old.feature_extraction_freq import extract_psd_features  # Używamy cech PSD


#
# Klasa EEGDataLoader pozostaje bez zmian
#
class EEGDataLoader:
    """
    Klasa EEGDataLoader (bez zmian)
    """

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.participants = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"Found {len(self.participants)} total participants in the dataset.")

    def get_all_data(self):
        all_X = []
        all_y = []
        for participant_uuid in self.participants:
            print(f"\n--- Processing participant: {participant_uuid} ---")
            participant_path = os.path.join(self.dataset_path, participant_uuid)
            try:
                honest_file = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))[0]
                deceitful_file = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))[0]
            except IndexError:
                print(f"Warning: Missing data files for participant {participant_uuid}. Skipping.")
                continue

            X_honest = self._process_single_file(honest_file)
            if X_honest is not None:
                all_X.append(X_honest)
                all_y.append(np.zeros(X_honest.shape[0]))

            X_deceitful = self._process_single_file(deceitful_file)
            if X_deceitful is not None:
                all_X.append(X_deceitful)
                all_y.append(np.ones(X_deceitful.shape[0]))

        X_final = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        return X_final, y_final

    def _process_single_file(self, file_path):
        """Helper function to run the full pipeline for one file."""
        try:
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            preprocessed_raw = preprocessing(raw)

            # Uproszczona logika epokowania (działała poprzednio)
            events, event_id = mne.events_from_annotations(preprocessed_raw, verbose=False)
            stimulus_event_ids = {k: v for k, v in event_id.items() if 'PersonalDataField' in k}
            epochs = mne.Epochs(
                preprocessed_raw, events=events, event_id=stimulus_event_ids,
                tmin=-0.2, tmax=0.8, preload=True, baseline=(-0.2, 0),
                reject=None, verbose=False
            )

            if epochs and len(epochs) > 0:
                # Używamy cech PSD
                X, y = extract_psd_features(epochs)
                return X
            return None
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            return None


if __name__ == '__main__':
    DATA_FOLDER = "dataset"

    # --- Sekcja filtrowania kobiet (bez zmian) ---
    print("--- EXPERIMENT: Tuning model for Female Participants Only (PSD FEATURES) ---")
    SURVEY_FILE = os.path.join(DATA_FOLDER, "Ankiety.xlsx")
    KOLUMNA_UUID = 'UUID'
    KOLUMNA_PLEC = 'Płeć'
    try:
        df_survey = pd.read_excel(SURVEY_FILE)
        df_survey[KOLUMNA_UUID] = df_survey[KOLUMNA_UUID].astype(str).str.lower()
        df_survey[KOLUMNA_PLEC] = df_survey[KOLUMNA_PLEC].astype(str).str.lower()
        female_uuids = df_survey[df_survey[KOLUMNA_PLEC] == 'k'][KOLUMNA_UUID].tolist()
        print(f"Found {len(female_uuids)} female participants in survey.")
    except Exception as e:
        print(f"BŁĄD: Nie mogłem wczytać lub przetworzyć Ankiety.xlsx: {e}")
        exit()

    # --- Krok 1: Ładowanie danych (bez zmian) ---
    print("--- Step 1: Loading data for FEMALE participants ---")
    data_loader = EEGDataLoader(DATA_FOLDER)
    original_participant_count = len(data_loader.participants)
    data_loader.participants = [
        p_folder for p_folder in data_loader.participants
        if any(uuid in p_folder.lower() for uuid in female_uuids)
    ]
    print(
        f"Filtered participant list: running on {len(data_loader.participants)} / {original_participant_count} participants.")
    X, y = data_loader.get_all_data()

    if X.shape[0] == 0:
        print("\nNo data was loaded. Exiting.")
    else:
        print(f"\n--- Total dataset loaded. X shape: {X.shape}, y shape: {y.shape} ---")

        # --- Krok 2: Podział danych (bez zmian) ---
        print("\n--- Step 2: Splitting data into train and test sets ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")

        # --- NOWY KROK 3: Definicja Pipeline i Siatki Parametrów ---
        print("\n--- Step 3: Setting up Pipeline and Hyperparameter Grid ---")

        # Tworzymy potok (pipeline)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        # Definiujemy siatkę parametrów do przeszukania
        # Używamy składni 'nazwakroku__parametr'
        param_grid = {
            'model__n_estimators': [100, 200, 300],  # Liczba drzew
            'model__max_depth': [None, 10, 20],  # Maksymalna głębokość
            'model__min_samples_leaf': [1, 2, 4]  # Minimalne próbki w liściu
        }

        # --- NOWY KROK 4: Uruchomienie GridSearchCV ---
        print("\n--- Step 4: Running GridSearchCV (This will take a long time!) ---")

        # Tworzymy obiekt GridSearchCV
        # cv=5 -> 5-krotna walidacja krzyżowa
        # n_jobs=-1 -> Użyj wszystkich dostępnych rdzeni CPU
        # verbose=2 -> Pokazuj szczegółowe logi postępu
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )

        # Uruchamiamy strojenie (tylko na danych treningowych)
        grid_search.fit(X_train, y_train)

        print("Model tuning complete.")
        print(f"Best parameters found: {grid_search.best_params_}")

        # 'grid_search' automatycznie staje się najlepszym znalezionym modelem
        best_model = grid_search.best_estimator_

        # --- NOWY KROK 5: Ocena Najlepszego Modelu ---
        print("\n--- Step 5: Final model evaluation (using best found model) ---")
        y_pred = best_model.predict(X_test)  # Używamy .predict na X_test

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy on Test Set (FEMALE + TUNED RF): {accuracy:.2f} ({accuracy * 100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['honest (0)', 'deceitful (1)']))

        print("\n--- Displaying Confusion Matrix for the final model ---")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['honest', 'deceitful'])
        disp.plot()
        plt.title("Final Confusion Matrix (Tuned Random Forest - FEMALE ONLY)")
        plt.show()