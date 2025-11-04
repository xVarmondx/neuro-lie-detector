import mne
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocessing
from epoching import create_epochs  # Importujemy zaktualizowaną funkcję

# --- Konfiguracja ścieżek ---
DATA_FOLDER = "dataset"
SURVEY_FILE = os.path.join(DATA_FOLDER, "Ankiety.xlsx")
participant_dirs = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]
print(f"Found {len(participant_dirs)} participant folders.")

# --- NOWY KROK 1: Zbuduj globalną (master) mapę event_id ---
print("--- Building master event_id map ---")
all_descriptions = set()  # Używamy 'set' aby automatycznie obsługiwać duplikaty

for participant_uuid in participant_dirs:
    participant_path = os.path.join(DATA_FOLDER, participant_uuid)
    files = glob.glob(os.path.join(participant_path, "*_raw.fif"))
    for f in files:
        try:
            # Wczytujemy plik tylko po to, by odczytać adnotacje (preload=False jest szybsze)
            raw = mne.io.read_raw_fif(f, preload=False, verbose=False)
            descriptions = np.unique(raw.annotations.description)
            all_descriptions.update(descriptions)  # Dodajemy unikalne opisy do naszego zbioru
        except Exception as e:
            print(f"Skipping file {f}: {e}")

# Tworzymy finalny słownik: {opis: numer_id}
# Sortujemy, aby zapewnić spójną kolejność za każdym razem
master_event_id = {desc: i + 1 for i, desc in enumerate(sorted(list(all_descriptions)))}
print(f"Master map created with {len(master_event_id)} unique events.")
print("--- Master map build complete. ---")

# --- Krok 2: Wczytanie i przetworzenie epok (już z użyciem mapy) ---
all_honest_epochs = []
all_deceitful_epochs = []

for participant_uuid in participant_dirs:
    participant_path = os.path.join(DATA_FOLDER, participant_uuid)
    print(f"--- Processing participant: {participant_uuid} ---")

    honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
    deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

    try:
        for h_file in honest_files:
            raw_h = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
            prep_h = preprocessing(raw_h)
            # Przekazujemy naszą globalną mapę do funkcji!
            epochs_h = create_epochs(prep_h, master_event_id=master_event_id)
            if epochs_h and len(epochs_h) > 0:
                all_honest_epochs.append(epochs_h)

        for d_file in deceitful_files:
            raw_d = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
            prep_d = preprocessing(raw_d)
            # Przekazujemy naszą globalną mapę do funkcji!
            epochs_d = create_epochs(prep_d, master_event_id=master_event_id)
            if epochs_d and len(epochs_d) > 0:
                all_deceitful_epochs.append(epochs_d)

    except Exception as e:
        print(f"Error processing {participant_uuid}: {e}")

print("\n--- Data loading complete. ---")

# --- Krok 3: Połączenie epok i stworzenie uśrednionych ERP ---
if all_honest_epochs and all_deceitful_epochs:
    # Ten kod powinien teraz zadziałać bez błędu
    combined_honest = mne.concatenate_epochs(all_honest_epochs)
    combined_deceitful = mne.concatenate_epochs(all_deceitful_epochs)

    print(f"Total honest epochs collected: {len(combined_honest)}")
    print(f"Total deceitful epochs collected: {len(combined_deceitful)}")

    evoked_honest = combined_honest.average()
    evoked_deceitful = combined_deceitful.average()

    print("\n--- Krok 4: Wizualizacja porównawcza ERP ---")
    evokeds = {"Honest (Szczere)": evoked_honest, "Deceitful (Nieszczere)": evoked_deceitful}

    mne.viz.plot_compare_evokeds(evokeds,
                                 picks='eeg',
                                 legend='upper left',
                                 title="ERP: Honest vs Deceitful (All Participants)")

    diff_erp = mne.combine_evoked([evoked_honest, evoked_deceitful], weights=[1, -1])
    diff_erp.plot_topomap(times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.show()
else:
    print("No epochs were loaded. Cannot perform analysis.")