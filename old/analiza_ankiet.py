import os
import mne
import matplotlib.pyplot as plt  # Ważny import!

# --- Ścieżki (bez zmian) ---
folder_danych = "dataset"
folder_uczestnika = "02F6BC66"
nazwa_pliku_eeg = "EEG_ExperimentBlock.DECEITFUL_RESPONSE_TO_FAKE_IDENTITY_raw.fif"
sciezka_do_eeg = os.path.join(folder_danych, folder_uczestnika, nazwa_pliku_eeg)

try:
    raw = mne.io.read_raw_fif(sciezka_do_eeg, preload=True)
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')

    # --- ICA ---
    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw_filtered)
    ica.exclude = [0]  # Usuwamy komponent ICA000

    # --- Przygotowanie wykresów (bez natychmiastowego pokazywania) ---
    print("Przygotowywanie wykresów...")

    # Wykres 1: Przed usunięciem
    raw_filtered.plot(duration=10, n_channels=20, title="Przed usunięciem ICA", show=False)

    # Wykres 2: Usuwany komponent
    ica.plot_sources(raw_filtered, picks=[0], title="Usuwany komponent: ICA000 - Mruganie", show=False)

    # Wykres 3: Po usunięciu (czysty sygnał)
    raw_cleaned = raw_filtered.copy()
    ica.apply(raw_cleaned)
    raw_cleaned.plot(duration=10, n_channels=20, title="Po usunięciu ICA (sygnał czysty)", show=False)

    # --- NOWA, KLUCZOWA CZĘŚĆ ---
    # To polecenie otwiera wszystkie przygotowane okna i czeka, aż je zamkniesz
    print("Wyświetlanie wszystkich wykresów. Zamknij je, aby zakończyć skrypt.")
    plt.show()

except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku: {sciezka_do_eeg}")
except Exception as e:
    print(f"Wystąpił nieoczekiwany błąd: {e}")