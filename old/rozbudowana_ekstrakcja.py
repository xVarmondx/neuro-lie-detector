import os
import mne
import glob
import pandas as pd
import numpy as np

# --- KONFIGURACJA (bez zmian) ---
FOLDER_DANYCH = "dataset"
PLIK_ANKIET = os.path.join(FOLDER_DANYCH, "Ankiety.xlsx")
SLOWO_KLUCZOWE_PRAWDA = "HONEST_RESPONSE_TO_TRUE_IDENTITY"
SLOWO_KLUCZOWE_KLAMSTWO = "DECEITFUL_RESPONSE_TO_TRUE_IDENTITY"


def znajdz_plik_uczestnika(folder_uczestnika, slowo_kluczowe):
    wzorzec = os.path.join(folder_uczestnika, f"*{slowo_kluczowe}*.fif")
    pliki = glob.glob(wzorzec)
    return pliki[0] if pliki else None


def przetworz_i_wyekstrahuj_cechy(sciezka_do_pliku, etykieta):
    if not sciezka_do_pliku:
        return []

    print(f"--- Przetwarzanie: {os.path.basename(sciezka_do_pliku)} ---")

    raw = mne.io.read_raw_fif(sciezka_do_pliku, preload=True, verbose='WARNING')
    raw.filter(l_freq=1.0, h_freq=40.0, verbose='WARNING')

    try:
        ica = mne.preprocessing.ICA(n_components=0.999, max_iter='auto', random_state=97, verbose='WARNING')
        ica.fit(raw)
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
        if eog_indices:
            ica.exclude = eog_indices
            ica.apply(raw, verbose='WARNING')
    except RuntimeError:
        print(f"!!! OSTRZEŻENIE KRYTYCZNE: ICA nie powiodło się dla pliku.")

    events, _ = mne.events_from_annotations(raw, verbose='WARNING')
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, preload=True,
                        baseline=(None, 0), reject=None, verbose='WARNING')

    wyniki = []
    for i in range(len(epochs)):
        epoch = epochs[i]

        # Cechy z domeny czasu
        p300_window = epoch.copy().crop(tmin=0.3, tmax=0.6).get_data()
        mean_amp = np.mean(p300_window)
        peak_amp = np.max(p300_window)

        # Cechy z domeny częstotliwości
        spectrum = epoch.compute_psd(method='welch', fmin=4, fmax=30, verbose='WARNING')

        # --- OSTATECZNA POPRAWKA ---
        # Pobieramy dane i częstotliwości w sposób gwarantujący ich zgodność
        psd = spectrum.get_data()
        freqs = spectrum.freqs
        # ---------------------------

        # Uśrednianie mocy w pasmach
        theta_power = np.mean(psd[0, :, (freqs >= 4) & (freqs < 8)])
        alpha_power = np.mean(psd[0, :, (freqs >= 8) & (freqs < 12)])
        beta_power = np.mean(psd[0, :, (freqs >= 12) & (freqs < 30)])

        wyniki.append({
            "p300_mean_amp": mean_amp, "p300_peak_amp": peak_amp,
            "theta_power": theta_power, "alpha_power": alpha_power, "beta_power": beta_power,
            "warunek": etykieta
        })

    return wyniki


# --- GŁÓWNY SKRYPT (bez zmian) ---
print(">>> Rozpoczynanie ROZBUDOWANEJ ekstrakcji cech dla wszystkich uczestników...")
df_ankiety = pd.read_excel(PLIK_ANKIET)
wszystkie_wyniki = []

for index, uczestnik in df_ankiety.iterrows():
    uuid = uczestnik['UUID']
    folder_uczestnika = os.path.join(FOLDER_DANYCH, uuid)

    if not os.path.isdir(folder_uczestnika):
        continue

    print(f"\nPrzetwarzanie uczestnika: {uuid}")

    plik_prawda = znajdz_plik_uczestnika(folder_uczestnika, SLOWO_KLUCZOWE_PRAWDA)
    plik_klamstwo = znajdz_plik_uczestnika(folder_uczestnika, SLOWO_KLUCZOWE_KLAMSTWO)

    wyniki_prawda = przetworz_i_wyekstrahuj_cechy(plik_prawda, "Prawda")
    wyniki_klamstwo = przetworz_i_wyekstrahuj_cechy(plik_klamstwo, "Kłamstwo")

    for r in wyniki_prawda + wyniki_klamstwo:
        r.update({"uuid": uuid, "wiek": uczestnik['Wiek'], "plec": uczestnik['Płeć']})

    wszystkie_wyniki.extend(wyniki_prawda)
    wszystkie_wyniki.extend(wyniki_klamstwo)

df_cechy = pd.DataFrame(wszystkie_wyniki)
nazwa_pliku_wynikowego = "features_rozbudowane.csv"
df_cechy.to_csv(nazwa_pliku_wynikowego, index=False)

print(f"\n\n✅ Zakończono! Zbudowano rozbudowany zbiór danych i zapisano do pliku '{nazwa_pliku_wynikowego}'.")
print("Oto pierwsze 5 wierszy:")
print(df_cechy.head())