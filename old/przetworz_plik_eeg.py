import os
import mne
import glob
import pandas as pd
import numpy as np

# --- KONFIGURACJA ---
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

    # --- NOWA, ODPORNA NA BŁĘDY SEKCJA ICA ---
    try:
        # Używamy elastycznego progu, aby unikać warningów
        ica = mne.preprocessing.ICA(n_components=0.999, max_iter='auto', random_state=97, verbose='WARNING')
        ica.fit(raw)

        # Automatyczne znajdowanie komponentu mrugania
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
        if eog_indices:
            ica.exclude = eog_indices
            ica.apply(raw, verbose='WARNING')
        else:
            print(
                "!!! OSTRZEŻENIE: Nie udało się automatycznie znaleźć komponentu mrugania. Dane mogą zawierać artefakty.")

    except RuntimeError as e:
        # Jeśli ICA zawiedzie z powodu problemów z danymi, wypisz ostrzeżenie i kontynuuj bez czyszczenia ICA
        print(
            f"!!! OSTRZEŻENIE KRYTYCZNE: ICA nie powiodło się dla pliku {os.path.basename(sciezka_do_pliku)}. Powód: {e}")
        print("Kontynuowanie bez usuwania artefaktów ICA dla tego pliku.")
        # Kontynuujemy z danymi, które są tylko przefiltrowane

    events, _ = mne.events_from_annotations(raw, verbose='WARNING')
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, preload=True,
                        baseline=(None, 0), reject=None, verbose='WARNING')

    dane_p300 = epochs.copy().crop(tmin=0.3, tmax=0.6).get_data()
    srednia_amplituda_p300 = np.mean(dane_p300, axis=(1, 2))

    wyniki = []
    for amplituda in srednia_amplituda_p300:
        wyniki.append({"amplituda_p300": amplituda, "warunek": etykieta})

    return wyniki


# --- GŁÓWNY SKRYPT (bez zmian) ---
print("Rozpoczynanie ekstrakcji cech dla wszystkich uczestników...")
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

    # Dodaj informacje demograficzne do wyników
    for r in wyniki_prawda + wyniki_klamstwo:
        r.update({"uuid": uuid, "wiek": uczestnik['Wiek'], "plec": uczestnik['Płeć']})

    wszystkie_wyniki.extend(wyniki_prawda)
    wszystkie_wyniki.extend(wyniki_klamstwo)

df_cechy = pd.DataFrame(wszystkie_wyniki)
nazwa_pliku_wynikowego = "features.csv"
df_cechy.to_csv(nazwa_pliku_wynikowego, index=False)

print(f"\n\nZakończono! Zbudowano zbiór danych i zapisano do pliku '{nazwa_pliku_wynikowego}'.")
print("Oto pierwsze 5 wierszy:")
print(df_cechy.head())