import mne
import numpy as np


def extract_hybrid_features(epochs_data, psd_fmin=4., psd_fmax=30., erp_tmin=0.4, erp_tmax=0.7):
    """
    Wyciąga hybrydowy zestaw cech (PSD + ERP) z obiektu MNE Epochs.

    1. Cechy PSD: Moc w pasmach (Theta, Alpha, Beta) dla każdego kanału.
    2. Cechy ERP: Średnia amplituda w krytycznym oknie czasowym (P300) dla każdego kanału.
    """

    # --- 1. Ekstrakcja Cech Częstotliwościowych (PSD) ---
    # (Ten kod jest taki sam jak w feature_extraction_freq.py)

    FREQ_BANDS = {
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30]
    }

    spectrum = epochs_data.compute_psd(method='welch', fmin=psd_fmin, fmax=psd_fmax, picks='eeg', verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)

    X_psd = []
    for epoch_psds in psds:
        epoch_features = []
        for ch_psds in epoch_psds:
            for band, f_range in FREQ_BANDS.items():
                band_indices = np.where((freqs >= f_range[0]) & (freqs < f_range[1]))[0]
                avg_power = np.mean(ch_psds[band_indices])
                epoch_features.append(avg_power)
        X_psd.append(epoch_features)

    X_psd = np.array(X_psd)  # Kształt: (n_epochs, 48)

    # --- 2. Ekstrakcja Cech Czasowych (ERP / P300) ---

    # Pobieramy surowe dane fali z epok
    # Kształt: (n_epochs, n_channels, n_times)
    erp_data = epochs_data.get_data(picks='eeg')

    # Znajdujemy indeksy próbek odpowiadające naszemu oknu czasu (np. 0.4s do 0.7s)
    tmin_index, tmax_index = epochs_data.time_as_index([erp_tmin, erp_tmax], use_rounding=True)

    # Wycina_my z danych tylko to interesujące nas okno
    # Kształt: (n_epochs, n_channels, n_samples_in_window)
    erp_window = erp_data[:, :, tmin_index:tmax_index]

    # Obliczamy średnią amplitudę w tym oknie dla każdego kanału i każdej epoki
    # axis=2 oznacza uśrednianie wzdłuż osi czasu
    X_erp = np.mean(erp_window, axis=2)  # Kształt: (n_epochs, 16)

    # --- 3. Połączenie Cech ---

    # Łączymy nasze cechy PSD (48) z cechami ERP (16) w jedną macierz
    # axis=1 oznacza łączenie "obok siebie" (w kolumnach)
    X_hybrid = np.concatenate((X_psd, X_erp), axis=1)  # Kształt: (n_epochs, 64)

    # Pobieramy etykiety
    y = epochs_data.events[:, -1]

    print(f"Hybrid Feature matrix X created with shape: {X_hybrid.shape}")

    return X_hybrid, y