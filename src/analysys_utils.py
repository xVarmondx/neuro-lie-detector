import os
import glob
import numpy as np
import mne
import pandas as pd
from src.preprocessing import preprocessing
from src.epoching import create_epochs
import matplotlib.pyplot as plt


def build_master_event_map(data_folder, participant_dirs):
    """
    Scans all participant files to build a single, consistent event_id map.

    Args:
        data_folder (str): Path to the main dataset folder.
        participant_dirs (list): List of participant folder names to scan.

    Returns:
        dict: A master_event_id map (e.g., {'event_name': 1, ...}).
    """
    print("--- Building master event_id map ---")
    all_descriptions = set()
    for participant_uuid in participant_dirs:
        participant_path = os.path.join(data_folder, participant_uuid)
        files = glob.glob(os.path.join(participant_path, "*_raw.fif"))
        for f in files:
            try:
                raw = mne.io.read_raw_fif(f, preload=False, verbose=False)
                descriptions = np.unique(raw.annotations.description)
                all_descriptions.update(descriptions)
            except Exception as e:
                print(f"Skipping file {os.path.basename(f)}: {e}")

    master_event_id = {desc: i + 1 for i, desc in enumerate(sorted(list(all_descriptions)))}
    print(f"Master map created with {len(master_event_id)} unique events.")
    return master_event_id


def load_epochs_by_condition(data_folder, participant_dirs, master_event_id):
    """
    Loads, processes, and segregates epochs into 'honest' and 'deceitful'.

    Args:
        data_folder (str): Path to the main dataset folder.
        participant_dirs (list): List of participant folders to process.
        master_event_id (dict): The master event_id map to use.

    Returns:
        tuple: (all_honest_epochs, all_deceitful_epochs) as lists of Epochs objects.
    """
    print("\n--- Processing data for all participants ---")
    all_honest_epochs = []
    all_deceitful_epochs = []

    for participant_uuid in participant_dirs:
        print(f"--- Processing participant: {participant_uuid} ---")
        participant_path = os.path.join(data_folder, participant_uuid)
        honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
        deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

        try:
            for h_file in honest_files:
                raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
                prep_honest = preprocessing(raw_honest)
                epochs_honest = create_epochs(prep_honest, master_event_id=master_event_id)
                if epochs_honest and len(epochs_honest) > 0:
                    all_honest_epochs.append(epochs_honest)

            for d_file in deceitful_files:
                raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
                prep_deceitful = preprocessing(raw_deceitful)
                epochs_deceitful = create_epochs(prep_deceitful, master_event_id=master_event_id)
                if epochs_deceitful and len(epochs_deceitful) > 0:
                    all_deceitful_epochs.append(epochs_deceitful)
        except Exception as e:
            print(f"Error processing {participant_uuid}: {e}")

    return all_honest_epochs, all_deceitful_epochs


def load_epochs_by_gender(data_folder, participant_dirs, master_event_id, survey_path, uuid_col, gender_col):
    """
    Loads, processes, and segregates epochs into four groups based on gender.

    Args:
        data_folder (str): Path to the dataset folder.
        participant_dirs (list): List of participant folders.
        master_event_id (dict): The master event_id map.
        survey_path (str): Path to the survey (Ankiety.xlsx) file.
        uuid_col (str): Name of the UUID column in Excel.
        gender_col (str): Name of the Gender column in Excel.

    Returns:
        tuple: (male_honest_epochs, male_deceitful_epochs, female_honest_epochs, female_deceitful_epochs)
    """
    try:
        df_survey = pd.read_excel(survey_path)
        df_survey[uuid_col] = df_survey[uuid_col].astype(str).str.lower()
        df_survey[gender_col] = df_survey[gender_col].astype(str).str.lower()
    except Exception as e:
        print(f"ERROR: Could not load or process {survey_path}: {e}")
        return [], [], [], []

    try:
        male_uuids = df_survey[df_survey[gender_col] == 'm'][uuid_col].tolist()
        female_uuids = df_survey[df_survey[gender_col] == 'k'][uuid_col].tolist()
        print(f"Found {len(male_uuids)} male and {len(female_uuids)} female participants in survey")
    except KeyError:
        print(f"ERROR: Column '{gender_col}' or '{uuid_col}' not found in survey ")
        return [], [], [], []

    print("\n--- Processing data with gender segregation ---")
    male_honest_epochs = []
    male_deceitful_epochs = []
    female_honest_epochs = []
    female_deceitful_epochs = []

    for participant_folder_name in participant_dirs:
        participant_folder_lower = participant_folder_name.lower()
        is_male = any(uuid in participant_folder_lower for uuid in male_uuids)
        is_female = any(uuid in participant_folder_lower for uuid in female_uuids)

        if not (is_male or is_female):
            continue

        participant_path = os.path.join(data_folder, participant_folder_name)
        honest_files = glob.glob(os.path.join(participant_path, "*HONEST*raw.fif"))
        deceitful_files = glob.glob(os.path.join(participant_path, "*DECEITFUL*raw.fif"))

        try:
            for h_file in honest_files:
                raw_honest = mne.io.read_raw_fif(h_file, preload=True, verbose=False)
                prep_honest = preprocessing(raw_honest)
                epochs_h = create_epochs(prep_honest, master_event_id=master_event_id)
                if epochs_h and len(epochs_h) > 0:
                    if is_male:
                        male_honest_epochs.append(epochs_h)
                    elif is_female:
                        female_honest_epochs.append(epochs_h)

            for d_file in deceitful_files:
                raw_deceitful = mne.io.read_raw_fif(d_file, preload=True, verbose=False)
                prep_deceitful = preprocessing(raw_deceitful)
                epochs_d = create_epochs(prep_deceitful, master_event_id=master_event_id)
                if epochs_d and len(epochs_d) > 0:
                    if is_male:
                        male_deceitful_epochs.append(epochs_d)
                    elif is_female:
                        female_deceitful_epochs.append(epochs_d)
        except Exception as e:
            print(f"Error processing {participant_folder_name}: {e}")

    return male_honest_epochs, male_deceitful_epochs, female_honest_epochs, female_deceitful_epochs


def plot_raw_data_inspection(raw):
    """
    Generates and displays inspection plots for a raw MNE object.

    Args:
        raw (mne.io.Raw): The raw data object to plot.
    """
    print("Generating raw signal plot...")
    raw.plot(n_channels=20, duration=10, scalings='auto',
             title="Plot 1: Raw Signal (Sample File)",
             show=False)

    print("Generating raw PSD plot...")
    raw.compute_psd(fmin=1, fmax=125, average=True, show=False)

    print("Generating events plot...")
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                            first_samp=raw.first_samp, event_id=event_id,
                            show=False)
    except Exception as e:
        print(f"Could not generate events plot: {e}")


def plot_raw_data(raw, title="Surowy sygnał EEG", duration=10, n_channels=20):
    """
    Displays the raw signal plot (raw.plot()).
    """
    print(f"Generating plot: {title}")
    fig = raw.plot(n_channels=n_channels, duration=duration, scalings='auto',
                   title=title, show=True)
    return fig

def plot_raw_psd(raw):
    """
    Displays the Power Spectral Density (PSD) plot.
    """
    print("Generating plot: Power Spectral Density (PSD)")

    fig = raw.compute_psd(fmin=1, fmax=125, average='mean').plot(show=True)
    return fig

# def plot_events(raw, title="Event"):
#   """
#   Displays the event (annotation) plot.
#   """
#   print(f"Generating plot: {title}")
#     try:
#         events, event_id = mne.events_from_annotations(raw, verbose=False)
#         fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
#                                   first_samp=raw.first_samp, event_id=event_id,
#                                   show=True)
#         return fig
#     except Exception as e:
#         print(f"Unable to generate event plot: {e}")
#         return None

def plot_events(raw, title="Rozkład zdarzeń w pliku", filter_str="PersonalDataField"):
    """
    Displays a bar chart showing the number of occurrences
    of each event (annotation), with optional filtering.

    Args:
        raw (mne.io.Raw): MNE Raw object.
        title (str): Plot title.
        filter_str (str | None): String for filtering annotations.
                                 If None, shows all.
    """
    print(f"Generating event bar chart (filter: '{filter_str}')...")
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        if filter_str:
            filtered_event_id = {key: value for key, value in event_id.items() if filter_str in key}
        else:
            filtered_event_id = event_id

        if not filtered_event_id:
            print(f"No events found matching filter: '{filter_str}'")
            return None

        id_to_name_map = {v: k for k, v in filtered_event_id.items()}
        all_event_codes = events[:, 2]
        event_names = [id_to_name_map.get(code) for code in all_event_codes]
        event_names_filtered = [name for name in event_names if name is not None]

        if not event_names_filtered:
            print(f"No occurrences found for filter: '{filter_str}'")
            return None

        counts = pd.Series(event_names_filtered).value_counts()

        fig, ax = plt.subplots(figsize=(10, len(counts) * 0.5 + 1))
        counts.sort_values().plot(kind='barh', ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Number of occurrences (Count)")
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Unable to generate event plot: {e}")
        return None


def plot_processed_data_inspection(processed_raw):
    """
    Generates and displays inspection plots for a preprocessed MNE object.

    Args:
        processed_raw (mne.io.Raw): The preprocessed (cleaned) data object.
    """
    print("Generating processed signal plot...")
    processed_raw.plot(n_channels=16, duration=10, scalings='auto',
                       title="Plot 2: Processed Signal (Filtered, EEG only)",
                       show=False)

    print("Generating processed PSD plot...")
    processed_raw.compute_psd(fmin=1, fmax=125, average=True, show=False)

def plot_erp_comparison(evokeds_dict, title="ERP Comparison"):
    """
    Generates a comparison plot (GFP) for a dictionary of evoked objects.

    Args:
        evokeds_dict (dict): A dictionary mapping labels to Evoked objects
                             (e.g., {"Honest": evoked_honest, "Deceitful": evoked_deceitful}).
        title (str): The title for the plot.
    """
    print(f"Generating plot: {title}")
    mne.viz.plot_compare_evokeds(evokeds_dict,
                                 picks='eeg',
                                 legend='upper left',
                                 title=title,
                                 show=True)


def plot_erp_difference_topomap(evoked_1, evoked_2, times=None):
    """
    Generates a topographic map of the difference between two evoked objects.

    Calculates (evoked_1 - evoked_2).

    Args:
        evoked_1 (mne.Evoked): The first evoked object (e.g., honest).
        evoked_2 (mne.Evoked): The second evoked object (e.g., deceitful).
        times (list | None): List of time points (in seconds) to plot.
                             If None, defaults to [0.1, 0.2, 0.3, 0.4, 0.5, 0.6].
    """
    print("Generating difference topomap plot...")
    if times is None:
        times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    diff_erp = mne.combine_evoked([evoked_1, evoked_2], weights=[1, -1])
    diff_erp.plot_topomap(times=times, show=True)

