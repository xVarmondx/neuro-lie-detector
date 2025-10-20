import os
import pandas as pd
import mne
import numpy as np
from matplotlib import pyplot as plt

DATA_FOLDER = "dataset"
SURVEY_FILE = os.path.join(DATA_FOLDER, "Ankiety.xlsx")

# Analisisty one of EEG file

EXAMPLE_USER = "2D663E30"
EXAMPLE_EEG_FILENAME = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
EXAMPLE_EEG_PATH = os.path.join(DATA_FOLDER, EXAMPLE_USER ,EXAMPLE_EEG_FILENAME)

# Opening survey
df_survey = pd.read_excel(SURVEY_FILE)
print(f"Number of participants: {len(df_survey)}") #number of rows
print(f"Columns names: {df_survey.columns.tolist()}")
#print(f"Rows: {df_survey.rows.tolist()}")
print(df_survey.head())

# Open sample EEG
raw = mne.io.read_raw_fif(EXAMPLE_EEG_PATH, preload=True, verbose=False)

print(f"\n------------------------Basic data information------------------------")
print(raw.info)

print(f"Sampling frequency: {raw.info["sfreq"]} Hz")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Channels: {raw.ch_names}")
print(f"Recording duration: {raw.times[-1]:.2f} seconds")

print("\n------------------------Annotations (Events)------------------------")
if raw.annotations:
    print(f"Number of annotations: {len(raw.annotations)}")
    print("Example annotations:")

    for i, ann in enumerate(raw.annotations):
        if i < 5: # Show only the first 5 annotations
            print(f"Start: {ann['onset']:.2f} s, Duration: {ann['duration']:.2f} s, Description: {ann['description']}")

    # You can show only unique annotations:
    print(f"UUnique annotation types: {np.unique(raw.annotations.description)}")
else:
    print("No annotations in this file")


# Raw EEG Signal Visualizatio
print("\n------------------------Raw EEG Signal Visualization------------------------")

# Raw plot
raw.plot(n_channels=30,  # Show 30 channels
         duration=10,    # 10 seconds on one screen
         scalings='auto', # Automatic amplitude scaling
         title=f"Raw EEG Signal: {os.path.basename(EXAMPLE_EEG_FILENAME)}",
         show=False)

print(raw.annotations)

# Plot Power Spectral Density
mne.viz.plot_raw_psd(raw, fmin=0.1, fmax=125, average=True, verbose=False)

# Plot events
events, event_id = mne.events_from_annotations(raw, verbose=False)
mne.viz.plot_events(mne.events_from_annotations(raw)[0], sfreq=raw.info['sfreq'],
                    first_samp=raw.first_samp, event_id=mne.events_from_annotations(raw)[1], verbose=False)

plt.show() # Raw plot