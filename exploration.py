import os
import pandas as pd
import mne

DATA_FOLDER = "dataset"
SURVEY_FILE = os.path.join(DATA_FOLDER, "Ankiety.xlsx")

#Analisisty one of EEG file

EXAMPLE_USER = "2D663E30"
EXAMPLE_EEG_FILENAME = "EEG_ExperimentBlock.HONEST_RESPONSE_TO_TRUE_IDENTITY_raw.fif"
EXAMPLE_EEG_PATH = os.path.join(DATA_FOLDER, EXAMPLE_USER ,EXAMPLE_EEG_FILENAME)

#Opening survey
try:
    df_survey = pd.read_excel(SURVEY_FILE)
    print(f"Number of participants: {len(df_survey)}") #number of rows
    print(f"Columns names: {df_survey.columns.tolist()}")
    #print(f"Rows: {df_survey.rows.tolist()}")
    print(df_survey.head())
except:
    print("Error")

print()

#Open sample EEG
try:
    raw = mne.io.read_raw_fif(EXAMPLE_EEG_PATH, preload=False, verbose="WARNING")
    print(raw.info)
    print(raw.annotations)
except:
    print("Error")