import pandas as pd
from pathlib import Path

# This is the path to the dataset processed by Clinica
# The TSV file contains the HC volumes as well as some additional data
# This script extracts a subset of the columns and does some renaming to
# have a simpler dataset to work on.
input_data = Path("/Users/nicolas.gensollen/merge_simplified.tsv")

df = pd.read_csv(input_data, sep="\t")

df1 = df.iloc[:int(len(df) / 2)]
df2 = df.iloc[:]

columns_to_keep = [
    "participant_id",
    "t1-volume_group-ADCN_atlas-Neuromorphometrics_ROI-Left-Hippocampus_intensity",
    "t1-volume_group-ADCN_atlas-Neuromorphometrics_ROI-Right-Hippocampus_intensity",
    "diagnosis",
]

df1 = df1[columns_to_keep]
df2 = df2[columns_to_keep]

renaming = {
    "participant_id": "subject_id",
    "t1-volume_group-ADCN_atlas-Neuromorphometrics_ROI-Left-Hippocampus_intensity": "HC_left_volume",
    "t1-volume_group-ADCN_atlas-Neuromorphometrics_ROI-Right-Hippocampus_intensity": "HC_right_volume",
}

df1.rename(columns=renaming, inplace=True)
df2.rename(columns=renaming, inplace=True)

df1.to_csv(input_data.with_name("dataset1.tsv"), sep="\t", index=False)
df2.to_csv(input_data.with_name("dataset2.tsv"), sep="\t", index=False)
