# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 1 : Code versionning using GIT
#
# This is the first part of the tutorial on code versioning using git.

# %%
import torch
import numpy as np
import pandas as pd
from torch import nn
from time import time
from os import path
from torchvision import transforms
import random
from copy import deepcopy

# %%
# Load the complete dataset
OASIS_df = pd.read_csv(
    'OASIS-1_dataset/tsv_files/lab_1/OASIS_BIDS.tsv', sep='\t',
    usecols=['participant_id', 'session_id', 'alternative_id_1', 'sex',
             'education_level', 'age_bl', 'diagnosis_bl', 'laterality', 'MMS',
             'cdr_global', 'diagnosis']
)
# Show first items of the table
print(OASIS_df.head())
# First visual inspection
_ = OASIS_df.hist(figsize=(16, 8))

# %%
# Study the characteristics of the AD & CN populations (age, sex, MMS, cdr_global)
from training import characteristics_table

population_df = characteristics_table(OASIS_df, OASIS_df)
population_df

# %%
### PREPROCESSING ###
from training import MRIDataset, CropLeftHC, CropRightHC



# %%
### VISUALIZATION ###
from training import show_slices

subject = 'sub-OASIS10003'
preprocessed_pt = torch.load(f'OASIS-1_dataset/CAPS/subjects/{subject}/ses-M00/' +
                    f'deeplearning_prepare_data/image_based/custom/{subject}_ses-M00_' +
                    'T1w_segm-graymatter_space-Ixi549Space_modulated-off_' +
                    'probability.pt')
raw_nii = nib.load(f'OASIS-1_dataset/raw/{subject}_ses-M00_T1w.nii.gz')

raw_np = raw_nii.get_fdata()

# %%
slice_0 = raw_np[:, :, 78]
slice_1 = raw_np[122, :, :]
slice_2 = raw_np[:, 173, :]
show_slices([slice_0, rotate(slice_1, 90), rotate(slice_2, 90)])
plt.suptitle(f'Slices of raw image of subject {subject}')
plt.show()
# %%
slice_0 = preprocessed_pt[0, 60, :, :]
slice_1 = preprocessed_pt[0, :, 72, :]
slice_2 = preprocessed_pt[0, :, :, 60]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle(f'Center slices of preprocessed image of subject {subject}')
plt.show()

# %%
leftHC_pt = CropLeftHC()(preprocessed_pt)
slice_0 = leftHC_pt[0, 15, :, :]
slice_1 = leftHC_pt[0, :, 20, :]
slice_2 = leftHC_pt[0, :, :, 15]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle(f'Center slices of left HC of subject {subject}')
plt.show()


# %%
### CROSS VALIDATION ###

train_df = pd.read_csv('OASIS-1_dataset/tsv_files/lab_1/train.tsv', sep='\t')
valid_df = pd.read_csv('OASIS-1_dataset/tsv_files/lab_1/validation.tsv', sep='\t')

train_population_df = characteristics_table(train_df, OASIS_df)
valid_population_df = characteristics_table(valid_df, OASIS_df)

print(f"Train dataset:\n {train_population_df}\n")
print(f"Validation dataset:\n {valid_population_df}")

# %%
### MODEL ###
from torch.utils.data import DataLoader
img_dir = path.join('OASIS-1_dataset', 'CAPS')
batch_size=4

example_dataset = MRIDataset(img_dir, OASIS_df, transform=CropLeftHC())
example_dataloader = DataLoader(example_dataset, batch_size=batch_size, drop_last=True)
for data in example_dataloader:
    pass

print(f"Shape of Dataset output:\n {example_dataset[0]['image'].shape}\n")

print(f"Shape of DataLoader output:\n {data['image'].shape}")


# %%
### TRAIN WITH LEFT HC ###

from training import CustomNetwork, train, test
img_dir = path.join('/Users/camille.brianceau/Downloads/OASIS-1_dataset', 'CAPS')
transform = CropLeftHC(2)

train_datasetLeftHC = MRIDataset(img_dir, train_df, transform=transform)
valid_datasetLeftHC = MRIDataset(img_dir, valid_df, transform=transform)

# Try different learning rates
learning_rate = 10**-4
n_epochs = 30
batch_size = 4

# Put the network on GPU
modelLeftHC = CustomNetwork() #.cuda()
train_loaderLeftHC = DataLoader(train_datasetLeftHC, batch_size=batch_size, shuffle=True, pin_memory=True)
# A high batch size improves test speed
valid_loaderLeftHC = DataLoader(valid_datasetLeftHC, batch_size=32, shuffle=False, pin_memory=True)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(modelLeftHC.parameters(), learning_rate)

best_modelLeftHC = train(modelLeftHC, train_loaderLeftHC, criterion, optimizer, n_epochs)

valid_resultsLeftHC_df, valid_metricsLeftHC = test(best_modelLeftHC, valid_loaderLeftHC, criterion)
train_resultsLeftHC_df, train_metricsLeftHC = test(best_modelLeftHC, train_loaderLeftHC, criterion)
print(valid_metricsLeftHC)
print(train_metricsLeftHC)


valid_resultsLeftHC_df = valid_resultsLeftHC_df.merge(OASIS_df, how='left', on='participant_id', sort=False)
valid_resultsLeftHC_old_df = valid_resultsLeftHC_df[(valid_resultsLeftHC_df.age_bl >= 62)]
#compute_metrics(valid_resultsLeftHC_old_df.true_label, valid_resultsLeftHC_old_df.predicted_label)


# %%
### TRAIN WITH RIGHT HC ###
img_dir = path.join('/Users/camille.brianceau/Downloads/OASIS-1_dataset', 'CAPS')
transform = CropRightHC(2)

train_datasetRightHC = MRIDataset(img_dir, train_df, transform=transform)
valid_datasetRightHC = MRIDataset(img_dir, valid_df, transform=transform)

learning_rate = 10**-4
n_epochs = 30
batch_size = 4

# Put the network on GPU
modelRightHC = CustomNetwork() #.cuda()
train_loaderRightHC = DataLoader(train_datasetRightHC, batch_size=batch_size, shuffle=True,  pin_memory=True)
valid_loaderRightHC = DataLoader(valid_datasetRightHC, batch_size=32, shuffle=False,  pin_memory=True)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(modelRightHC.parameters(), learning_rate)

best_modelRightHC = train(modelRightHC, train_loaderRightHC, criterion, optimizer, n_epochs)

valid_resultsRightHC_df, valid_metricsRightHC = test(best_modelRightHC, valid_loaderRightHC, criterion)
train_resultsRightHC_df, train_metricsRightHC = test(best_modelRightHC, train_loaderRightHC, criterion)
print(valid_metricsRightHC)
print(train_metricsRightHC)

# %%
from training import compute_metrics
### SOFT VOTING ###

def softvoting(leftHC_df, rightHC_df):
    df1 = leftHC_df.set_index('participant_id', drop=True)
    df2 = rightHC_df.set_index('participant_id', drop=True)
    results_df = pd.DataFrame(index=df1.index.values,
                              columns=['true_label', 'predicted_label',
                                       'proba0', 'proba1'])
    results_df.true_label = df1.true_label
    # Compute predicted label and probabilities
    results_df.proba1 = 0.5 * df1.proba1 + 0.5 * df2.proba1
    results_df.proba0 = 0.5 * df1.proba0 + 0.5 * df2.proba0
    results_df.predicted_label = (0.5 * df1.proba1 + 0.5 * df2.proba1 > 0.5).astype(int)

    return results_df

valid_results = softvoting(valid_resultsLeftHC_df, valid_resultsRightHC_df)
valid_metrics = compute_metrics(valid_results.true_label, valid_results.predicted_label)
print(valid_metrics)


# %%
### TRAIN AUTOENCODER ###
from training import trainAE, testAE, AutoEncoder
learning_rate = 10**-3
n_epochs = 30
batch_size = 4

AELeftHC = AutoEncoder()#.cuda()
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(AELeftHC.parameters(), learning_rate)

best_AELeftHC = trainAE(AELeftHC, train_loaderLeftHC, criterion, optimizer, n_epochs)

# %%
### VISUALIZATION ###

import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import rotate

subject = 'sub-OASIS10003'
preprocessed_pt = torch.load(f'/Users/camille.brianceau/Downloads/OASIS-1_dataset/CAPS/subjects/{subject}/ses-M00/' +
                    'deeplearning_prepare_data/image_based/custom/' + subject +
                    '_ses-M00_'+
                    'T1w_segm-graymatter_space-Ixi549Space_modulated-off_' +
                    'probability.pt')
input_pt = CropLeftHC()(preprocessed_pt).unsqueeze(0)#.cuda()
_, output_pt = best_AELeftHC(input_pt)

# %%
slice_0 = input_pt[0, 0, 15, :, :].cpu()
slice_1 = input_pt[0, 0, :, 20, :].cpu()
slice_2 = input_pt[0, 0, :, :, 15].cpu()
show_slices([slice_0, slice_1, slice_2])
plt.suptitle(f'Center slices of the input image of subject {subject}')
plt.savefig("/Users/camille.brianceau/aramis/NOW-2023/figures/1.png")

# %%
slice_0 = output_pt[0, 0, 15, :, :].cpu().detach()
slice_1 = output_pt[0, 0, :, 20, :].cpu().detach()
slice_2 = output_pt[0, 0, :, :, 15].cpu().detach()
show_slices([slice_0, slice_1, slice_2])
plt.suptitle(f'Center slices of the output image of subject {subject}')
plt.savefig("/Users/camille.brianceau/aramis/NOW-2023/figures/2.png")


# %%
### CLUSTERING ###

from training import compute_dataset_features
# train_codes, train_labels, names = compute_dataset_features(train_loaderBothHC, best_AEBothHC)
train_codes, train_labels, names = compute_dataset_features(train_loaderLeftHC, best_AELeftHC)

from sklearn import mixture
from sklearn.metrics import adjusted_rand_score

n_components = 2
model = mixture.GaussianMixture(n_components)
model.fit(train_codes)
train_predict = model.predict(train_codes)

metrics = compute_metrics(train_labels, train_predict)
ari = adjusted_rand_score(train_labels, train_predict)
print(f"Adjusted random index: {ari}")

data_np = np.concatenate([names, train_codes,
                          train_labels[:, np.newaxis],
                          train_predict[:, np.newaxis]], axis=1)
columns = ['feature %i' % i for i in range(train_codes.shape[1])]
columns = ['participant_id'] + columns + ['true_label', 'predicted_label']
data_df = pd.DataFrame(data_np, columns=columns).set_index('participant_id')

merged_df = data_df.merge(OASIS_df.set_index('participant_id'), how='inner', on='participant_id')

plt.title('Clustering values according to age and MMS score')
for component in range(n_components):
    predict_df = merged_df[merged_df.predicted_label == str(component)]
    plt.plot(predict_df['age_bl'], predict_df['MMS'], 'o', label=f"cluster {component}")
plt.legend()
plt.xlabel('age')
plt.ylabel('MMS')
plt.savefig("/Users/camille.brianceau/aramis/NOW-2023/figures/6.png")

