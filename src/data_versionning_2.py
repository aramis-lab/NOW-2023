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
# ## Chapter 2 : Versionning large datasets with DVC
#
# In the first chapter of the second part of this tutorial, we saw how to install and use DVC to version code and data in a simple way.
#
# However, our input dataset was extremely simple and could have been versionned with GIT without difficulty. 
#
# The main objective of this second chapter is to show the true power of DVC. We will keep our objective of predicting AD vs Control subjects, but instead of assuming a simple preprocessed dataset with hypocampus volumes already extracted, we will work directly with brain images and train a deep learning model to perform this task.
#
# The second objective of this chapter is to show some more advanced features of DVC like experiment managment.
#
# Let's dive in !
#
# ### Setup the repo
#
# Since this is a new notebook meant to be independant from the notebook of the first chapter, we will start a new research project from scratch.
#
# Start by configuring the GIT and DVC repo:

# %%
# Dirty for now...
# ! pip install git+https://github.com/aramis-lab/NOW-2023-lib.git/

# %%
# ! pip install dvc
# ! git init
# ! dvc init

# %% [markdown]
# As with the previous notebook, you might need to configure your username and email:

# %%
# ! git config --local user.email "john.doe@inria.fr"
# ! git config --local user.name "John Doe"

# %%
# ! git commit -m "initialize DVC"

# %% [markdown]
# ### Database
#
# In this session we use the images from a public research project: OASIS-1. Two labels exist in this dataset:
#
# - CN (Cognitively Normal) for healthy participants.
# - AD (Alzheimer’s Disease) for patients affected by Alzheimer’s disease.
#
# The original images were preprocessed using Clinica: a software platform for clinical neuroimaging studies.
#
# Preprocessed images and other files are distributed in a tarball, if you haven't downloaded the images before, run the following commands to download and extract them:

# %%
# Only run if necessary !
#
# # ! wget --no-check-certificate --show-progress https://aramislab.paris.inria.fr/files/data/databases/DL4MI/OASIS-1-dataset_pt_new.tar.gz

# %% [markdown]
# Once downloaded, you can take a look at it:

# %%
from pathlib import Path

oasis_folder = Path("/Users/nicolas.gensollen/NOW_2023/OASIS-1_dataset/")

# %%
import pandas as pd

columns_to_use = [
    "participant_id",
    "session_id",
    "alternative_id_1",
    "sex",
    "education_level",
    "age_bl",
    "diagnosis_bl",
    "laterality",
    "MMS",
    "cdr_global",
    "diagnosis",
]
OASIS_df = pd.read_csv(
    oasis_folder / "tsv_files" / "lab_1" / "OASIS_BIDS.tsv",
    sep="\t",
    usecols=columns_to_use,
)
print(OASIS_df.head())

_ = OASIS_df.hist(figsize=(16, 8))

# %%
from now_2023.utils import characteristics_table

population_df = characteristics_table(OASIS_df, OASIS_df)
population_df

# %% [markdown]
# ### Preprocessing
#
# Theoretically, the main advantage of deep learning methods is to be able to work without extensive data preprocessing. However, as we have only a few images to train the network in this lab session, the preprocessing here is very extensive. More specifically, the images encountered:
#
# - Non-linear registration.
# - Segmentation of grey matter.
# - Conversion to tensor format (.pt).
#
# As mentioned above, to obtain the preprocessed images, we used some pipelines provided by Clinica and ClinicaDL in order to:
#
# - Convert the original dataset to BIDS format ([clinica convert oasis-2-bids](https://aramislab.paris.inria.fr/docs/public/latest/Converters/OASIS2BIDS/)).
# - Get the non-linear registration and segmentation of grey mater (pipeline [t1-volume](https://aramislab.paris.inria.fr/docs/public/latest/Pipelines/T1_Volume/)).
# - Obtain the preprocessed images in tensor format ([tensor extraction using ClinicaDL, clinicadl extract](https://clinicadl.readthedocs.io/en/stable/Preprocessing/Extract/)).
# - The preprocessed images are store in the [CAPS folder structure](http://www.clinica.run/doc/CAPS/Introduction/) and all have the same size (121x145x121).
#
# To facilitate the training and avoid overfitting due to the limited amount of data, the model won’t use the full image but only a part of the image (size 30x40x30) centered on a specific neuroanatomical region: the hippocampus (HC). This structure is known to be linked to memory, and is atrophied in the majority of cases of Alzheimer’s disease patients.
#
# Before going further, let's take a look at the images we have downloaded and let's compute a cropped image of the left HC for a randomly selected subject:

# %%
import torch
from now_2023.plotting import plot_image, plot_tensor
from now_2023.utils import CropLeftHC

subject = 'sub-OASIS10003'
image_folder = (
    oasis_folder /
    "CAPS" /
    "subjects" /
    subject /
    "ses-M00" /
    "deeplearning_prepare_data" /
    "image_based" /
    "custom"
)
image_filename = f"{subject}_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.pt"
preprocessed_pt = torch.load(image_folder / image_filename)

plot_image(
    oasis_folder / "raw" / f"{subject}_ses-M00_T1w.nii.gz",
    cut_coords=(78, 122, 173),
    title=f'Slices of raw image of subject {subject}',
)
plot_tensor(
    preprocessed_pt,
    cut_coords=(60, 72, 60),
    title=f'Center slices of preprocessed image of subject {subject}',
)
plot_tensor(
    CropLeftHC()(preprocessed_pt),
    cut_coords=(15, 20, 15),
    title=f'Center slices of left HC of subject {subject}',
)

# %% [markdown]
# ## Use only the left HC
#
# We are going to generate a new dataset consisting only of images of the left hippocampus (the last image above). 
#
# For simplicity, this is the dataset that we will consider as our input dataset, and thus the dataset we will version. In other word, we will assume that we didn't computed the cropped images from the raw dataset downloaded above, but versioning the raw dataset would work in the same way.
#
# Note also that to improve the training and reduce overfitting, we can add a random shift to the cropping function. This means that the bounding box around the hippocampus may be shifted by a limited amount of voxels in each of the three directions.
#
# Let's generate our dataset:

# %%
from now_2023.data_generation import generate_cropped_hc_dataset

data_folder = Path("./data")

generate_cropped_hc_dataset(
    oasis_folder,
    hemi="left",
    output_folder=data_folder,
    verbose=False,
)

# %% [markdown]
# This should have created a new `data` folder in the current working space, which should have the following structure:

# %%
# ! tree data | head -n 14

# %% [markdown]
# As you can see, we have one tensor image for each subject, representing the extracted left hippocampus.
#
# Let's take a look at some of these images:

# %%
from now_2023.plotting import plot_hc

plot_hc(data_folder, 'sub-OASIS10001', "left", cut_coords=(15, 20, 15))
plot_hc(data_folder, 'sub-OASIS10003', "left", cut_coords=(10, 30, 25))

# %% [markdown]
# ### Cross-validation
#
# In order to choose hyperparameters the set of images is divided into a training set (80%) and a validation set (20%). The data split was performed in order to ensure a similar distribution of diagnosis, age and sex between the subjects of the training set and the subjects of the validation set. Moreover the MMS distribution of each class is preserved.

# %%
train_df = pd.read_csv(oasis_folder / "tsv_files" / "lab_1" / "train.tsv", sep="\t")
valid_df = pd.read_csv(oasis_folder / "tsv_files" / "lab_1" / "validation.tsv", sep="\t")
train_df["hemi"] = "left"
valid_df["hemi"] = "left"

train_population_df = characteristics_table(train_df, OASIS_df)
valid_population_df = characteristics_table(valid_df, OASIS_df)

print(f"Train dataset:\n {train_population_df}\n")
print(f"Validation dataset:\n {valid_population_df}")

# %% [markdown]
# ### Model
#
# We propose here to design a convolutional neural network that takes for input a patch centered on the left hippocampus of size 30x40x30.
#
# The architecture of the network was found using a Random Search on architecture + optimization hyperparameters.

# %%
import json
from now_2023.models import CNNModel

model = CNNModel(learning_rate=10**-4, n_epochs=30, batch_size=4)
model.fit(data_folder, train_df)

results_training_left, metrics_training_left = model.predict(data_folder, train_df)
results_validation_left, metrics_validation_left = model.predict(data_folder, valid_df)

print(f"Metrics on training set :\n{json.dumps(metrics_training_left, indent=4)}")
print(f"Metrics on validation set :\n{json.dumps(metrics_validation_left, indent=4)}")

# %% [markdown]
# Let's save the model and validation metrics:

# %%
model.save("model.h5")
with open("metrics.json", "w") as fp:
    json.dump(metrics_validation_left, fp, indent=4)

# %% [markdown]
# We can now version both the data used for training and validation as well as the trained model:

# %%
# ! dvc add data
# ! dvc add model.h5

# %%
# ! git add model.h5.dvc .gitignore data.dvc metrics.json
# ! git commit -m "First model, trained with images cropped around left HC"
# ! git tag -a "v1.0" -m "model v1.0, left HC only"

# %%
# ! git log

# %% [markdown]
# ## Use both the left and right HC
#
# Let's imagine now that we receive additional data in the form of cropped images of the right HC.
#
# Of course we are only pretending here, recall that we can easily generate these images from the downloaded dataset:

# %%
generate_cropped_hc_dataset(oasis_folder, hemi="right", output_folder=Path("./data"), verbose=False)

# %%
# ! tree data | head -n 16

# %% [markdown]
# As you can see, we now have two images for each subject, one for the left HC that we already had in the previous section, and one for the right HC that we just generated.

# %%
plot_hc(data_folder, 'sub-OASIS10001', "left", cut_coords=(15, 20, 15))
plot_hc(data_folder, 'sub-OASIS10001', "right", cut_coords=(15, 20, 15))

# %% [markdown]
# We need to update our train and validation dataframes to encode this:

# %%
import numpy as np

train_df = train_df.loc[np.repeat(train_df.index, 2)].reset_index(drop=True)
train_df["hemi"][::2] = "right"
valid_df = valid_df.loc[np.repeat(valid_df.index, 2)].reset_index(drop=True)
valid_df["hemi"][::2] = "right"
train_df

# %% [markdown]
# We can now instantiate a new `CNNModel` and train it on both HC:

# %%
model = CNNModel(learning_rate=10**-4, n_epochs=30, batch_size=4)
model.fit(data_folder, train_df)

results_training_both, metrics_training_both = model.predict(data_folder, train_df)
results_validation_both, metrics_validation_both = model.predict(data_folder, valid_df)

print(f"Metrics on training set :\n{json.dumps(metrics_training_both, indent=4)}")
print(f"Metrics on validation set :\n{json.dumps(metrics_validation_both, indent=4)}")

# %%
model.save("model.h5")
with open("metrics.json", "w") as fp:
    json.dump(metrics_validation_left, fp, indent=4)

# %%
# ! dvc add data
# ! dvc add model.h5

# %%
# ! git add model.h5.dvc data.dvc metrics.json
# ! git commit -m "Second model, trained with images cropped around left and right HC"
# ! git tag -a "v2.0" -m "model v2.0, left and right HC"

# %%
# ! git log

# %% [markdown]
# Cleaning:

# %%
# Do not run this unless you want to start over from scratch...
# ! rm model.*
# ! rm -rf .git
# ! rm -rf .dvc
# ! rm .gitignore
# ! rm .dvcignore
# ! rm metrics.json
# ! rm data.dvc
# ! rm -r data

# %%
