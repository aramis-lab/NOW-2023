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
# The main objective of this second chapter is to show the true power of DVC.
#
# We will keep our objective of predicting AD vs Control subjects, but instead of assuming a simple preprocessed dataset with hypocampus volumes already extracted, we will work directly with brain images and train a deep learning model to perform this task.
#
# This chapter also aims at showing some more advanced features of DVC like experiment managment. We will define our experiment as a dependency graph using DVC commands and see how easily we can reproduce past experiment with this infrastructure.
#
# Let's dive in !
#
# ### Setup the repo
#
# Since this is a new notebook meant to be independant from the notebook of the first chapter, we will start a new project from scratch.
#
# Start by configuring the GIT and DVC repo:

# %%
# Dirty for now...
# ! pip install git+https://github.com/aramis-lab/NOW-2023-lib.git/

# %%
# If you are running on collab or if you don't have tree installed:
# # ! apt-get install tree

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
# Throughout this tutorial, we will write things to files. Usually, you would do this using your favorite IDE. However, we need this tutorial to be runnable as a notebook (for example for the attendees running it on Collab). Because of this, we will be using IPython magic commands `writefile` and `run` in order to write to a file the content of the cell, and run a given python script. If you are following locally, you can of course edit the different files directly !
#
# Add some patterns to the `.gitignore` file in order to not display IPython notebooks related files when doing a `git status`:

# %%
# %%writefile -a .gitignore

*.ipynb
__pycache__
.DS_Store
.ipynb_checkpoints

# %% [markdown]
# ### Dataset
#
# In this session we use the images from a public research project: OASIS-1. Two labels exist in this dataset:
#
# - CN (Cognitively Normal) for healthy participants.
# - AD (Alzheimer’s Disease) for patients affected by Alzheimer’s disease.
#
# The original images were preprocessed using [Clinica](https://www.clinica.run): a software platform for clinical neuroimaging studies.
#
# Preprocessed images and other files are distributed in a tarball, if you haven't downloaded the images before, run the following commands to download and extract them:

# %%
# Only run if necessary !
#
# # ! wget --no-check-certificate --show-progress https://aramislab.paris.inria.fr/files/data/databases/DL4MI/OASIS-1-dataset_pt_new.tar.gz
# # ! tar -xzf OASIS-1-dataset_pt_new.tar.gz

# %%
# %%writefile -a run_experiment.py 

from pathlib import Path

# oasis_folder = Path("./OASIS-1_dataset/")
oasis_folder = Path("/Users/nicolas.gensollen/NOW_2023/OASIS-1_dataset/")

# %%
# %run run_experiment.py

# %% [markdown]
# Once downloaded, you can take a look at it:

# %%
# %%writefile -a prepare_train_validation_sets.py

import pandas as pd
from pathlib import Path

# oasis_folder = Path("./OASIS-1_dataset/")
oasis_folder = Path("/Users/nicolas.gensollen/NOW_2023/OASIS-1_dataset/")

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

# %%
# %run prepare_train_validation_sets.py

# %%
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
# %%writefile -a run_experiment.py

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

# %%
# %run run_experiment.py

# %%
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
# %%writefile -a run_experiment.py

from now_2023.data_generation import generate_cropped_hc_dataset

data_folder = Path("./data")

# %%
# %run run_experiment.py

# %%
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
# %%writefile -a prepare_train_validation_sets.py

from now_2023.utils import characteristics_table

train_df = pd.read_csv(oasis_folder / "tsv_files" / "lab_1" / "train.tsv", sep="\t")
valid_df = pd.read_csv(oasis_folder / "tsv_files" / "lab_1" / "validation.tsv", sep="\t")

train_df["hemi"] = "left"
valid_df["hemi"] = "left"

train_population_df = characteristics_table(train_df, OASIS_df)
valid_population_df = characteristics_table(valid_df, OASIS_df)

print("*" * 50)
print(f"Train dataset:\n {train_population_df}\n")
print("*" * 50)
print(f"Validation dataset:\n {valid_population_df}")
print("*" * 50)

train_df.to_csv("train.csv")
valid_df.to_csv("validation.csv")

# %%
# ! dvc stage add -n prepare_train_validation -d prepare_train_validation_sets.py -d data \
#   -o train.csv -o validation.csv python prepare_train_validation_sets.py
# ! dvc repro

# %%
# ! dvc add data
# ! git add dvc.yaml .gitignore dvc.lock data.dvc

# %%
# ! git status

# %% [markdown]
# ### Model
#
# We propose here to design a convolutional neural network that takes for input a patch centered on the left hippocampus of size 30x40x30.
#
# The architecture of the network was found using a Random Search on architecture + optimization hyperparameters.
#
# Let's save the model and validation metrics:
#
# Our model requires some hyper-parameters. Since the values of these hyper-parameters impact the results of our experiment, we will define them in a specific `params.yaml` file: 

# %%
# %%writefile -a params.yaml

train:
    learning_rate: 0.0001
    n_epochs: 30
    batch_size: 4

# %% [markdown]
# You can either define them all at the root, or organize them in a dict-like fashion. Here we have put all parameters inside a `train` section, but we could have defined them all at the root as well.

# %%
# %%writefile -a run_experiment.py

import dvc.api
import json
import pandas as pd
from now_2023.models import CNNModel

train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("validation.csv")

params = dvc.api.params_show()
learning_rate = params['train']['learning_rate']
n_epochs = params['train']['n_epochs']
batch_size = params['train']['batch_size']

print("*" * 50)
model = CNNModel(
    learning_rate=learning_rate,
    n_epochs=n_epochs,
    batch_size=batch_size,
)
print(f"Fitting model on {len(train_df)} samples...")
model.fit(data_folder, train_df)

results_training_left, metrics_training_left = model.predict(data_folder, train_df)
results_validation_left, metrics_validation_left = model.predict(data_folder, valid_df)

print("*" * 50)
print(f"Metrics on training set :\n{json.dumps(metrics_training_left, indent=4)}")
print(f"Metrics on validation set :\n{json.dumps(metrics_validation_left, indent=4)}")
print("*" * 50)

model.save("model.h5")

with open("metrics.json", "w") as fp:
    json.dump(metrics_validation_left, fp, indent=4)

with open("notes.txt", "w") as fp:
    fp.write(
        f"CNN model fitted on {len(train_df)} samples with hyperparameters:\n"
        f"- learning rate = {learning_rate}\n"
        f"- number of epochs = {n_epochs}\n"
        f"- size of batch = {batch_size}\n"
    )

# %% [markdown]
# The code above implements the core of our experiment and does a few things:
#
# - Load the train and validation dataframes from disk. This is required because we are passing these dataframes between the first and second stage of our experiment.
# - Retrieve the hyper-parameters we defined in the `params.yaml` file. We use the DVC Python API to do so.
# - Instantiate a model with the obtained hyper-parameters.
# - Train the model on the input data.
# - Compute some metrics like accuracy, loss and so on...
# - Save the model's weights to disk.
# - Save the metrics as JSON files.
# - Write a small human-readable note describing what we did. This could be useful, for example, to your future self revisiting this experiment in a few months or years...
#
# Now, we need to define a new stage to our experiment using the `dvc stage` command. We will call this stage `run_experiment`. This stage requires three parameters (defined in the `params.yaml` file), and has fours dependencies (the `run_experiment.py` file which is the code implementing what needs to be done in this stage, the `data` folder which is our input dataset, and the two csv files `train.csv` and `validation.csv` which define the training and validation sets.
#
# In addition, it generates two outputs (the model's weights after training, and our small text file of notes), and one metric file.
#
# Finally, we need to give the command that should be run for executing this stage. In our case, it is simply `python run_experiment.py`:

# %%
# ! dvc stage add -n run_experiment \
#   -p train.learning_rate,train.n_epochs,train.batch_size   \
#   -d run_experiment.py -d data -d train.csv -d validation.csv \
#   -o model.h5 -o notes.txt -M metrics.json python run_experiment.py

# %%
# ! dvc repro

# %% [markdown]
# We can now version both the data used for training and validation as well as the trained model:

# %%
# ! git add .gitignore dvc.yaml dvc.lock run_experiment.py prepare_train_validation_sets.py

# %%
# ! git status

# %%
# ! git commit -m "First model, trained with images cropped around left HC"
# ! git tag -a "v1.0" -m "model v1.0, left HC only"
# ! git log

# %% [markdown]
# DVC also provides some utilities to visualize your experiments as DAGs :

# %%
# ! dvc dag

# %% [markdown]
# We have two stages: `prepare_train_validation` and `run_experiment`. Both depend on the input data, and the second depends on the first through the `train.csv` and `validation.csv` output files.
#
# Note that, if we re-run `dvc repro`, nothing will happen since everything is up-to-date:

# %%
# ! dvc repro

# %% [markdown]
# ## Use both the left and right HC
#
# Let's imagine now that we receive additional data in the form of cropped images of the right HC.
#
# Of course we are only pretending here, recall that we can easily generate these images from the downloaded dataset:

# %%
# Generate the cropped images of the right HC
generate_cropped_hc_dataset(
    oasis_folder,
    hemi="right",
    output_folder=Path("./data"),
    verbose=False,
)

# %%
# ! tree data | head -n 16

# %% [markdown]
# As you can see, we now have two images for each subject, one for the left HC that we already had in the previous section, and one for the right HC that we just generated.
#
# We can take a look at these images. For example, plot both HC for a specific subject:

# %%
plot_hc(data_folder, 'sub-OASIS10001', "left", cut_coords=(15, 20, 15))
plot_hc(data_folder, 'sub-OASIS10001', "right", cut_coords=(15, 20, 15))

# %% [markdown]
# At this point, we want to re-train our model with this enlarged input dataset.
#
# Given our setup, the only thing we need to do is to update our train and validation dataframes to encode the fact that we now have two samples per subject:

# %%
# %%writefile -a prepare_train_validation_sets.py

import numpy as np

train_df = train_df.loc[np.repeat(train_df.index, 2)].reset_index(drop=True)
train_df["hemi"][::2] = "right"
valid_df = valid_df.loc[np.repeat(valid_df.index, 2)].reset_index(drop=True)
valid_df["hemi"][::2] = "right"

train_df.to_csv("train.csv")
valid_df.to_csv("validation.csv")

# %% [markdown]
# Note that, for simplificity due to the constraints of the notebook format, we just appenned the previous code to the `prepare_train_validation_sets.py` file instead of modifying it, which would be much cleaner. Feel free to open the python file and replace the relevant portion of the code with the previous cell.
#
# And... that's all we need to do. We can call `dvc repro` to run the full experiment with the new dataset:

# %%
# ! dvc repro

# %%
# !git status

# %%
# ! git add data.dvc dvc.lock prepare_train_validation_sets.py

# %%
# ! git commit -m "Second model, trained with images cropped around left and right HC"
# ! git tag -a "v2.0" -m "model v2.0, left and right HC"

# %%
# ! git log

# %% [markdown]
# ## Reproducing
#
# Similarly as in chapter 1, imagine that we have to go back to the first version of our experiment. As we saw in chapter 1, we can restore the state of the project with the `checkout` commands:

# %%
# ! git checkout v1.0
# ! dvc checkout

# %% [markdown]
# Now, we are going to see the benefits of having defined our experiment with DVC. If we check the status, DVC tells us that there is a change in the `run_experiment` stage:

# %%
# ! dvc status

# %% [markdown]
# And, if we run `dvc repro`, DVC is clever enough to know which stages to run given the dependencies that were modified. In addition, since we didn't modify anything besides going back to version 1 of the experiment, DVC does not re-run the `run_experiment` stage, it uses its cache to instantaneously restore the expeiments results:

# %%
# ! dvc repro

# %% [markdown]
# We can verify that our human-readable notes correspond to the experiment we are currently running (300 samples with learning rate of 0.0001):

# %%
# ! cat notes.txt

# %% [markdown]
# Now, let's imagine we want to experiment the first version of our model with a different learning rate. All we need to do is to open the `params.yaml` file and update the parameter to the desired value.
#
# Let's try increasing it to `10^3`:

# %%
# %%writefile params.yaml

train:
    learning_rate: 0.001
    n_epochs: 30
    batch_size: 4

# %%
# ! dvc repro

# %% [markdown]
# In this case, DVC knows it has to re-run the `run_experiment` stage because there is no results cached which match the new set of dependencies.
#
# We can again look at the notes to see that the learning rate has been changed to the new desired value:

# %%
# !cat notes.txt

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
# ! rm run_experiment.py prepare_train_validation_sets.py
# ! rm dvc.*
# ! rm train.csv validation.csv
# ! rm notes.txt
# ! rm params.yaml

# %%
