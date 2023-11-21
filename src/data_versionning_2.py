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
# # Chapter 3 : Versioning large datasets with DVC
#
# In the previous chapter, we saw how to install and use [DVC](https://dvc.org) to version code and data in a simple way. However, our input dataset was extremely simple and could have been versionned with [Git](https://git-scm.com) without difficulty.
#
# The main objective of this second chapter is to show the true power of [DVC](https://dvc.org).
#
# We will keep our objective of predicting [Alzheimer's disease](https://en.wikipedia.org/wiki/Alzheimer%27s_disease) (AD) vs Control subjects, but instead of assuming a simple preprocessed dataset with the [hippocampus](https://en.wikipedia.org/wiki/Hippocampus) volumes already computed, we will work directly with brain images and train a deep learning model to perform this task.
#
# This chapter also aims at showing some more advanced features of [DVC](https://dvc.org) like [data pipelines](https://dvc.org/doc/start/data-management/data-pipelines).
#
# We will define our experiment as a pipeline using [DVC](https://dvc.org) commands, and see how easily we can reproduce past experiments with this infrastructure.
#
# Let's dive in !

# %% [markdown]
# ## Install dependencies
#
# `````{admonition} Virtual environments
# :class: tip
#
# We strongly encourage you to create a virtual environment specifically for this tutorial.
# If you haven't done it already and if you are using conda, you can do the following:
#
# ```bash
# $ conda create --name now python=3.10
# $ conda activate now
# ```
#
# `````
#
# In order to focus on the code and data management side of things, we abstracted as much code as possible. To do so, we created a very small Python library called [now_2023](https://now-2023.readthedocs.io/en/latest/index.html) that we will use in order to plot brain images, train deep learning models, and save results.
#
# If you are running this notebook on Collab, then you need to install it:

# %%
# # ! pip install now-2023

# %%
# If you are running on collab or if you don't have tree installed:
# # ! apt-get install tree

# %%
# ! pip install dvc

# %% [markdown]
# ## Setup the repo
#
# Since this is a new notebook meant to be independant from the notebooks of the first chapters, we will start a new project from scratch.
#
# Start by configuring the [Git](https://git-scm.com) and [DVC](https://dvc.org) repo.
#
# ```{warning}
# If you are running this notebook on Collab, or if you are using an old version of Git, you need to run the following cell which will make sure your default branch is nammed `main` and not `master` as this default was changed a couple years ago.
#
# Otherwise, you would have to change `main` to `master` manually in all the commands of this notebook.
# ```

# %%
# ! git config --global init.defaultBranch main

# %%
# ! git init
# ! dvc init

# %% [markdown]
# As with the previous notebook, you might need to configure your username and email:

# %%
# ! git config --local user.email "john.doe@inria.fr"
# ! git config --local user.name "John Doe"
# ! git commit -m "initialize DVC"

# %% [markdown]
# ```{note}
# Throughout this tutorial, we will write things to files. Usually, you would do this using your favorite IDE. However, we need this tutorial to be runnable as a notebook (for example for the attendees running it on Collab). Because of this, we will be using IPython magic commands `%%writefile` and `%run` in order to write to a file the content of a cell, and run a given python script.
#
# If you are following locally, you can of course edit the different files directly !
# ```
#
# Let's add some patterns to the [.gitignore](https://git-scm.com/docs/gitignore) file in order to not display IPython notebooks related files when doing a [git status](https://git-scm.com/docs/git-status):

# %%
# %%writefile -a .gitignore

*.ipynb
__pycache__
.DS_Store
.ipynb_checkpoints
.config
OASIS-1-dataset*
sample_data

# %% [markdown]
# ## Getting the input dataset
#
# In this session we use the images from a public research project: [OASIS-1](https://www.oasis-brains.org). Two labels exist in this dataset:
#
# - CN (Cognitively Normal) for healthy participants.
# - AD (Alzheimer’s Disease) for patients affected by [Alzheimer's disease](https://en.wikipedia.org/wiki/Alzheimer%27s_disease).
#
# The original images were preprocessed using [Clinica](https://www.clinica.run): a software platform for clinical neuroimaging studies.
#
# Preprocessed images and other files are distributed in a tarball, if you haven't downloaded the images before the tutorial, run the following commands to download and extract them:

# %%
# Only run if necessary !
#
# # ! wget --no-check-certificate --show-progress https://aramislab.paris.inria.fr/files/data/databases/DL4MI/OASIS-1-dataset_pt_new.tar.gz
# # ! tar -xzf OASIS-1-dataset_pt_new.tar.gz

# %% [markdown]
# ## Write a Python script to run the experiment
#
# Let's start writing a Python file which will contain the code required to train our model. Our objective is to be able to run
#
# ```
# $ python train.py
# ```
#
# from the command-line to train our model on our input dataset.
#
# We are going to progressively add things to this file in order to fullfil this goal.
#
# Let's start by defining the path to the raw dataset folder we just downloaded. If you run the cell above, it should be `./OASIS-1_dataset`. If you downloaded the data before the workshop, it is probably somewhere else on your machine.

# %%
# %%writefile -a train.py 

from pathlib import Path

# If you just downloaded the data using the cell above, then uncomment:
#
# oasis_folder = Path("./OASIS-1_dataset/")
#
# Otherwise, modify this path to the folder in which you extracted the data:
#
oasis_folder = Path("/Users/nicolas.gensollen/NOW_2023/OASIS-1_dataset/")

# %% [markdown]
# Executing the previous cell wrote its content to our file, let's also execute this file to have the declared imports and variables in this Python session:

# %%
# %run train.py

# %%
print([file.name for file in oasis_folder.iterdir()])

# %% [markdown]
# The raw dataset contains:
#
# - a `tsv_files` folder in which we have metadata relative to the different subjects
# - a `README.md` file giving some information on the dataset
# - a `CAPS` folder holding the preprocessed brain images
# - a `raw` folder holding the raw brain images

# %% [markdown]
# At the same time, we will write a second Python file `prepare_train_validation_sets.py` which will be responsible for splitting our list of subjects into a training set and a validation set. This is important because we will compute validation metrics once our model has been trained:

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

# %% [markdown]
# Let's take a look at some statistics in order to better understand our data:

# %%
print(OASIS_df.head())

_ = OASIS_df.hist(figsize=(16, 8))

# %% [markdown]
# From these graphics, it’s possible to have an overview of the distribution of the data, for the numerical values. For example, the educational level is well distributed among the participants of the study. Also, most of the subjects are young (around 20 years old) and healthy (MMS score equals 30 and null CDR score).
#
# We can use the [characteristics_table](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.utils.characteristics_table.html#now_2023.utils.characteristics_table) function from [now-2023](https://now-2023.readthedocs.io/en/latest/index.html) to get some useful statistics at the population level:

# %%
from now_2023.utils import characteristics_table

population_df = characteristics_table(OASIS_df, OASIS_df)
population_df

# %% [markdown]
# ## Preprocessing
#
# Theoretically, the main advantage of deep learning methods is to be able to work without extensive data preprocessing. However, as we have only a few images to train the network in this lab session, the preprocessing here is very extensive. More specifically, the images encountered:
#
# - Non-linear registration.
# - Segmentation of grey matter.
# - Conversion to tensor format (.pt).
#
# As mentioned above, to obtain the preprocessed images, we used some pipelines provided by [Clinica](https://www.clinica.run) and [ClinicaDL](https://clinicadl.readthedocs.io/en/latest/) in order to:
#
# - Convert the original dataset to [BIDS](https://bids-specification.readthedocs.io/en/stable/) format ([clinica convert oasis-2-bids](https://aramislab.paris.inria.fr/docs/public/latest/Converters/OASIS2BIDS/)).
# - Get the non-linear registration and segmentation of grey mater (pipeline [t1-volume](https://aramislab.paris.inria.fr/docs/public/latest/Pipelines/T1_Volume/)).
# - Obtain the preprocessed images in tensor format ([tensor extraction using ClinicaDL, clinicadl extract](https://clinicadl.readthedocs.io/en/stable/Preprocessing/Extract/)).
# - The preprocessed images are store in the [CAPS folder structure](http://www.clinica.run/doc/CAPS/Introduction/) and all have the same size (121x145x121).
#
# To facilitate the training and avoid overfitting due to the limited amount of data, the model won’t use the full image but only a part of the image (size 30x40x30) centered on a specific neuroanatomical region: the [hippocampus](https://en.wikipedia.org/wiki/Hippocampus) (HC). This structure is known to be linked to memory, and is atrophied in the majority of cases of Alzheimer’s disease patients.
#
# Before going further, let's take a look at the images we have downloaded and let's compute a cropped image of the left HC for a randomly selected subject:

# %%
import torch

# Select a random subject
subject = 'sub-OASIS10003'

# The path to the hc image
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

# The image file name as a specific structure
image_filename = f"{subject}_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.pt"
preprocessed_pt = torch.load(image_folder / image_filename)

# %% [markdown]
# You can use the [CropLeftHC](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.utils.CropLeftHC.html) class to automatically crop the left HC from the preprocessed tensor images. You can also use [plot_image](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.plotting.plot_image.html) and [plot_tensor](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.plotting.plot_tensor.html) functions from the [now-2023](https://now-2023.readthedocs.io/en/latest/index.html) library to visualize these images.
#
# `````{admonition} View different slices
# :class: tip
#
# Do not hesitate to play with the `cut_coords` argument to view different slices!
# `````

# %%
from now_2023.plotting import plot_image, plot_tensor
from now_2023.utils import CropLeftHC

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
# We are going to generate a new dataset consisting only of images of the left [hippocampus](https://en.wikipedia.org/wiki/Hippocampus). Our dataset will basically consist of images like the last image above. 
#
# For simplicity, this is the dataset that we will consider as our input dataset, just like we pretended that the TSV file with the HC volumes computed was our input dataset in the previous chapter. This also means that the dataset we are about to generate will be the one we will version.
#
# ```{note}
# Note that there is nothing preventing us to version the raw dataset. We are doing this because we will pretend to receive a data update later on consisting of the right HC images.
# ```
#
# Note also that to improve the training and reduce overfitting, we can add a random shift to the cropping function. This means that the bounding box around the [hippocampus](https://en.wikipedia.org/wiki/Hippocampus) may be shifted by a limited amount of voxels in each of the three directions.
#
# Let's generate our dataset with the [generate_cropped_hc_dataset](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.data_generation.generate_cropped_hc_dataset.html#now_2023.data_generation.generate_cropped_hc_dataset) function from the [now-2023](https://now-2023.readthedocs.io/en/latest/index.html) library.

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
# This should have created a new `data` folder in the current workspace, which should have the following structure:

# %%
# ! tree data | head -n 14

# %% [markdown]
# As you can see, we have one tensor image for each subject, representing the extracted left hippocampus.
#
# Let's take a look at some of these images. To do so, you can use the [plot_hc](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.plotting.plot_hc.html) function from the [now-2023](https://now-2023.readthedocs.io/en/latest/index.html) library which is a wrapper around the [plot_tensor](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.plotting.plot_tensor.html) function you used before.
#
# Again, feel free to play with the parameters to view different slices or different subjects: 

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

# %% [markdown]
# At this point, the step of our experiment consisting of preparing the training and validation sets is complete. We could simply run 
#
# ```bash
# python prepare_train_validation_sets.py
# ```
#
# but we are going to see another way to do that. Instead of just running this command, we are going to use [DVC](https://dvc.org) to codify our first experimental step. To do that, we will rely on the [dvc stage](https://dvc.org/doc/command-reference/stage) comand which is a bit more complicated than the previous [DVC](https://dvc.org) commands we saw before:

# %%
# ! dvc stage add -n prepare_train_validation \
#   -d prepare_train_validation_sets.py -d data \
#   -o train.csv -o validation.csv \
#   python prepare_train_validation_sets.py

# %% [markdown]
# Let's take a closer look at this long command:
#
# - The `-n` option enables us to give a name to this step. In our case, we named it "prepare_train_validation".
# - The `-d` option enables us to declare dependencies, that is things on which this step depends. In our case, the step depends on the input data (the `data` folder), and the python file itself (`prepare_train_validation_sets.py`).
# - The `-o` option enables ut to declare outputs, that is things that are produced by this step. In our case, the step produces two CSV files (`train.csv` and `validation.csv`)
# - The final part of the command tells [DVC](https://dvc.org) the command it should run to perform this step. In our case, it needs to run `python prepare_train_validation_sets.py` to run the Python script.
#
# OK, great, but what did this command actually do ?
#
# As you can see from its output, it generated a `dvc.yaml` file which encode our experiment stage as well as its dependencies. Let's have a look at it:

# %%
# ! cat dvc.yaml

# %% [markdown]
# As you can see, it is pretty easy to read and understand. There is nothing more than what we just described above.
#
# However, [dvc stage](https://dvc.org/doc/command-reference/stage) didn't run anything, it just generated this file. To run our first step, we can use the [dvc repro](https://dvc.org/doc/command-reference/repro) command:

# %%
# ! dvc repro

# %% [markdown]
# In the output of [dvc repro](https://dvc.org/doc/command-reference/repro) we can see the output generated by our Python scripts as well as the output generated by [DVC](https://dvc.org).
#
# It looks like [DVC](https://dvc.org) generated a `dvc.lock` file:

# %%
# ! cat dvc.lock

# %% [markdown]
# We won't go into the details here, but the main idea is that this lock file is what [DVC](https://dvc.org) uses to know whether it should re-run a given stage given the state of the current workspace. It basically does this by computing the hash values of the stage dependencies and outputs and comparing these values to the ones in this file. If there is at least one mismatch, then the stage should be run again, otherwise [DVC](https://dvc.org) will use the cached inputs and outputs.
#
# Note that these new files (`dvc.yaml` and `dvc.lock`) are still very small files which size does not depend on the input data size. This means that we are totally fine versioning them with [Git](https://git-scm.com), and this is precisely what [DVC](https://dvc.org) is telling us to do here.
#
# Let's add the data with [dvc add](https://dvc.org/doc/command-reference/add) (we still haven't done that...), and do the same with [Git](https://git-scm.com) for the files we have generated (`dvc.yaml`, `dvc.lock`, `data.dvc`), and modified (`.gitignore`):

# %%
# ! dvc add data
# ! git add dvc.yaml .gitignore dvc.lock data.dvc

# %%
# ! git status

# %% [markdown]
# [DVC](https://dvc.org) also provides some utilities to visualize the data pipelines as graphs. We can visualize our pipeline with the [dvc dag](https://dvc.org/doc/command-reference/dag) command:

# %%
# ! dvc dag

# %% [markdown]
# By default, the [dvc dag](https://dvc.org/doc/command-reference/dag) command returns a simple textual representation of the graph, but there exists various ways to get more complex representation like [mermaid flowcharts](https://mermaid.js.org/syntax/flowchart.html) for example:

# %%
# ! dvc dag --mermaid

# %% [markdown]
# ### Train the model
#
# We propose here to use a convolutional neural network model to make our prediction. Again, we won't go into the architectural details of this model as this isn't the objective of this tutorial. Instead, we will use the [CNNModel](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.models.CNNModel.html#now_2023.models.CNNModel) from the [now-2023](https://now-2023.readthedocs.io/en/latest/index.html) library.
#
# The network model requires some hyper-parameters:
#
# - the *learning_rate*
# - the number of *epochs*
# - the *batch size*
#
# We could just hardcode them in our Python script when instantiating the model but we are going to implement this in a more flexible way. We will see later in this tutorial why it is interresting to define them this way.
#
# [DVC](https://dvc.org) enables ut to define parameters in specific YAML files. Let's create a `params.yaml` file and define our hyper-parameters in it: 

# %%
# %%writefile -a params.yaml

train:
    learning_rate: 0.0001
    n_epochs: 30
    batch_size: 4

# %% [markdown]
# The structure is pretty simple, you define the parameters as key-value pairs and you can either define them all at the root, or organize them in a hierarchical way.
#
# Here we decided to put all the parameters inside a `train` section, but we could have defined them all at the root as well.
#
# We are now ready to finish our Python script responsible for creating the model, training it, and saving the results. Let's do this step by step.
#
# First, we need to retrieve the hyper-parameters we just defined in the `params.yaml` file. We use the [DVC Python API](https://dvc.org/doc/api-reference) to do so.

# %%
# %%writefile -a train.py

import dvc.api

params = dvc.api.params_show()

# Parameters can be accessed through a dictionnary interface
# We get a nested dict because we defined them within a section named 'train'
learning_rate = params['train']['learning_rate']
n_epochs = params['train']['n_epochs']
batch_size = params['train']['batch_size']

# %% [markdown]
# Then, we need to load the training and validation sets which are available as two CSV files thanks to the previous step of the pipeline:

# %%
# %%writefile -a train.py

import pandas as pd

train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("validation.csv")

# %% [markdown]
# At this point, we have everything we need to instantiate and train our network model:

# %%
# %%writefile -a train.py

from now_2023.models import CNNModel

data_folder = Path("./data")

print("*" * 50)
model = CNNModel(
    learning_rate=learning_rate,
    n_epochs=n_epochs,
    batch_size=batch_size,
)
print(f"Fitting model on {len(train_df)} samples...")
model.fit(data_folder, train_df)

# %% [markdown]
# The only thing left to do is to compute metrics and save our ouptuts to disk. We generate the following outputs:
#
# - The model's weights in `model.h5`.
# - The metrics in `metrics.json`.
# - A small human-readable note describing what we did in `notes.txt` (this could be useful, for example, to your future self revisiting this experiment in a few months or years...).

# %%
# %%writefile -a train.py

import json

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
# We can now add this new stage to our pipeline with the [dvc stage](https://dvc.org/doc/command-reference/stage) command. As we saw with the previous stage, this command is quite long so we need to think about this stage dependencies and outputs:
#
# First, we will call this stage `train_network`. Then, this stage requires three parameters (defined in the `params.yaml` file), and it has four dependencies (the `train.py` file which is the Python code implementing what needs to be done in this stage, the `data` folder which contains our input images on which the network is trained, and the two CSV files `train.csv` and `validation.csv` which define the training and validation sets).
#
# In addition, it generates two outputs (`model.h5` which contains the model's weights after training, and `notes.txt` which is our small note), as well as one metric file, `metrics.json`, that [DVC](https://dvc.org) handles in a slightly different way than the other outputs.
#
# Finally, we need to give the command that should be run for executing this stage. In our case, it is simply `python train.py`.
#
# With this in mind, we can write the [dvc stage](https://dvc.org/doc/command-reference/stage) command:

# %%
# ! dvc stage add -n train_network \
#   -p train.learning_rate,train.n_epochs,train.batch_size   \
#   -d train.py -d data -d train.csv -d validation.csv \
#   -o model.h5 -o notes.txt -M metrics.json python train.py

# %% [markdown]
# Let's check the `dvc.yaml` file which should have been modified:

# %%
# ! cat dvc.yaml

# %% [markdown]
# Great ! The `dvc.yaml` file has been modified with the new `train_network` stage we just created. We can see that this stage depends on the outputs of the `prepare_train_validation` stage as well as on the input data. Let's take a look at the graph representation to verify this:

# %%
# ! dvc dag

# %% [markdown]
# Indeed, [DVC](https://dvc.org) was able to infer the dependencies between the different stages of our pipeline.
#
# Let's run the pipeline:
#
# ```{note}
# Remember that we are training a neural network here. This step will take a few minutes to run, especially if you are running this using a CPU. For people running on Collab and having selected a GPU, this should be much faster.
# ```

# %%
# ! dvc repro

# %% [markdown]
# Let's get some information on the state of our workspace:

# %%
# ! git status

# %% [markdown]
# Let's version the [DVC](https://dvc.org) files as well as our Python scripts and commit the whole thing:

# %%
# ! git add .gitignore dvc.yaml dvc.lock train.py prepare_train_validation_sets.py params.yaml
# ! git commit -m "First model, trained with images cropped around left HC"
# ! git tag -a "v1.0" -m "model v1.0, left HC only"
# ! git log

# %% [markdown]
# Note that, if we re-run [dvc repro](https://dvc.org/doc/command-reference/repro), nothing will happen since everything is up-to-date:

# %%
# ! dvc repro

# %% [markdown]
# As a final note on this section, we can take a look at our [DVC](https://dvc.org) cache and see that it is much more complicated than the one we had in the previous chapter. Which is expected since we are versioning a much more complex dataset.

# %%
# ! tree .dvc/cache/files/md5 | head -n 15

# %% [markdown]
# ## Use both the left and right HC
#
# Let's imagine now that we receive additional data in the form of cropped images of the right HC.
#
# ```{note}
# Of course we are only pretending here! Recall that we are in fact generating these images from the downloaded raw dataset.
# ```

# %%
# Generate the cropped images of the right HC
generate_cropped_hc_dataset(
    oasis_folder,
    hemi="right",
    output_folder=Path("./data"),
    verbose=False,
)

# %% [markdown]
# This changed our input dataset:

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
# At this point, we want to re-train our model with this larger input dataset.
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
# ```{note}
# Note that, for simplificity due to the constraints of the notebook format, we just appenned the previous code to the `prepare_train_validation_sets.py` file instead of modifying it, which would be much cleaner. Feel free to open the python file and replace the relevant portion of the code with the previous cell.
# ```
#
# And... that's all we need to do ! We don't need to think about what should be re-run and what shouldn't, [DVC](https://dvc.org) is taking care of that for us. We can simply call [dvc repro](https://dvc.org/doc/command-reference/repro) to run the pipeline:

# %%
# ! dvc repro

# %% [markdown]
# Let's take a look at our workspace:

# %%
# !git status

# %% [markdown]
# The Python script was modified by us and [DVC](https://dvc.org) modified `data.dvc` and `dvc.lock`.
#
# The change to `data.dvc` is due to the fact that we added more data to our input dataset:

# %%
# ! git diff data.dvc

# %% [markdown]
# The changes made to `dvc.lock` are hash value updates to encode the changes to the dependencies and outputs:

# %%
# ! git diff dvc.lock

# %% [markdown]
# Let's commit our changes and tag this as our V2:

# %%
# ! git add data.dvc dvc.lock prepare_train_validation_sets.py
# ! git commit -m "Second model, trained with images cropped around left and right HC"
# ! git tag -a "v2.0" -m "model v2.0, left and right HC"
# ! git log

# %% [markdown]
# ## Reproducing
#
# As in the previous chapter, imagine that we have to go back to the first version of our experiment. As we saw, we can restore the state of the project with the `checkout` commands:

# %%
# ! git checkout v1.0
# ! dvc checkout

# %% [markdown]
# Now, we are going to see the benefits of having defined our experiment as a data pipeline with [DVC](https://dvc.org).
#
# If we check the status, [DVC](https://dvc.org) tells us that there is a change in the `run_experiment` stage:

# %%
# ! dvc status

# %% [markdown]
# And, if we run [dvc repro](https://dvc.org/doc/command-reference/repro), [DVC](https://dvc.org) is clever enough to know which stages to run given the dependencies that were modified.
#
# Since we didn't modify anything besides going back to version 1 of the experiment, [DVC](https://dvc.org) won't re-run anything because everything is already in the cache. It only needs to point to the correct files in the cache:

# %%
# ! dvc repro

# %% [markdown]
# We can verify that our human-readable notes correspond to the experiment we are currently running (300 samples with learning rate of 0.0001):

# %%
# ! cat notes.txt

# %% [markdown]
# Now, let's imagine we want to experiment the first version of our model with a different learning rate. 
#
# All we need to do is to open the `params.yaml` file and update the parameter to the desired value.
#
# Let's try increasing it to `10^3`:

# %%
# %%writefile params.yaml

train:
    learning_rate: 0.001
    n_epochs: 30
    batch_size: 4

# %% [markdown]
# Again, we don't need to remember what this small change will impact in our experiment. [DVC](https://dvc.org) handles this very well:

# %%
# ! dvc repro

# %% [markdown]
# In this case, [DVC](https://dvc.org) knows it can re-use the output of the `prepare_train_validation` stage since none of its dependecies changed. However it has to re-run the `train_network` stage because there is no results cached which match the new set of dependencies.
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
# ! rm train.py prepare_train_validation_sets.py
# ! rm dvc.*
# ! rm train.csv validation.csv
# ! rm notes.txt
# ! rm params.yaml

# %%
