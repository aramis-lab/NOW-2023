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
# # Part 2 : Data versionning using DVC
#
# ## Chapter 1 : Getting started with DVC
#
# This is the second part of the tutorial on data versionning using [DVC](https://dvc.org).
#
# In this first chapter, we will install DVC and take a look at how DVC works with GIT in its most simple usage.
#
# ### Install DVC
#
# First, we need to install [DVC](https://dvc.org). There are various ways to install it depending on you OS, which you can browse [here](https://dvc.org/doc/install). For example, on MacOS, you can install it with `brew`, `conda`, or `pip`.
#
# If you are following this tutorial on your own machine, chose the option that makes the most sense. If you are following on the notebook, we will install DVC with `pip`:

# %%
# ! pip install dvc

# %% [markdown]
# We can check that DVC is installed:

# %%
# ! which dvc

# %%
# ! dvc --version

# %% [markdown]
# ### Initialize a DVC repository
#
# Now that we have DVC installed we can start using it !
#
# First of all, it is very important to understand that DVC is not a replacement for GIT. It is a tool designed to work WITH GIT as it solves a different problem than GIT.
#
# In other words, you need both GIT and DVC to manage both code and data.
#
# To initialize a DVC repository, we need to be in a GIT-initialized repository, so let's do that:

# %%
# ! git init

# %% [markdown]
# You can check that a `.git` hidden folder was created:

# %%
# ! ls -lah

# %% [markdown]
# Now we can initialize the DVC repository:

# %%
# ! dvc init

# %% [markdown]
# In the exact same way as for `git init`, `dvc init` created a hidden folder named `.dvc`:

# %%
# ! ls -lah

# %% [markdown]
# In addition to this, DVC created a few files for us. To see that, we can use the `git status` command since we have a git repository:

# %%
# ! git status

# %% [markdown]
# As we can see, DVC created 2 files in the `.dvc` folder as well as one file in the current workspace :
#
# - `.dvc/.gitignore`
# - `.dvc/config`
# - `.dvcignore`
#
# These files need to be versionned with GIT, DVC already added them to the stagging aread, so all we need to do is commit them:

# %%
# ! git commit -m "initialize DVC"

# %%
# ! git log

# %% [markdown]
# And that's it, we have successfully initialized a DVC repository and we are now ready to track some code and data !
#
# ### Track code and data
#
# In this section we are going to download some data that we will use as input for a classification model. The objective is to predict whether a patient has AD or not. We are going to keep this same objective for the whole tutorial but we will start very simple and add some complexity up a real experiment scenario.
#
# #### Basic data with a basic model
#
# Let's start VERY simple with a dataset in the form of a TSV file with one row per subject and four columns:
#
# - the patient ID which is just a string identifier
# - the volume of this patient's left hypocampus (this will be our first predictive feature)
# - the volume of this patient's right hypocampus (this will be second first predictive feature)
# - the category of the patient: "AD" if the patient has Alzeihmer disease or "CN" for control (this will be out target)
#
# Let's download the data first. For this, we will use a special DVC command: `dvc get`:

# %%
# ! dvc get https://github.com/aramis-lab/dataset-registry NOW_2023/toy_dataset/version_1
# ! mv version_1/dataset.tsv .
# ! rm -r version_1

# %% [markdown]
# Let's open the TSV file we just downloaded with Pandas and take a look at the subjects data:

# %%
import pandas as pd

df = pd.read_csv("dataset.tsv", sep="\t")
df.head()

# %% [markdown]
# We have 500 subjects:

# %%
df.tail()

# %% [markdown]
# In this tutorial, we won't focus on the machine learning aspect of things but rather on the code and data management aspects.
#
# This means that we will treat models as black boxes. For simplicity, models have been wrapped in custom classes offering a very simple API:
#
# - `model.fit()` for fitting the model with some input data
# - `model.save()` for saving the model's weights to disk
# - `model.plot()` to give us some visuals
#
# Here, we use a toy model (a simple non-linear SVM), and we fit it with the data we just downloaded:

# %%
from toy_model import Model

model = Model.from_dataframe(df)
model.fit()  # Fit the model
model.save() # Serialize the trained model
model.plot() # Plot the decision function with the data

# %% [markdown]
# We can see that the model is able to learn a decision function to classify subjects with AD from control subjects.
#
# At this point, we are happy we our experiment and we wish to commit our changes. This is the moment where we need to understand what should be tracked with DVC and what should be tracked with GIT.
#
# First, `dataset.tsv` is our input data, so this is clearly something we shouldn't version with GIT (although GIT would manage it with such a simple dataset). We should clearly use DVC to track our dataset. The same goes with our experiment results: `model.pkl` which is our trained model serialized.
#
# Let's track these two files with DVC:

# %%
# ! dvc add dataset.tsv model.pkl

# %% [markdown]
# As you can see, DVC is helping us by telling us what we should do next !
#
# Tracking `dataset.tsv` and `model.pkl` generated two small metadata files:
#
# - `dataset.tsv.dvc`
# - `model.pkl.dvc`
#
# Furthermore, DVC modified our `.gitignore` file to tell GIT to NOT track `dataset.tsv` and `model.pkl`.
#
# All these files are super small and easily managable for GIT. Let's version them with GIT:

# %%
# ! git add dataset.tsv.dvc model.pkl.dvc .gitignore

# %% [markdown]
# At this point we can commit our changes and optionally tag the commit.
#
# Here, we will call this the "v1.0" of our experiment, for which we used 500 subjects:

# %%
# ! git commit -m "First model, trained with 500 subjects"
# ! git tag -a "v1.0" -m "model v1.0, 500 subjects"

# %%
# ! git log

# %% [markdown]
# #### Data can change
#
# Datasets are not always fixed in time, they may evolve. For example, new data can be collected and added in new dataset releases.
#
# Let's imagine that our dataset received a brand new release with 500 additional subjects. Chances are that this will impact our previous experiment.
#
# How should we handle such an update ?
#
# Let's dive in and download the updated dataset:

# %%
# ! dvc get https://github.com/aramis-lab/dataset-registry NOW_2023/toy_dataset/version_2
# ! mv version_2/dataset.tsv .
# ! rm -r version_2

# %% [markdown]
# As advertized, we now have data for 1000 subjects:

# %%
df = pd.read_csv("dataset.tsv", sep="\t")
print(df.head(3))
print(df.tail(3))

# %% [markdown]
# Let's recreate a model instance and fit it with our new dataset:

# %%
model = Model.from_dataframe(df)
model.fit()
model.save()
model.plot()

# %% [markdown]
# Great ! we now have our new model and we can use DVC to see what happened to our tracked data:

# %%
# ! dvc status

# %% [markdown]
# As expected, DVC is telling us that both our dataset and our serialized model have changed.
#
# As we would do with GIT, let's add these changes with DVC:

# %%
# ! dvc add dataset.tsv model.pkl

# %% [markdown]
# Again, DVC tells us what to do next, let's add the changes to the `.dvc` files with GIT:

# %%
# ! git add dataset.tsv.dvc model.pkl.dvc

# %% [markdown]
# We can now commit these changes and tag this commit with a "v2.0" tag:

# %%
# ! git commit -m "Second model, trained with 1000 subjects"
# ! git tag -a "v2.0" -m "model v2.0, 1000 subjects"

# %%
# ! git log

# %% [markdown]
# #### Going back in time
#
# Now, let's imagine that we submitted a paper with the results of our first experiment and that we received the review after having done the new experiment just above.
#
# The reviewer is asking us to add some details on some of our plots. Clearly, we cannot use the new dataset and model to make the new plots !
#
# We need to go back to the state we were in when we generated those plots the first time. Fortunately, this is very easy to do thanks to both GIT and DVC.
#
# In our example, we were smart enough to tag the state of our experiment with the "v1.0" tag (we could have used something more informative like "publication for journal XXX"). 
#
# We can thus simply do:

# %%
# ! git checkout v1.0
# ! dvc checkout

# %% [markdown]
# Such that we are now in the exact same state as before the data update:

# %%
df = pd.read_csv("dataset.tsv", sep="\t")
print(len(df))  # Dataset has 500 subjects

# %% [markdown]
# Cleaning:

# %%
# Do not run this unless you want to start over from scratch...
# ! rm dataset.tsv*
# ! rm model.pkl*
# ! rm -rf .git
# ! rm -rf .dvc
# ! rm .gitignore
# ! rm .dvcignore

# %%
