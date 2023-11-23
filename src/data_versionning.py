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
# # Chapter 2 : Getting started with DVC
#
# In this chapter, we will install [DVC](https://dvc.org) and take a look at how [DVC](https://dvc.org) works with [Git](https://git-scm.com) in its most simple usage.
#

# %%
# If you are running on collab or if you don't have tree installed:
# # ! apt-get install tree

# %% [markdown]
# ## Install DVC
#
# First, we need to install [DVC](https://dvc.org). There are various ways to install it depending on you OS, which you can browse [here](https://dvc.org/doc/install). For example, on MacOS, you can install it with [brew](https://brew.sh), [conda](https://anaconda.org), or [pip](https://pip.pypa.io/en/stable/).
#
# If you are following this tutorial on your own machine, chose the option that makes the most sense.
#
# `````{admonition} Virtual environments
# :class: tip
#
# We strongly encourage you to create a virtual environment specifically for this tutorial. For example, if you are using conda, you can do the following:
#
# ```bash
# $ conda create --name now python=3.10
# $ conda activate now
# ```
#
# `````
#
# If you are following on Collab you can `pip install` things without worrying about this.
#
# We will install [DVC](https://dvc.org) with [pip](https://pip.pypa.io/en/stable/):

# %%
# # brew install dvc
# # conda install dvc
# ! pip install dvc

# %% [markdown]
# ```{note}
# Note that if you are running this notebook on [Collab](https://colab.google), you might need to click on the "RESTART RUNTIME" button just above.
# ```
#
# We can check that [DVC](https://dvc.org) is installed:

# %%
# ! which dvc

# %%
# ! dvc --version

# %% [markdown]
# ## Initialize a DVC repository
#
# Now that we have [DVC](https://dvc.org) installed we can start using it !
#
# First of all, it is very important to understand that [DVC](https://dvc.org) is not a replacement for [Git](https://git-scm.com). It is a tool designed to work **WITH** [Git](https://git-scm.com) as it solves a different problem than [Git](https://git-scm.com).
#
# In other words, you need both [Git](https://git-scm.com) and [DVC](https://dvc.org) to manage both code and data.
#
# To initialize a [DVC](https://dvc.org) repository, we first need to be within a [Git](https://git-scm.com)-initialized repository, so let's do that!
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

# %% [markdown]
# You can check that a `.git` hidden folder was created:

# %%
# ! ls -lah

# %% [markdown]
# Now we can initialize the [DVC](https://dvc.org) repository:

# %%
# ! dvc init

# %% [markdown]
# In the exact same way as for [git init](https://git-scm.com/docs/git-init), [dvc init](https://dvc.org/doc/command-reference/init) created a hidden folder named `.dvc`:

# %%
# ! ls -lah

# %% [markdown]
# In addition to this, [DVC](https://dvc.org) created a few files for us.
#
# To see that, we can use the [git status](https://git-scm.com/docs/git-status) command since we have a [Git](https://git-scm.com) repository:

# %%
# ! git status

# %% [markdown]
# As we can see, [DVC](https://dvc.org) created 2 files in the `.dvc` folder as well as one file in the current workspace :
#
# - `.dvc/.gitignore`
# - `.dvc/config`
# - `.dvcignore`
#
# These files are very small and need to be versionned with [Git](https://git-scm.com).
#
# [DVC](https://dvc.org) behaves very similarly to [Git](https://git-scm.com) and will often print helpful messages on what to do. In this case, [DVC](https://dvc.org) already added the new files to the stagging area, so all we need to do is commit them using the [git commit](https://git-scm.com/docs/git-commit) command:

# %% [markdown]
# ```{note}
# If you are running the current notebook on Collab, or if you have never configured git before, you need to configure your user name and email address with the `git config` command. You can either use the `--global` option to configure this globally on your machine, or the `--local` option to configure this only for the current project. Note that, if you are running on Collab, it doesn't really matter as the configuration will only leave for the duration of the session:
# ```
#

# %%
# ! git config --local user.email "john.doe@inria.fr"
# ! git config --local user.name "John Doe"

# %%
# ! git commit -m "initialize DVC"

# %% [markdown]
# We can visualize the history of our project using the [git log](https://git-scm.com/docs/git-log) command:

# %%
# ! git log

# %% [markdown]
# And that's it, we have successfully initialized a [DVC](https://dvc.org) repository and we are now ready to track some code and data !
#
# Let's quickly create a [.gitignore](https://git-scm.com/docs/gitignore) file with some useful patterns to avoid tracking jupyter notebook related files with [Git](https://git-scm.com). This will make our [git status](https://git-scm.com/docs/git-status) command outputs much cleaner and enable us to focus on the important files of the work space:

# %%
# %%writefile -a .gitignore

*.ipynb
__pycache__
.DS_Store
.ipynb_checkpoints
.config
sample_data

# %%
# ! git add .gitignore
# ! git commit -m "add gitignore file"
# ! git log

# %%
# ! git status

# %% [markdown]
# ## Track code and data
#
# In this section we are going to download some data that we will use as input for a [classification](https://en.wikipedia.org/wiki/Statistical_classification) model. The objective is to predict whether a patient has [Alzheimer's disease](https://en.wikipedia.org/wiki/Alzheimer%27s_disease) (AD) or not. We are going to keep this same objective for the whole tutorial, but we will start very simple and add some complexity up to a real experiment scenario.
#
# ```{note}
# The [Hippocampus](https://en.wikipedia.org/wiki/Hippocampus) is a major component of the brain of humans and other vertebrates. Humans and other mammals have two hippocampi, one in each side of the brain. The [hippocampus](https://en.wikipedia.org/wiki/Hippocampus) is part of the limbic system, and plays important roles in the consolidation of information from short-term memory to long-term memory, and in spatial memory that enables navigation. This structure is known to be linked to memory, and is atrophied in the majority of cases of [Alzheimer's disease](https://en.wikipedia.org/wiki/Alzheimer%27s_disease) patients.
# ```
#
# In this tutorial we will rely extensively on the volumes of the hippocampi to make our predictions.

# %% [markdown]
# ### Basic data with a basic model
#
# Let's start **VERY** simple with a dataset in the form of a single TSV file. This file contains one row per subject and four columns:
#
# - the **patient ID** which is just a string identifier
# - the **volume** of this patient's **left hippocampus** (this will be our first predictive feature)
# - the **volume** of this patient's **right hippocampus** (this will be second first predictive feature)
# - the **diagnosis** of the patient: "AD" if the patient has [Alzheimer's disease](https://en.wikipedia.org/wiki/Alzheimer%27s_disease) or "CN" for control (this will be out target)
#
# Let's download the data first. For this, we will use a special [DVC](https://dvc.org) command: [dvc get](https://dvc.org/doc/command-reference/get):

# %%
# ! dvc get https://github.com/aramis-lab/dataset-registry NOW_2023/hc_volumes/version_1
# ! mv version_1/dataset1.tsv dataset.tsv
# ! rm -r version_1

# %% [markdown]
# Let's open the TSV file we just downloaded with [Pandas](https://pandas.pydata.org) and take a look at the subjects data:

# %%
import pandas as pd

df = pd.read_csv("dataset.tsv", sep="\t")
df.head()

# %% [markdown]
# We can see that we have 208 subjects:

# %%
df.tail()

# %% [markdown]
# Recall that this tutorial is on code and data versioning, not on machine or deep learning ! In orther words, we won't focus on the machine learning aspect of things, but rather on the code and data management aspects.
#
# This means that **we will treat models as black boxes**. For simplicity, we will rely in this tutorial on [now-2023](https://now-2023.readthedocs.io/en/latest/index.html), a small library created in order to offer a very simple API. In this chapter, we will use the [ToyModel](https://now-2023.readthedocs.io/en/latest/modules/generated/now_2023.models.ToyModel.html#now_2023.models.ToyModel), which has the following public API:
#
# - `from_dataframe()` for simple instantiation.
# - `fit()` for fitting the model with some input data.
# - `save()` for saving the model's weights to disk.
# - `plot()` to give us some visuals.
#
# If you are running this notebook locally and have followed the installation steps from the README, then now-2023 should already be installed. Otherwise, or if you are running this notebook on Collab, you need to install it:

# %%
# pip install now-2023

# %%
from now_2023.models import ToyModel

model = ToyModel.from_dataframe(
    df,
    predictive_features=["HC_left_volume", "HC_right_volume"],
    target_feature="diagnosis",
)
model.fit()  # Fit the model
model.save() # Serialize the trained model
model.plot() # Plot the decision function with the data

# %% [markdown]
# We can see that the model is able to learn a decision function to classify subjects with AD (red dots) from control subjects. Furthermore, we can see that AD subjects seem to have smaller volumes than control subjects which is in accordance with our expectations. 
#
# But what happened to our work space ? We can use [git status](https://git-scm.com/docs/git-status) to get some useful information:

# %%
# ! git status

# %% [markdown]
# We have two new untracked files: `dataset.tsv` which is our dataset we just downloaded, and `model.pkl` which is our trained model serialized to disk.
#
# At this point, we are happy with our experiment and we wish to commit our changes. This is the moment where we need to understand what should be tracked with [DVC](https://dvc.org) and what should be tracked with [Git](https://git-scm.com).
#
# First, `dataset.tsv` is our input data, so this is clearly something we shouldn't version with [Git](https://git-scm.com) (although [Git](https://git-scm.com) would manage it with such a simple dataset). We should clearly use [DVC](https://dvc.org) to track our dataset. The same goes with our experiment results: `model.pkl` which is our trained model serialized.
#
# Let's track these two files with [DVC](https://dvc.org). To do so, we will use the [dvc add](https://dvc.org/doc/command-reference/add) command. Again, the [DVC](https://dvc.org) command is very similar to the git command:

# %%
# ! dvc add dataset.tsv model.pkl

# %% [markdown]
# A few things happened here !
#
# Tracking `dataset.tsv` and `model.pkl` generated two small metadata files with a `.dvc` extension:
#
# - `dataset.tsv.dvc`
# - `model.pkl.dvc`

# %% [markdown]
# ### The `.dvc` files and the dvc cache
#
# Let's take a look at them:

# %%
# !cat dataset.tsv.dvc model.pkl.dvc

# %% [markdown]
# These are very small and simple files ! Believe it or not, but this is all the information [DVC](https://dvc.org) needs to track our `dataset.tsv` and `model.pkl` files !
#
# The quite long sequence of characters corresponding to the `md5` field is the computed hash value of `dataset.tsv` and `model.pkl`. This is the key mechanism behind [DVC](https://dvc.org). By computing the hash value of `dataset.tsv` and comparing this value to the one stored in `dataset.tsv.dvc`, [DVC](https://dvc.org) can tell whether a given file or directory has changed.
#
# Of course there is a lot of hidden complexity here. For the interrested reader, you can take a look at [this page](https://dvc.org/doc/user-guide/project-structure/internal-files#structure-of-the-cache-directory) to understand better how this works.
#
# Without going to much into the implementation details of [DVC](https://dvc.org), it is nice to understand that [DVC](https://dvc.org) is actualling caching our data in the hidden folder `.dvc/cache`:

# %%
# ! tree .dvc/cache/files/md5

# %% [markdown]
# We can actually see that [DVC](https://dvc.org) is using the first two characters of the hash values as folders and the rest as file names. We can verify the content corresponding to the hash value of our `dataset.tsv` file:

# %%
# ! head -n 3 .dvc/cache/files/md5/58/ac275459541571c78e9292635f63c8

# %% [markdown]
# In other words, the `.dvc` files we just generated act as "pointers" to the actual data that [DVC](https://dvc.org) cached.
#
# In addition to the `.dvc` files, [DVC](https://dvc.org) modified our [.gitignore](https://git-scm.com/docs/gitignore) file to tell [Git](https://git-scm.com) to **NOT** track `dataset.tsv` and `model.pkl`. This will prevent us to commit large data files by mistake.

# %%
# ! git diff .gitignore

# %% [markdown]
# As you can see from the output of the [dvc add](https://dvc.org/doc/command-reference/add) command above, [DVC](https://dvc.org) is helping us by telling us what we should do next ! It is telling us to version these files with git.
#
# Indeed, all these files are super small and easily managable for [Git](https://git-scm.com).
#
# Let's follow the recommandations of [DVC](https://dvc.org) and version them with [Git](https://git-scm.com):

# %%
# ! git add dataset.tsv.dvc model.pkl.dvc .gitignore

# %% [markdown]
# At this point we can commit our changes and optionally tag the commit with the [git tag](https://git-scm.com/docs/git-tag) command.
#
# Here, we will call this the "v1.0" of our experiment, for which we used 208 subjects:

# %%
# ! git commit -m "First model, trained with 208 subjects"
# ! git tag -a "v1.0" -m "model v1.0, 208 subjects"
# ! git log

# %% [markdown]
# Our workspace is clear, everything is tracked either with [Git](https://git-scm.com) or with [DVC](https://dvc.org):

# %%
# ! git status

# %% [markdown]
# ### Data can change
#
# So far so good, but datasets are not always fixed in time, they may evolve. For example, new data can be collected and added in new dataset releases (think about ADNI or UK-Biobank for example).
#
# Let's imagine that our small dataset received a brand new release with 208 additional subjects. Chances are that this will impact our previous experiment.
#
# How should we handle such an update ?
#
# Let's dive in and download the updated dataset:

# %%
# ! dvc get https://github.com/aramis-lab/dataset-registry NOW_2023/hc_volumes/version_2
# ! mv version_2/dataset2.tsv dataset.tsv
# ! rm -r version_2

# %% [markdown]
# As advertized, we now have data for 416 subjects:

# %%
df = pd.read_csv("dataset.tsv", sep="\t")
print(df.head(3))
print(df.tail(3))

# %% [markdown]
# Let's create a new instance of our toy model, and fit it with our new dataset:

# %%
model = ToyModel.from_dataframe(
    df,
    predictive_features=["HC_left_volume", "HC_right_volume"],
    target_feature="diagnosis",
)
model.fit()
model.save()
model.plot()

# %% [markdown]
# Great ! We now have our new model trained with our new data and the results look amazing !
#
# Let's check what happened to our workspace:

# %%
# ! git status

# %% [markdown]
# Wait, whaaat !?
#
# [Git](https://git-scm.com) is telling is that nothing changed... But we just re-ran our experiment, we should have new results !
#
# Remember that each tool has its responsibilities. We are using [Git](https://git-scm.com) to track our `.dvc` metadata files but we never changed them since our latest commit.
#
# Instead we changed `dataset.tsv` (we added 208 new subjects) and `model.pkl` (we re-trained and saved our toy model). These files aren't tracked with [Git](https://git-scm.com), but with [DVC](https://dvc.org) !
#
# This means that we should use a [DVC](https://dvc.org) command rather than a [Git](https://git-scm.com) command. Again, [DVC](https://dvc.org) makes it very easy for us because the command to use is [dvc status](https://dvc.org/doc/command-reference/status) which should look familiar:

# %%
# ! dvc status

# %% [markdown]
# As expected, [DVC](https://dvc.org) is telling us that both our dataset and our serialized model have changed.
#
# As we would do with [Git](https://git-scm.com), let's add these changes with [DVC](https://dvc.org):

# %%
# ! dvc add dataset.tsv model.pkl

# %% [markdown]
# Now we changed the `.dvc` metadata files ! Let's verify this:

# %%
# ! git status
# ! git diff dataset.tsv.dvc

# %% [markdown]
# Alright, the hash value changed and the size of the `dataset.tsv` file more or less doubled which is expected since we added 208 new subjects to the previous 208.
#
# Note that, although our data just doubled in size, the metadata that we are tracking with [Git](https://git-scm.com) didn't change at all, we still have a `.dvc` file with 5 lines !
#
# However there is no magic here. The data tracked with [Git](https://git-scm.com) didn't change much in size, but our cache did change:

# %%
# ! tree .dvc/cache/files/md5

# %% [markdown]
# The cache contains both version of our data:

# %%
# ! wc -l  .dvc/cache/files/md5/58/ac275459541571c78e9292635f63c8
# ! wc -l  .dvc/cache/files/md5/41/7fedd2a5927bfc5284ddc01b74ad2a

# %% [markdown]
# For projects with long history, the cache growth can become an issue. Although this is beyond the scope of this tutorial, note that [DVC](https://dvc.org) has some commands like [dvc gc](https://dvc.org/doc/command-reference/gc) which allows you to clean old data entries.

# %% [markdown]
# OK, this is the part where we use [Git](https://git-scm.com) ! And, again, [DVC](https://dvc.org) is helping us by suggesting our next move !
#
# Let's add the changes to the `.dvc` files with [Git](https://git-scm.com):

# %%
# ! git add dataset.tsv.dvc model.pkl.dvc

# %% [markdown]
# We can now commit these changes and tag this commit with a "v2.0" tag:

# %%
# ! git commit -m "Second model, trained with 416 subjects"
# ! git tag -a "v2.0" -m "model v2.0, 416 subjects"
# ! git log

# %% [markdown]
# And again, our workspace is clean:

# %%
# ! git status

# %% [markdown]
# ### Going back in time
#
# OK, this is great ! We can update our datasets, models, and code files and commit all these changes in a linear clean history. But can we navigate this timeline ?
#
# Let's imagine that we submitted a paper with the results of our first experiment (the model trained on 208 subjects only). Time passed and we made our latest experiment with the 208 new subjects. We have everything commited and clean, but we finally receive the review for our paper.
#
# Unfortunately, we got some work to do if we wish to publish... The reviewer is asking us to add some more details on some of our plots.
#
# Well, that's easy ! Let's update the plotting code, create a new model instance, fit the model, and plot the results. But wait, our dataset now consists of 416 subjects, this will change our figure and contradict the experimental setups described in our paper.
#
# Clearly, we cannot use the new dataset and model to make the new plots. We need to remove the added subjects, but which ones were added already ? It's been a long time and we have forgotten the labels of the initial subjects...
#
# Without our setup, this could easily become a nightmare ! (Of course this toy example case is easy but in more realistic scenarios, and months after having performed the initial experiment, it can clearly be !)
#
# We need to go back to the state we were in when we generated the plots of the paper we submitted. This means both code **AND** data. 
#
# Fortunately for us, this is very easy to do thanks to both [Git](https://git-scm.com) and [DVC](https://dvc.org).
#
# In our example, we were smart enough to tag the commit corresponding to the first version of our experiment with the "v1.0" tag (we could have used something more informative like "publication for journal XXX"). 
#
# Lets' use the `git checkout` command with the tag label as a parameter to navigate to this commit:

# %%
# ! git checkout v1.0

# %% [markdown]
# [Git](https://git-scm.com) is telling us that we are in "detached HEAD" state, which could sound like a scary thing, but it only means that we are on a commit that doesn't have a branch.
#
# Let's see what the [git status](https://git-scm.com/docs/git-status) command tells us:

# %%
# ! git status

# %% [markdown]
# Not much, everything seems to be clean... Let's check how many lines we have in our dataset:

# %%
# ! wc -l dataset.tsv

# %% [markdown]
# Hmmm, not exactly what we want. We still have 416 subjects in our dataset, but why ?
#
# Because [Git](https://git-scm.com) isn't responsible for managing your dataset, [DVC](https://dvc.org) is !
#
# [Git](https://git-scm.com) did its job just right by restoring the metadata files:

# %%
# ! cat dataset.tsv.dvc

# %% [markdown]
# The `.dvc` file points to the first `dataset.tsv` version but the file in our workspace was not updated. You need to use [DVC](https://dvc.org) to perform this update:

# %%
# ! dvc checkout

# %% [markdown]
# Let's verify that we didn't get fooled again:

# %%
# ! wc -l dataset.tsv

# %% [markdown]
# Awesome ! We have the 208 initial subjects !
#
# We are now in the exact same state as before the data update:

# %%
df = pd.read_csv("dataset.tsv", sep="\t")
print(len(df))  # Dataset has 500 subjects

# %% [markdown]
# We could now update our plotting code, re-train our model, and re-do our plot to satisfy the reviewer !

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
