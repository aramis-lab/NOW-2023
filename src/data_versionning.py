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

# %%
