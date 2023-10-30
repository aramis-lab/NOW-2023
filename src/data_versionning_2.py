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
# ! which dvc

# %%
