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
# # Part 2 : Code versionning using DVC
# This is the second part of the tutorial on code and data versionning using [GitHub](https://docs.github.com/) and [DVC](https://dvc.org).
# Start a small Python project
## I. Initialize the project


# %% [markdown]
# Create a directory `dummymaths`:
# %%
!mkdir dummymaths
!cd dummymaths
# %% [markdown]
# Initialize a Git local repository in the `dummymaths` directory:
# %%
!git init
# %% [markdown]
# Add a `README.md` file and then check the state of your local copy:
# vérifiez l'état de votre copie locale:
# %%
!git status
# %% [markdown]
# Add the README.md file to the staging area and check again the state of your local copy:
# %%
!git add .
# %% [markdown]
# or, more explicitly:
# %%
!git add README.md
# %% [markdown]
# Edit the `README.md` file (for example, add a title, a short description of this small Python project), save and check again the state of your local copy. You should have changes both in the staging area and in the working directory (local changes). Display the changes, first in the staging area and then the local changes:
# %%
!git diff --staged
!git diff
# %% [markdown]
# Commit all changes in the `README.md` file (both in staging and local) and check one last time the state of the local copy:
# %%
!git add README.md
!git commit -m "initial commit"
# %% [markdown]
# ## II. Ajouter le fichier myfuncs.py

# Add the file `myfuncs.py` with the following content:
# %%
"""Some useless mathematical utility functions."""

def add(a, b):
    """Return the sum of a and b."""
    return a + b

def sub(a, b):
    """Substract b from a."""
    return a - b


# %% [markdown]
# Commit this file:
# %%
!git add myfuncs.py
!git commit -m "initial version of myfuncs.py"
# %% [markdown]
# ## III. Ajouter le fichier `test_myfuncs.py` et lancer les tests
# Add the file `test_myfuncs.py` with the following content:
# %%python
import pytest
from myfuncs import add

@pytest.mark.parametrize(
    "a,b,res",
    [
        (0, 0, 0),
        (0, 42, 42),
        (42, 0, 42),
        (42, -42, 0),
        (-42, 42, 0),
    ]
)
def test_add(a, b, res):
    assert add(a, b) == res
# %% [markdown]
# Use `pytest` (install it using `pip install pytest`) to run the tests, verify that they pass and then commit `test_myfuncs.py` (and only this one!):
# %%
!pytest -v
!git add test_myfuncs.py 
!git commit -m "tests: add test function for add"
# %% [markdown]

## IV. Ignore generated files
# At this stage, they are Python bytecode generated files displayed when running `git status`. And we don't want to commit them inadvertently: this is the purpose of the `.gitignore` file.
# Add the `.gitignore` to the base directory of your working copy with the following content:
# %%
!*pyc
# %% [markdown]
# Check that bytecode generated files are not listed anymore when running `git status`.
# Commit the `.gitignore` file:
# %%
!git add .gitignore
!git commit -m "ignore Python generated files"
# %%
