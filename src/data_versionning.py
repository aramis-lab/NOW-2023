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
# This is the second part of the tutorial on data versionning using [DVC](https://dvc.org).
#
# First, we need to install DVC:

# %%
# ! pip install dvc

# %% [markdown]
# We can check that DVC is installed:

# %%
# ! which dvc

# %%
# ! dvc --version

# %%
def g(x: str) -> str:
    return x + x


# %%
g("test")

# %%
