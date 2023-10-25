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
def f(x: int) -> int:
    return x ** 2


# %% [markdown]
# This is a **test cell** :

# %%
print(f"Results = {f(10)}")

# %%

# %% [markdown]
# Another test cell :

# %%
f(f(10))

# %%
