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
# # Chapter 1 : Introduction to data versioning
#
# In this introduction, we will present some concepts behind data versioning and introduce some tools enabling to perform related taks. Finally, we will focus on [DVC](https://dvc.org) which is the data versioning tool we will use in the rest of the tutorial.

# %% [markdown]
# ## What is data versioning ?
#
# TODO
#
# ## What are the tools ?
#
# TODO

# %% [markdown]
# ## What is DVC ?
#
# [DVC](https://dvc.org) is a tool for data science that takes advantage of existing software engineering toolset. It helps machine learning teams manage large datasets, make projects reproducible, and collaborate better.
#
# [DVC](https://dvc.org) does not replace or include Git. You must have git in your system to enable important features such as data versioning and quick experimentation.
#
# [DVC](https://dvc.org) is written in Python and can be esily installed on the most common OS (Linux, MacOS, and Windows). We will see how to install [DVC](https://dvc.org) in the next chapter.
#
# ### What is the target users of DVC ?
#
# [DVC](https://dvc.org) targets people who needs to store and process data files or datasets to produce other data or machine learning models. In other words, anyone willing to:
#
# - track and save data and machine learning models the same way they capture code
# - create and switch between versions of data and ML models easily
# - understand how datasets and ML artifacts were built in the first place
# - compare model metrics among experiments
# - adopt engineering tools and best practices in data science projects
#
# ### Versionning data and models
#
# #### The dirty way
#
# <img src="https://github.com/aramis-lab/NOW-2023/blob/main/images/data_versioning_old_way.png" style="height: 200px;">
#
# #### The cleaner way

# %%
