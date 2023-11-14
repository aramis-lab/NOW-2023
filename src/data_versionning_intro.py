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
#
# ## Why DVC ?
#
# - easy to install
# - dvc commands are very similar to the git commands

# %% [markdown]
# ## What is DVC ?
#
# [DVC](https://dvc.org) is a tool for data science that takes advantage of existing software engineering toolset. It helps machine learning teams manage large datasets, make projects reproducible, and collaborate better.
#
# [DVC](https://dvc.org) does not replace or include [Git](https://git-scm.com). You must have [Git](https://git-scm.com) in your system to enable important features such as data versioning and quick experimentation.
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
# <img src="https://dvc.org/static/d40892521e2fff94dac9e59693f366df/958d8/data-ver-complex.png" style="height: 400px;">
#
# The need for versioning tool can often be seen through the proliferation of suffixes in the file / folder names. These suffixes usually convey information about:
#
# - time (date, version number, revision number...)
# - author
# - specific milestones (a publication, a conference / workshop...)
#
# This primitive style of versioning reaches its limits very quickly for diverse reasons:
#
# - the number of files and directories explodes very fast in non trivial cases
# - the project becomes a mess very fast as people forget what the different names actually mean
# - the project total weight quickly explodes as everything is copy-pasted
# - there is no history / lineage, only snapshots in total disorder
#
# #### The cleaner way
#
# <img src="https://dvc.org/static/39d86590fa8ead1cd1247c883a8cf2c0/cb690/project-versions.png" style="height: 500px;">
#
# Data Version Control lets you capture the versions of your data and models in Git commits, while storing them on-premises or in cloud storage. It also provides a mechanism to switch between these different data contents. The result is a single history for data, code, and ML models that you can traverse.
#
# As you use [DVC](https://dvc.org), unique versions of your data files and directories are cached in a systematic way (preventing file duplication). The working data store is separated from your workspace to keep the project light, but stays connected via file links handled automatically by [DVC](https://dvc.org).
#
# The benefits of this approach include:
#
# - **Lightweight:** [DVC](https://dvc.org) is a free, open-source command line tool that doesn't require databases, servers, or any other special services.
# - **Consistency:** Projects stay readable with stable file names — they don't need to change because they represent variable data. No need for complicated paths like `data/20190922/labels_v7_final` or for constantly editing these in source code.
# - **Efficient data management:** Use a familiar and cost-effective storage solution for your data and models (e.g. SFTP, S3, HDFS, etc.) — free from Git hosting constraints. [DVC](https://dvc.org) optimizes storing and transferring large files.
# - **Collaboration:** Easily distribute your project development and share its data internally and remotely, or reuse it in other places.
# - **Data compliance:** Review data modification attempts as Git pull requests. Audit the project's immutable history to learn when datasets or models were approved, and why.
# - **GitOps:** Connect your data science projects with the Git ecosystem. Git workflows open the door to advanced CI/CD tools (like CML), specialized patterns such as data registries, and other best practices.
#
# [DVC](https://dvc.org) also supports multiple advanced features out-of-the-box: Build, run, and versioning data pipelines, manage experiments effectively, and more.

# %%
