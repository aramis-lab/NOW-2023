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
# # Chapter 1: Introduction to code versioning
#
# The objective of this chapter is to give a brief overview of [Git](https://git-scm.com) and the main concepts behind its basic usage.
#
# We will use these concepts in the next chapter throughout the development of a small Python library.
#
# ```{note}
# The content of this chapter has been adapted from a Git tutorial made by *Mathilde Lannes* and *Séverine Candelier* from the ICM.
# ```

# %% [markdown]
# ## What is git?
#
# - [Git](https://git-scm.com) is a version-control system. The purpose of [Git](https://git-scm.com) is to manage a code project or a set of files, as they **change over time**.
# - [Git](https://git-scm.com) stores this information in a data structure called a **repository**.
# - The [Git](https://git-scm.com) repository is stored in the same directory as the project itself, in a hidden subdirectory called `.git`.

# %% [markdown]
# ## Why do you need git?
#
# - Version control softwares keep track of every modification to the code.
# - In case of an error, you can go back in time.
# - A whole team can work on the same project.
# - No fear of "deleting everything" by accident.
# - Your code is constantly backed up on a server.
# - You can work on your projects on any number of computers.
# - Every company uses a version control system.

# %% [markdown]
# ## Local vs. remote
#
# There is a difference between:
#
# - What is in your online repository (remote)?
# - What is in your local repository?
#
# There won't be any modification on the remote server ([Github](https://github.com), [Gitlab](https://gitlab.com)...) until you publish it.
#
# ```{note}
# Note that you can have multiple remotes and multiple local copies of the same repository.
# ```

# %% [markdown]
# ## What is a commit?
#
# It is a snapshot of modified code.

# %%
from IPython.display import Image
Image(filename="../images/git-commit.png")

# %% [markdown]
# ## Git workflow

# %%
Image(filename="../images/git-workflow.png")

# %% [markdown]
# - Working directory: files in your working directory that you can normally access and edit.
# - Staging Area: a temporary area to store all the modifications for the next commit.
# - HEAD: A reference to the version of the last commit in the local repository.

# %% [markdown]
# ### Add to the staging area
#
# This is done with the [git add](https://git-scm.com/docs/git-add) command:
#
# ```
# $ git add myfile
# ```
#
# ### Add staged changes to a commit
#
# This is done with the [git commit](https://git-scm.com/docs/git-commit) command:
#
# ```
# $ git commit -m "My commit message"
# ```
#
# ### Check where you are
#
# You can get information with the [git status](https://git-scm.com/docs/git-status) command. This command will never hurt you but will probably save you a lot, so do not hesitate to use it often !
#
# ```
# $  git status
# ```
#
# You can compare things with the [git diff](https://git-scm.com/docs/git-diff) command. The arguments of the command depend on what you want to compare:
#
# - changes between the working tree and the index 
# - changes between two trees
# - changes resulting from a merge
# - changes between two blob objects
# - changes between two files on disk
#
# ```
# $ git diff
# ```

# %% [markdown]
# ### Publish commits
#
# You can publish your local work to a remote with the [git push](https://git-scm.com/docs/git-push) command. When you push, your local version becomes the version of the remote:
#
# ```
# $ git push
# ```
#
# ### Getting the latest commits from your co-workers
#
# You can retrieve locally what is on a remote with the [git pull](https://git-scm.com/docs/git-pull) command. When you pull, you are merging the version of the remote inside your local repository:
#
# ```
# $ git pull
# ```

# %% [markdown]
# ### Cancel what you have done
#
# You can change a modified file in the working directory to the HEAD version with the [git restore](https://git-scm.com/docs/git-restore) command and the file name as argument:
#
# ```
# $ git restore myfile
# ```
# You can still use the [git checkout](https://git-scm.com/docs/git-checkout) command if you're used to.

# %% [markdown]
# You can create a commit to undo your last commit(s) with the [git revert](https://git-scm.com/docs/git-revert) command:
#
# ```
# $ git revert
# ```

# %% [markdown]
# ## The concept of branches
#
# Branches are used to develop features isolated from each other.
#
# The `main` (previously `master`) branch is the "default" branch when you create a repository.
#
# ### How to use branches
#
# To create a new branch, you can use the [git switch](https://git-scm.com/docs/git-switch) command with the `-c` option:
#
# ```
# $ git switch -c mybranch
# ```

# %%
Image(filename="../images/git-branch-1.png")

# %% [markdown]
# You can list the different branches and verify on which branch you are with the [git branch](https://git-scm.com/docs/git-branch) command:
#
# ```
# $ git branch
# ```
#
# Once you are on a branch, you can work and make commits which will be added to the branch. If you move to another branch, you won't have the commits you only made in the first branch.

# %%
Image(filename="../images/git-branch-2.png")

# %% [markdown]
# You can publish your branch on a remote (named `origin` here):
#
# ```
# $ git push origin mybranch
# $ git push --set-upstream origin mybranch
# ```
#
# However, the other branches can continue to evolve while you are working on your new branch. In this case, the branches are diverging:

# %%
Image(filename="../images/git-branch-3.png")

# %% [markdown]
# ## What is a merge
#
# Merging is automatic if different portions of the code have been modified.
#
# However, you must resolve the conflicts if the SAME portions of code have been touched.
#
# You can perform a merge with the [git merge](https://git-scm.com/docs/git-merge) command:
#
# ```
# $ git switch main
# $ git merge mybranch
# ```

# %%
