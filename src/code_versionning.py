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
# # Chapter 2 : Code versioning using Git
#
# In this chapter, we will install [Git](https://git-scm.com) and take a look at how it works and how to version code.
#
# ```{note}
# The content of this chapter has been adapted from a tutorial made by *Alexandre Abadie* and *Kim Tâm Huynh* from the [INRIA SED of Paris](https://sed.paris.inria.fr).
# ```
#
# ## Getting started
#
# First you will start a small Python project
# %% [markdown]
# ### Initialize the project
# %% [markdown]
# Create a directory `dummymaths`:
# %%
# ! mkdir dummymaths
# %cd dummymaths
# %% [markdown]
# Initialize a [Git](https://git-scm.com) local repository in the `dummymaths` directory with the [git init](https://git-scm.com/docs/git-init) command:
# %%
# ! git init

# %% [mardown]
# Configure your git account using simple configuration files
# 3 levels:
#   * __local__ : `<local copy>/.git/config`, by default
#   * __global__ : `~/.gitconfig`, option `--global`
#   * __system__ : `/etc/gitconfig`, option `--system`

# Either edit the files directly, or use `git config` command
# %% [markdown]
# To configure your user name and email address, you need to use the [git config](https://git-scm.com/docs/git-config) command:

# %%
# ! git config --local user.name "John Doe"
# ! git config --local user.email john.doe@inria.fr
# %% [markdown]
# Create a `README.md` file and then check the state of your local copy:
# %%
# ! echo "This is the README file" > README.md
# ! git status
# %% [markdown]
# Add the `README.md` file to the staging area and check again the state of your local copy:
# %%
# ! git add README.md
# ! git status
# %% [markdown]
# Edit the `README.md` file (for example, add a title, a short description of this small Python project), save and check again the state of your local copy. 
#
# You should have changes both in the staging area and in the working directory (local changes). Display the changes, first in the staging area and then the local changes:
# %%
# ! echo "This is the new README file" > README.md

# %%
# ! git diff --staged
# ! git diff
# %% [markdown]
# Commit all changes in the `README.md` file (both in staging and local) and check one last time the state of the local copy:
# %%
# ! git add README.md # staging
# ! git commit -m "initial commit" # local
# ! git status
# %% [markdown]
# ### Add a python file to the project
#
# Add the file `myfuncs.py` with the following content:
# %%
# %%writefile myfuncs.py

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
# ! git add myfuncs.py
# ! git commit -m "initial version of myfuncs.py"
# %% [markdown]
# ### Add a testing file with pytest
# Add the file `test_myfuncs.py` with the following content:
# %%
# %%writefile test_myfuncs.py

import pytest

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
    from myfuncs import add

    assert add(a, b) == res
# %% [markdown]
# Use `pytest` (install it using `pip install pytest`) to run the tests, verify 
# that they pass and then commit `test_myfuncs.py` (and only this one!):
# %%
# ! pip install pytest

# %%
# ! pytest test_myfuncs.py
# %% [markdown]
# Note that you can use the verbose option (`-v`) to have more information:

# %%
# ! pytest -v

# %% [markdown]
# Let's commit these tests:

# %%
# ! git add test_myfuncs.py 
# ! git commit -m "tests: add test function for add"
# %% [markdown]
# ### Ignore generated files
#
# At this stage, they are Python bytecode generated files displayed when running [git status](https://git-scm.com/docs/git-status):
# %%
# ! git status

# %% [markdown]
# We don't want to commit them inadvertently: this is the purpose of the `.gitignore` file.
#
# Add the `.gitignore` to the base directory of your working copy with the following content:

# %%
# ! echo "*pyc" > .gitignore
# %% [markdown]
# Check that bytecode generated files are not listed anymore when running [git status](https://git-scm.com/docs/git-status):
# %%
# ! git status

# %% [markdown]
# Commit the `.gitignore` file:
# %%
# ! git add .gitignore
# ! git commit -m "ignore Python generated files"

# %% [markdown]
# ## Manage the changes
#
# Let's continue with the `dummymaths` Python project and check the history of changes there.
#
# 1. Display the history of changes already committed, using `git log`:
#   * Only the last 2 changes with `-2` flag along with their corresponding differences using `-p` flag
#   * Display the commit information with the format `<small hash> - <message> - <date> - <email>` (check the help of log)

# %%
# ! git log

# %%
# ! git log -2

# %%
# ! git log -2 -p

# %% [markdown]
# 2. Let's extend the tests in `test_myfuncs.py` with a test for the `sub` function:

# %%
# %%writefile -a test_myfuncs.py

@pytest.mark.parametrize(
    "a,b,res",
    [
        (0, 0, 0),
        (0, 42, -42),
        (42, 0, 42),
        (42, 42, 0),
        (42, -42, 84),
        (-42, 42, -84),
    ]
)
def test_sub(a, b, res):
    from myfuncs import sub

    assert sub(a, b) == res


# %%
# ! pytest -v

# %% [markdown]
# Add these changes to the staging area:
# %%
# ! git add test_myfuncs.py

# %% [markdown]
# Check the state of your local copy with [git status](https://git-scm.com/docs/git-status): there's something in the staging area and nothing in the local changes.
# %%
# ! git status

# %%
# ! git diff  # This command should return nothing

# %% [markdown]
# Verify that the changes added in the staging are the ones expected:
# %%
# ! git diff --staged
# %% [markdown]
# Remove the changes from the staging area:
# %%
# ! git reset
# %% [markdown]
# Check the modifications are still there, in the local changes:
# %%
# ! git status
# %%
# ! git diff

# %% [markdown]
# Repeat 4. and 5. but this time completely revert the changes added to the staging area (`git reset --hard`)
# %% [markdown]
# Apply one last time the changes above to the `test_myfuncs.py` file and commit them:
# %%
# ! git add test_myfuncs.py
# ! git commit -m "add test function for sub"
# %% [markdown]
# Check the diff contained in this last commit:
# %%
# ! git log -1 -p

# %% [markdown]
# ## Working with remote repositories

# %% [markdown]
# Some preliminary checks:
# In your local working copy, check that no remote repository is already configured:
# %%
# ! git remote
# %% [markdown]
# Move to another directory, out of the `dummymaths` one, and initialize there a bare repository. We will use it as a remote repository for `dummymaths`
# %%
# %cd ..
# ! mkdir dummymaths_remote
# %cd dummymaths_remote
# ! git init --bare
# %% [markdown]
# Move back to the `dummymaths` directory, that contains your initial git working copy and from there add the newly created remote repository. The url of this repository is just a path in your filesystem:
# %%
# %cd ../dummymaths
# ! git remote add origin ../dummymaths_remote
# %%
# ! git remote -v

# %% [markdown]
#  Push your `main` branch and enable upstream tracking in the meantime:
# %%
# ! git push origin main -u
# %% [markdown]
# Check that the `main` branch is now referenced on the remote repository:
# %%
# ! git remote show origin
# %% [markdown]
# In another directory, clone the repository of the source code of the tutorial that is hosted on gitlab:
# %%
# %cd ..
# ! git clone https://gitlab.inria.fr/git-tutorial/git-tutorial.git
# %% [markdown]
# You can check the status of your local copy and the information about the remote repository:
# %%
# %cd git-tutorial
# ! git status
# ! git remote -v

# %% [markdown]
# Manipulating branches

# %% [markdown]
# Move back to the `dummymaths` directory and list your local and remote branches:
# %%
# %cd ../dummymaths
# ! git branch
# %%
# ! git branch -a

# %% [markdown]
# Create a branch called `multiply` and list the branches again:
# %%
# ! git branch multiply
# ! git branch
# %% [markdown]
# Current branch is still `main` but there's a new `multiply` branch. Also note
# how immediate it is to create a new branch.
# %% [markdown]
# Switch to the `multiply` branch and list the local branches again:
# %%
# ! git checkout multiply
# ! git branch
# %% [markdown]
# Now let's display the history of commits on both branches:
# %%
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# You can also try with a graphical tool, such as `gitk` using `gitk --all`
# %% [markdown]
# Let's add a new multiply function to the `myfuncs.py` module:
# %%
# ! cat myfuncs.py

# %%
# %%writefile -a myfuncs.py

def multiply(a, b):
    """Multiply a by b."""
    return a * b


# %% [markdown]
# Commit the changes above, they should end up in the `multiply` branch, and
# display the history of changes, like before:
# %%
# ! git commit -am "myfuncs: add the multiply function"
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# The `multiply` branch is now one commit ahead of `main`.
# %% [markdown]
# Now switch back to the `main` branch:

# %%
# ! git checkout main

# %% [markdown]
# And add a test function to `test_myfuncs.py` to test our new multiply function:

# %%
# %%writefile -a test_myfuncs.py

@pytest.mark.parametrize(
    "a,b,res",
    [
        (0, 0, 0),
        (0, 42, 0),
        (42, 0, 0),
        (42, 1, 42),
        (1, 42, 41),
        (-1, 42, -42),
    ]
)
def test_multiply(a, b, res):
    from myfuncs import multiply
    
    assert multiply(a, b) == res
# %% [markdown]
# Finally, commit the changes above and display the branch history:
# %%
# ! git commit -am "tests: add test function for multiply"
# ! git log --decorate --graph --oneline --all

# %% [markdown]
# The branches are diverging
# Let's see how to merge them and to fix potential conflicts.

# %% [markdown]
# ## Merging branches
#
# Back to the `dummymaths` Python project, let's create several branches and
# merge them in the `main` branch. For this, we'll imagine a (non-realistic)
# scenario to write a function implementing division along with a test function.
#
# ### The fast-forward merge

# %% [markdown]
# Let's create the `divide`and the `todo` branches from `main`, which is your current branch:
# %%
# ! git branch divide
# %%
# ! git checkout -b todo

# %%
# ! git branch

# %%
# ! git status

# %% [markdown]
# You are now on the `todo` branch. Edit the README.md file and add the following content at the end:
# %%
# %%writefile -a README.md

## TODO
Add _divide_ function
# %%
# ! cat README.md

# %% [markdown]
# Commit the change above:
# %%
# ! git commit -am "README.md: bootstrap todo section with a divide function item"
# ! git status
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# The `todo` is now one commit ahead of `main`
# %% [markdown]
# Switch back to `main` and merge `todo` in main:
# %%
# ! git checkout main
# %%
# ! git merge todo

# %% [markdown]
# Git will just "fast-forward `main` to `todo`
# %%
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# As good citizens, now that the `todo` branch is not needed anymore, let's remove 
# it:
# %%
# ! git branch -d todo
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# ### The merge commit case
#
# The `multiply` and `main` branches are currently diverging. Normally the changes introduced in `multiply` are separate enough from the changes added to `main` such that merging `multiply` in `main `should not conflict.
#
# Merge the `multiply` branch into `main`:
# %%
# ! git merge multiply --no-edit
# %% [markdown]
# The merge command created a merge commit:

# %%
# ! git log --decorate --graph --oneline --all

# %% [markdown]
# ### The conflict!
#
# Now let's trigger a conflict on purpose by finally switching to the `divide` branch:
# %%
# ! git checkout divide
# %% [markdown]
# Add the missing `divide` function to the `myfuncs.py` module:
# %%
# %%writefile -a myfuncs.py

def divide(a, b):
      """Divide a by b."""
      try:
          return a / b
      except ZeroDivisionError:
          return None


# %% [markdown]
# And commit that change:
# %%
# ! git commit -am "myfuncs: add divide function"
# %% [markdown]
# Add the related test function to the `test_myfuncs.py` module:
# %%
# %%writefile -a test_myfuncs.py

@pytest.mark.parametrize(
      "a,b,res",
      [
          (0, 0, None),
          (0, 42, 0),
          (42, 0, None),
          (42, 1, 42),
          (1, 2, 0.5),
          (-1, 2, -0.5),
      ]
)
def test_divide(a, b, res):
    from myfuncs import divide
      
    assert divide(a, b) == res

# %% [markdown]
# And commit that change:
# %%
# ! git commit -am "tests: add test function for divide"
# %% [markdown]
# Now try to merge the `divide` branch in `main`:
# %%
# ! git checkout main
# %%
# ! git merge divide --no-edit

# %% [markdown]
# Try to solve the conflict! 2 possibilities:
#
# 1. Manually:
#     
# First, fix the conflict by manually editing the file:
# %%
# ! cat myfuncs.py

# %% [markdown]
# As you can see, Git is showing us the conflicting portions of the code. In our case, the `main` branch contains the `multiply` function while the `divide` branch contains the `divide` function.
#
# Solving the conflict in this situation is quite easy. Since we want to keep both functions, we simply need to remove the lines added by Git, save the file, and add it:

# %%
# ! sed -i '' -e 's#<<<<<<< HEAD##' myfuncs.py
# ! sed -i '' -e 's#>>>>>>> divide##' myfuncs.py
# ! sed -i '' -e 's#=======##' myfuncs.py

# %%
# ! cat myfuncs.py

# %% [markdown]
# Seems clean enough ! Let's add it and continue the merge process:

# %%
# ! git add myfuncs.py

# %%
# ! git commit --no-edit

# %%
# ! git log --decorate --graph --oneline --all

# %% [markdown]
# 2. Using a graphical tool:

# %%
# Not demoed since we already fix the conflict
# Also this interactive mode does not play well within a notebook.
#
# # ! git mergetool
# # ! git add myfuncs.py
# # ! git commit
# %% [markdown]
# ### One last thing
#
# Once all features are merged, it's time to sync your local `main` branch with
# the remote repository:

# %%
# ! git remote -v

# %%
# ! git push origin main
# %% [markdown]
# Also, now that the `multiply` and `divide` branches are not needed anymore, you
# can delete them:

# %%
# ! git branch -d multiply divide


# %% [markdown]
# ## Simple rebasing
#
# Let's again extend the `dummymaths` Python project with a `power` function.
#
# Check that your current branch is `main`:

# %%
# ! git status
# %% [markdown]
# Create a new `power` branch and switch to it:
# %%
# ! git checkout -b power
# %% [markdown]
# Extend the `myfuncs.py` module with the `power` function :
# %%
# %%writefile -a myfuncs.py

def power(a, b):
    """Return a power b."""
    return a ** b


# %% [markdown]
# Commit your change:
# %%
# ! git commit -am "myfuncs: add power function"
# %% [markdown]
# Add the related test function to the `test_myfuncs.py` module:
# %%
# %%writefile -a test_myfuncs.py

@pytest.mark.parametrize(
    "a,b,res",
    [
        (0, 0, 1),
        (0, 2, 0),
        (1, 2, 1),
        (2, 0, 1),
        (2, 1, 2),
        (2, 2, 4),
        (2, -1, 0.5),
    ]
)
def test_power(a, b, res):
    from myfuncs import power

    assert power(a, b) == res

# %% [markdown]
# Commit your change:
# %%
# ! git commit -am "tests: add test function for power"
# %% [markdown]
# Switch back to `main` and remove the `TODO` section from the README (the divide function is merged already!).
# %%
# ! git checkout main

# %%
# ! echo "This is the new README file" > README.md

# %%
# ! git add README.md
# ! git commit -m "README: remove TODO section"
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# At this point, the `main` and `power` branches have diverged.
# Switch back to `power` and rebase it on top of `main`:
# %%
# ! git checkout power
# %%
# ! git rebase main

# %%
# ! git log --decorate --graph --oneline --all

# %% [markdown]
# Merge power into `main`:
# %%
# ! git checkout main
# %%
# ! git merge power --no-edit

# %%
# ! git log --decorate --graph --oneline --all

# %% [markdown]
# This is a fast-forward move of `main` towards `power` !
#
# Finally push `main` and delete `power`:
# %%
# ! git push origin main
# ! git branch -d power

# %% [markdown]
# ## Using bisect to find bugs
#
# The `dummymaths` Python project provides a set of unit tests. In the first
# exercise you used `pytest` to run them.
# Since then, new code was added but you didn't rerun `pytest`.

# %% [markdown]
# Let's start by running `pytest`. It should fail. If it doesn't this exercise
# becomes useless!
# %%
# ! pytest -v
# %% [markdown]
# Even if the problem is obvious, you'll use `git bisect` to find the commit that introduced the problem. In some more complex cases, that can help understand what is the origin of the bug.
#
# Before starting bisect, you must find a commit that works.
#
# We know one: the one that contains the "add" function test. Let's use `git log` with some special parameters to find it:
# %%
# ! git log --oneline --grep "tests: add test function for add"
# %% [markdown]
# The reply contains the short hash of the matching commit that you'll use as
# good commit.
# %% [markdown]
# Let's now start bisecting:
# %%
# ! git bisect start
# %%
# ! git bisect bad

# %% [markdown]
# You now have to switch the commit that we think is good, test it and tell git:
# %%
# Change the hash value to the one in the output of the cell above
# ! git checkout 6ed5a35
# %%
# ! pytest -v

# %%
# ! git bisect good

# %% [markdown]
# Since `pytest` is the command that is used to check if a commit is good or
# not, you can run it at each step:
# %%
# ! git bisect run pytest
# %%
# ! git bisect reset

# %% [markdown]
# This command should tell you quite fast what is the first bad commit. Check
# it's content:
# %%
# Change the hash value to the one in the output of the cell above
# ! git show ba855a2
# %% [markdown]
# Point 5. reveals that the problem comes from one the multiply test case.
#
# Let's fix it:
# %%
# ! sed -i '' -e 's#(1, 42, 41),#(1, 42, 42),#' test_myfuncs.py

# %%
# ! pytest -v

# %% [markdown]
# Commit your changes and push the `main` branch:
# %%
# ! git add test_myfuncs.py
# ! git commit -am "tests: fix multiply test case"
# ! git push origin main


# %% [markdown]
# Cleaning:

# %%
# Cleaning...
# %cd ..
# ! rm -rf dummymaths_remote/ dummymaths/ git-tutorial/

# %%
