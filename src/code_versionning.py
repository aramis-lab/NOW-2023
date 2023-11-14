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

# In this chapter, we will install Git and take a look at how it works and how to version code.
# # Getting started
# First you will start a small Python project
# %% [markdown]
# ## Initialize the project
# %% [markdown]
# Create a directory `dummymaths`:
# %%
!mkdir dummymaths
%cd dummymaths
# %% [markdown]
# Initialize a Git local repository in the `dummymaths` directory:
# %%
!git init

# %% [mardown]
# Configure your git account using simple configuration files
# 3 levels:
#   * __local__ : `<local copy>/.git/config`, by default
#   * __global__ : `~/.gitconfig`, option `--global`
#   * __system__ : `/etc/gitconfig`, option `--system`

# Either edit the files directly, or use `git config` command
# %%
!git help config
!git config --global user.name "First Last"
!git config --global user.email your.email@organisation.com
# %% [markdown]
# Add a `README.md` file and then check the state of your local copy:
# vérifiez l'état de votre copie locale:
# %%
!echo "This is the README file" > README.md
!git status
# %% [markdown]
# Add the README.md file to the staging area and check again the state of your local copy:
# %%
!git add README.md
!git status
# %% [markdown]
# Edit the `README.md` file (for example, add a title, a short description of this small 
# Python project), save and check again the state of your local copy. You should have changes 
# both in the staging area and in the working directory (local changes). Display the changes, 
# first in the staging area and then the local changes:
# %%
!echo "This is the new README file" > README.md

# %%
!git diff --staged
!git diff
# %% [markdown]
# Commit all changes in the `README.md` file (both in staging and local) and check one last time the state of the local copy:
# %%
!git add README.md # staging
!git commit -m "initial commit" # local
!git status
# %% [markdown]
# ## Add a python file to the project
# Add the file `myfuncs.py` with the following content:
# %%
%%writefile myfuncs.py

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
# ## Add a testing file with pytest
# Add the file `test_myfuncs.py` with the following content:
# %%
%%writefile test_myfuncs.py
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
# Use `pytest` (install it using `pip install pytest`) to run the tests, verify 
# that they pass and then commit `test_myfuncs.py` (and only this one!):
# %%
!pip install pytest
!pytest test_myfuncs.py
# %%
!pytest -v
!git add test_myfuncs.py 
!git commit -m "tests: add test function for add"
# %% [markdown]

## IV. Ignore generated files
# At this stage, they are Python bytecode generated files displayed when running `git status`. 
# And we don't want to commit them inadvertently: this is the purpose of the `.gitignore` file.
# Add the `.gitignore` to the base directory of your working copy with the following content:
# %%
!echo "*pyc" > .gitignore
# %% [markdown]
# Check that bytecode generated files are not listed anymore when running `git status`.
# %%
!git status

# %% [markdown]
# Commit the `.gitignore` file:
# %%
!git add .gitignore
!git commit -m "ignore Python generated files"

# %% [markdown]
# # Manage the changes
# Let's continue with the `dummymaths` Python project and check the history of changes there.
# 1. Display the history of changes already committed, using `git log`:
#   * Only the last 2 changes with `-2` flag along with their corresponding differences using `-p` flag
#   * Display the commit information with the format `<small hash> - <message> - <date> - <email>` (check the help of log)
# 2. Let's extend the tests in `test_myfuncs.py` with a test for the `sub` function, e.g with the following content:

# %%
from google.colab import files
files.view('test_myfuncs.py')
# %% [markdown]
# Add this import line in the test_myfuncs.py file.
# ```python
# from myfuncs import add, sub

# [...]

# # Add this function at the end of the file
# @pytest.mark.parametrize(
#     "a,b,res",
#     [
#         (0, 0, 0),
#         (0, 42, -42),
#         (42, 0, 42),
#         (42, 42, 0),
#         (42, -42, 84),
#         (-42, 42, -84),
#     ]
# )
# def test_sub(a, b, res):
#     assert sub(a, b) == res
# ```

# %% [markdown]
# Once applied, add them to the staging area:
# %% 
!git add test_myfuncs.py

# %% [markdown]
# Check the state of your local copy with `git status`: there's something in the staging area and nothing in the local changes.
# %% 
!git status
!git diff  # This command should return nothing

# %% [markdown]
# Verify that the changes added in the staging are the ones expected:
# %% 
!git diff --staged
# %% [markdown]
# Remove the changes from the staging area:
# %% 
!git reset
# %% [markdown]
# Check the modifications are still there, in the local changes:
# %% 
!git status
!git diff
# %% [markdown]
# Repeat 4. and 5. but this time completely revert the changes added to the staging area (`git reset --hard`)
# %% [markdown]
# Apply one last time the changes above to the `test_myfuncs.py` file and commit them:
# %%
from google.colab import files
files.view('test_myfuncs.py')
# %% 
!git add test_myfuncs.py
!git commit -m "add test function for sub"
# %% [markdown]
# Check the diff contained in this last commit:
# %% 
!git log -1 -p

# %% [markdown]
# # Working with remote repositories

# %% [markdown]
# Some preliminary checks:
# In your local working copy, check that no remote repository is already configured:
# %% 
!git remote
# %% [markdown]
# Move to another directory, out of the `dummymaths` one, and initialize there a bare repository. We will use it as a remote repository for `dummymaths`
# %% 
%cd ..
!mkdir dummymaths_remote
%cd dummymaths_remote
!git init --bare
# %% [markdown]
# Move back to the `dummymaths` directory, that contains your initial git working copy and from there add the newly created remote repository. The url of this repository is just a path in your filesystem:
# %%
%cd ../dummymaths
!git remote add origin ../dummymaths_remote
# %% [markdown]
#  Push your `main` branch and enable upstream tracking in the meantime:
# %% 
!git push origin main -u
# %% [markdown]
# Check that the `main` branch is now referenced on the remote repository:
# %% 
!git remote show origin
# %% [markdown]
# In another directory, clone the repository of the source code of the tutorial that is hosted on gitlab:
# %% 
%cd ..
!git clone https://gitlab.inria.fr/git-tutorial/git-tutorial.git
# %% [markdown]
# You can check the status of your local copy and the information about the remote repository:
# %% 
%cd git-tutorial
!git status
!git remote -v

# %% [markdown]
# Manipulating branches

# %% [markdown]
# Move back to the `dummymaths` directory and list your local and remote branches:
# %% 
%cd ../dummymaths
!git branch
!git branch -a
# %% [markdown]
# Create a branch called `multiply` and list the branches again:
# %% 
!git branch multiply
!git branch
# %% [markdown]
# Current branch is still `main` but there's a new `multiply` branch. Also note
# how immediate it is to create a new branch.
# %% [markdown]
# Switch to the `multiply` branch and list the local branches again:
# %% 
!git checkout multiply
!git branch
# %% [markdown]
# Now let's display the history of commits on both branches:
# %% 
!git log --decorate --graph --oneline --all
# %% [markdown]
# You can also try with a graphical tool, such as `gitk` using `gitk --all`
# %% [markdown]
# Let's add a new multiply function to the `myfuncs.py` module:
# %% 
files.view('myfuncs.py')
# %% [markdown]
# ```python
# multiply(a, b):
# """Multiply a by b."""
#   return a * b
# ```
# %% [markdown]
# Commit the changes above, they should end up in the `multiply` branch, and
# display the history of changes, like before:
# %% 
!git commit -am "myfuncs: add the multiply function"
!git log --decorate --graph --oneline --all
# %% [markdown]
# The `multiply` branch is now one commit ahead of `main`.
# %% [markdown]
# Now switch back to the `main` branch, and add the following test function to `test_myfuncs.py`:

# %%
!git checkout main
files.view('test_myfuncs.py')
# %% 
#   # Add this import line
#```python
# from myfuncs import multiply

# @pytest.mark.parametrize(
#     "a,b,res",
#     [
#         (0, 0, 0),
#         (0, 42, 0),
#         (42, 0, 0),
#         (42, 1, 42),
#         (1, 42, 41),
#         (-1, 42, -42),
#     ]
# )
# def test_multiply(a, b, res):
#     assert multiply(a, b) == res
# ```
# %% [markdown]
# Finally, commit the changes above and display the branch history:
# %% 
!git commit -am "tests: add test function for multiply"
!git log --decorate --graph --oneline --all

# %% [markdown]
# The branches are diverging
# Let's see how to merge them and to fix potential conflicts.

# %% [markdown]
# # Merging branches

# Back to the `dummymaths` Python project, let's create several branches and
# merge them in the `main` branch. For this, we'll imagine a (non-realistic)
# scenario to write a function implementing division along with a test function.

# ## The fast-forward merge

# %% [markdown]
# Let's create the `divide`and the `todo` branches from `main`, which is your current branch:
# %% 
!git branch divide
!git checkout -b todo
!git status
# %% [markdown]
# You are now on the `todo` branch. Edit the README.md file and add the following content at the end:
# %% 
# ## TODO
# Add _divide_ function
# %% [markdown]
# Commit the change above:
# %% 
!git commit -am "README.md: bootstrap todo section with a divide function item"
!git status
!git log --decorate --graph --oneline --all
# %% [markdown]
# The `todo` is now one commit ahead of `main`
# %% [markdown]
# Switch back to `main` and merge `todo` in main:
# %% 
!git checkout main
!git merge todo
# %% [markdown]
# Git will just "fast-forward `main` to `todo`
# %% 
!git log --decorate --graph --oneline --all
# %% [markdown]
# As good citizens, now that the `todo` branch is not needed anymore, let's remove 
# it:
# %% 
!git branch -d todo
!git log --decorate --graph --oneline --all
# %% [markdown]
# ## The merge commit case

# At the end of the `04-branches` exercise, the `multiply` and `main` branches
# were diverging. Normally the changes introduced in `multiply` are separate
# enough from the changes added to main and merging `multiply` in `main `should
# not conflict.
# Merge the `multiply` branch into `main`:
# %% 
!git merge multiply
!git log --decorate --graph --oneline --all
# %% [markdown]
# The merge command created a merge commit.

# ## The conflict!

# Now let's trigger a conflict on purpose by finally switching to the `divide` branch:
# %% 
!git checkout divide
# %% [markdown]
# Add the missing `divide` function to the `myfuncs.py` module:
# ```python
#   def divide(a, b):
#       """Divide a by b."""
#       try:
#           return a / b
#       except ZeroDivisionError:
#           return None
# ``` 
# %% 
files.view('myfuncs.py')

# %% [markdown]
# commit that change:
# %% 
!git commit -am "myfuncs: add divide function"
# %% [markdown]
# Add the related test function to the `test_myfuncs.py` module:
# Add this import line
# ```python
#   from myfuncs import divide

#   [...]

#   @pytest.mark.parametrize(
#       "a,b,res",
#       [
#           (0, 0, None),
#           (0, 42, 0),
#           (42, 0, None),
#           (42, 1, 42),
#           (1, 2, 0.5),
#           (-1, 2, -0.5),
#       ]
#   )
#   def test_divide(a, b, res):
#       assert divide(a, b) == res
# ``` 
# %%
files.view('test_myfuncs.py')
# %% [markdown]
# commit that change:
# %% 
!git commit -am "tests: add test function for divide"
# %% [markdown]
# Now try to merge the `divide` branch in `main`:
# %% 
!git checkout main
!git merge divide
# %% [markdown]
# Try to solve the conflict! 2 possibilities:
#   1. Manually:
#     - Fix it by editing the file
#     - Run `git add`
#     - Run `git commit`
#   2. Using a graphical tool:
# %% 
!git mergetool
!git add myfuncs.py
!git commit
# %% [markdown]
# ## One last thing

# Once all features are merged, it's time to sync your local `main` branch with
# the remote repository:

# %% 
!git push origin main
# %% [markdown]
# Also, now that the `multiply` and `divide` branches are not needed anymore, you
# can delete them:

# %% 
!git branch -d multiply divide


# %% [markdown]
# # Simple rebasing

# Let's again extend the `dummymaths` Python project with a `power` function.
# Check that your current branch is `main`:

 # %% 
!git status
# %% [markdown]
# Create a new `power` branch and switch to it:
# %% 
!git checkout -b power
# %% [markdown]
# Extend the `myfuncs.py` module with the following content:
# ```python
#   def power(a, b):
#       """Return a power b."""
#       return a ** b
# ```
# %% 
files.view('myfuncs.py')

# %% [markdown]
# Commit your change:
# %% 
!git commit -am "myfuncs: add power function"
# %% [markdown]
# Add the related test function to the `test_myfuncs.py` module:
# ```python
#   # Add this import line
#   from myfuncs import power

#   @pytest.mark.parametrize(
#       "a,b,res",
#       [
#           (0, 0, 1),
#           (0, 2, 0),
#           (1, 2, 1),
#           (2, 0, 1),
#           (2, 1, 2),
#           (2, 2, 4),
#           (2, -1, 0.5),
#       ]
#   )
#   def test_power(a, b, res):
#       assert power(a, b) == res
# ```
# %% 
files.view('test_myfuncs.py')

# %% [markdown]
# Commit your change:
# %% 
!git commit -am "tests: add test function for power"
# %% [markdown]
# Switch back to main
# Remove the `TODO` section from the README (the divide function is merged already!).
# %% 
!git status
!git add README.md
!git commit -m "README: remove TODO section"
!git log --decorate --graph --oneline --all
# %% [markdown]
# At this point, the `main` and `power` branches have diverged.
# Switch back to `power` and rebase it on top of `main`:
# %% 
!git checkout power
!git rebase main
!git log --decorate --graph --oneline --all
# %% [markdown]
# Merge power into `main`:
# %% 
!git checkout main
!git merge power
!git log --decorate --graph --oneline --all
# %% [markdown]
# This is a fast-forward move of main towards power!
# Finally push `main` and delete `power`:
 # %% 
!git push origin main
!git branch -d power



# %% [markdown]
# # Using bisect to find bugs

# The `dummymaths` Python project provides a set of unit tests. In the first
# exercise you used `pytest` to run them.
# Since then, new code was added but you didn't rerun `pytest`.

# %% [markdown]
# Let's start by running `pytest`. It should fail. If it doesn't this exercise
# becomes useless!
# %% 
%cd dummymaths
pytest
# %% [markdown]
# Even if the problem is obvious, you'll use `git bisect` to find the commit
#   that introduced the problem. In some more complex cases, that can help
#   understand what is the origin of the bug.
#   Before starting bisect, you must find a commit that works. We know one: the
#   one that contains the "add" function test that was added in exercise
#   `01-start`. Let's use `git log` with some special parameters to find it:
# %% 
!git log --oneline --grep "tests: add test function for add"
# %% [markdown]
# The reply contains the short hash of the matching commit that you'll use as
# good commit.
# %% [markdown]
# Let's now start bisecting:
# %% 
!git bisect start
!git bisect bad
# %% [markdown]
# You now have to switch the commit that we think is good, test it and tell git:
# %% 
!git checkout <short commit hash returned in point 2.>
pytest
!git bisect good
# %% [markdown]
# Since `pytest` is the command that is used to check if a commit is good or
# not, you can run it at each step:
# %% 
!git bisect run pytest
!git bisect reset
# %% [markdown]
# This command should tell you quite fast what is the first bad commit. Check
# it's content:
# %% 
!git show <first bad commit>
# %% [markdown]
# Point 5. reveals that the problem comes from one the multiply test case. It
# can be fixed by using the following parametrize values:
# ```python
# [
#     (0, 0, 0),
#     (0, 42, 0),
#     (42, 0, 0),
#     (42, 1, 42),
#     (1, 42, 42),  # problem was here
#     (-1, 42, -42),
# ]
# ```
# %%
files.view('test_myfuncs.py')

# %% [markdown]
# Commit your changes and push the `main` branch:
# %% 
!git commit -am "tests: fix multiply test case"
!git push origin main


# %% [markdown]
# # Working with GitHub

# In this exercise, you'll fork the existing `dummymaths` repository on
# [GitHub](https://github.com), extend it like in exercise `08-gitlab` and
# propose your changes upstream using a pull request.

# %% [markdown]
# ## Setup your GitHub account

# 1. Check that you can login to https://github.com
# 2. If not done already, add a public SSH key to your [GitHub account](https://github.com/settings/keys)
# 3. Check that you can access the [dummymaths project](https://github.com/aabadie/dummymaths) on GitHub

# %% [markdown]
# ## Setup your local repository

# Here you'll reuse the `dummymaths` cloned in the `dummymaths-remote` directory
# from the previous exercise on GitLab.
# %% [markdown]
# 1. On the [GitHub web interface](https://github.com/aabadie/dummymaths), fork
#   the project by clicking the fork button on the top right corner
# 2. Add your fork as a new remote in your local copy (use your login on github, or just `github` as fork name):
# %% 
!git remote add <fork name> git@github.com:<github login>/dummymaths.git
# %% [markdown]
# Check the configuration of your remotes:
# %% 
!git remote -v
# %% [markdown]
# You should have:
#   - origin: the upstream repository on gitlab
#   - `<fork name>`: your fork on Gitlab
#   - `<github fork name>`: your fork on GitHub

# %% [markdown]
# ## Open the pull-request
# Push the existing `modulo` branch to your github fork:
# %% 
!git push <github fork name> modulo
# %% [markdown]
# The command above should propose you if you want to open a pull request. The
#   answer is obviously yes. You can either:
#   - click on the proposed link
#   - click on the web interface. There's a link in a header of the
#     [dummymaths project](https://github.com/aabadie/dummymaths)
# %% [markdown]
# In the web interface, do a last minute check of your changes (everything isthere ?, no typo ?).
# %% [markdown]
# If everything is fine, click the "Open pull-request" green button, and wait for the reviews
