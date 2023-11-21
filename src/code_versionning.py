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
# In this chapter, we will take a look at how [Git](https://git-scm.com) works and how to use it to version code.
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
#
# #### Create and initialize a repository
# %% [markdown]
# Create a directory `dummymaths`:
# %%
# ! mkdir dummymaths
# %cd dummymaths
# %% [markdown]
# ```{warning}
# If you are running this notebook on Collab, or if you are using an old version of Git, you need to run the following cell which will make sure your default branch is nammed `main` and not `master` as this default was changed a couple years ago.
#
# Otherwise, you would have to change `main` to `master` manually in all the commands of this notebook.
# ```

# %%
# ! git config --global init.defaultBranch main

# %% [markdown]
# Initialize a [Git](https://git-scm.com) local repository in the `dummymaths` directory with the [git init](https://git-scm.com/docs/git-init) command:
# %%
# ! git init

# %% [markdown]
# #### Configure your name and email
#
# Configure your git account using simple configuration files 3 levels:
#   * __local__ : `<local copy>/.git/config`, by default
#   * __global__ : `~/.gitconfig`, option `--global`
#   * __system__ : `/etc/gitconfig`, option `--system`
#
# To configure your user name and email address, you can either edit the config files directly, or use the [git config](https://git-scm.com/docs/git-config) command:
# %%
# ! git config --local user.name "John Doe"
# ! git config --local user.email john.doe@inria.fr
# %% [markdown]
# ### Add a simple text file
#
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
# %%
# ! echo "This is the new README file" > README.md

# %% [markdown]
# You should have changes both in the staging area and in the working directory (local changes). We can display the changes between HEAD and the staging area:

# %%
# ! git diff --staged

# %% [markdown]
# And the changes between the staging area and the workspace:

# %%
# ! git diff
# %% [markdown]
# The outputs of these commands with the green and red lines are what is commonly called "a diff". Diffs are everywhere in Git and it is very important to get used to reading them. When reviewing Pull Requests on Github or Gitlab for example, you usually look at the diff.
#
# You can read [the following article](https://www.atlassian.com/git/tutorials/saving-changes/git-diff) to get more details on how to read them. After practicing a bit, this will become very natural.
#
# In order to follow the tutorial, remember that a diff should be red at the line level: lines in red with a `-` symbol at the begining are lines which were removed, while green lines with a `+` symbol are lines which were added.

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
# Use `pytest` to run the tests, verify 
# that they pass and then commit `test_myfuncs.py` (and only this one!):
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
# We don't want to commit them inadvertently: this is the purpose of the [.gitignore](https://git-scm.com/docs/gitignore) file.
#
# The [.gitignore](https://git-scm.com/docs/gitignore) file is basically a list of patterns for files to ignore. Here, we want to ignore all files ending with `pyc`:

# %%
# ! echo "*pyc" > .gitignore
# %% [markdown]
# Check that bytecode generated files are not listed anymore when running [git status](https://git-scm.com/docs/git-status):
# %%
# ! git status

# %% [markdown]
# The Python bytecode files do not appear in the ouput, but the [.gitignore](https://git-scm.com/docs/gitignore) file does.
#
# Let's add it and commit it as we wish to version it together with our code:
# %%
# ! git add .gitignore
# ! git commit -m "ignore Python generated files"

# %% [markdown]
# ## Manage the changes
#
# Let's continue working on our `dummymaths` Python project!
#
# ### Visualize the project's history
#
# First, we would like to refresh our mind on the latest modifications that were made to the project. [Git](https://git-scm.com) is your friend and provides you the [git log](https://git-scm.com/docs/git-log) command to display the history of changes already committed.
#
# `````{admonition} Peaceful commands
# :class: tip
#
# As for [git status](https://git-scm.com/docs/git-status), this command can never hurt you and will probably save you a lot by reminding you the project's history, so do not hesitate to use it!
# `````
#
# Without any argument, [git log](https://git-scm.com/docs/git-log) displays all the commits of the project **in reverse chronological order** (the first commit is the most recent commit):

# %%
# ! git log

# %% [markdown]
# You can configure this command in a lot of ways that we cannot cover here. Do not hesitate to take a look at [the documentation](https://git-scm.com/docs/git-log), you will likely find pretty cool ways to visualize a project's history.
#
# As an example, the `-some_integer` option lets you control the number of commits you want to display:

# %%
# ! git log -2

# %% [markdown]
# The `-p` option prints the differences between the commits:

# %%
# ! git log -2 -p

# %% [markdown]
# ### Undoing simple mistakes
#
# Let's first extend the tests in `test_myfuncs.py` with a test for the `sub` function:

# %%
# %%writefile -a test_myfuncs.py

@pytest.mark.parametrize(
    "a,b,res",
    [
        (0, 0, 0),
        (0, 42, -42),
        (42, 0, 42),
        (42, 42, 1),
        (42, -42, 84),
        (-42, 42, -84),
    ]
)
def test_sub(a, b, res):
    from myfuncs import sub

    assert sub(a, b) == res

# %% [markdown]
# We are happy with our test and we are in a rush so we add and commit our changes right away:

# %%
# ! git add test_myfuncs.py
# ! git commit -m "add test function for sub"

# %% [markdown]
# But wait, we aren't as good mathematicians as we thought and we made a mistake:

# %%
# ! pytest -v

# %% [markdown]
# Obviously, `42 - 42` is equal to 0 and not 1 !
#
# Now, we could change the test file to correct that and make a new commit with a message saying that we are fixing a previously introduced error.
#
# That is a **totally fine approach** but we will opt for a different one here. Indeed, we are a bit ashamed and we would like to re-write the history to **make it look like we never made this stupid mistake**.
#
# How can we use git to do that ?
#
# Luckily for us, this is the best case scenario here since our mistake is in the very last commit and we didn't actually push our commits on any remote. This means that our local copy is the only place where this commit exists.
#
# We have at least three options here:
#
# - Make the modification to the code and use `git commit --amend` to rewrite the latest commit. This is probably the **easiest solution** but it only works because the error is in the last commit.
# - Use [git revert](https://git-scm.com/docs/git-revert) with the hash of the commit we want to undo. This will create a new commit actually undoing the specified commit. We would then modify the file and make a new commit. This is a good approach but our mistake would still be visible in the history of the project.
# - Use [git reset](https://git-scm.com/docs/git-reset) to undo the last commit, make our modifications, and re-make a brand new commit as if nothing ever happened.
#
# Lets' try the third solution, the first and second ones can be good exercices for the interested reader.

# %%
# ! git reset --hard HEAD~1 

# %% [markdown]
# The integer after the `~` symbol specifies how many commits we want to erase. In our case, there is only one.
#
# ```{warning}
# We used the `--hard` option here which resets both the index **AND** the working tree. We are doing this to simply re-write our test function but this is VERY dangerous as you could easily lose important work. Usually, it is much better to use the `--mixed` option, which is the default anyway, which only resets the index but leaves the tree.
# ```
#
# Our project is in a clean state:

# %%
# ! git status

# %% [markdown]
# But we lost our latest commit as well as the changes we made to the test file:

# %%
# ! git log -2

# %%
# ! cat test_myfuncs.py

# %% [markdown]
# You can clearly see the danger with this command. If we haden't this notebook with the code we initially wrote, we would have to start our `test_sub` function from scratch...
#
# Let's re-write our test function with the fix:

# %%
# %%writefile -a test_myfuncs.py

@pytest.mark.parametrize(
    "a,b,res",
    [
        (0, 0, 0),
        (0, 42, -42),
        (42, 0, 42),
        (42, 42, 0), # Fixed
        (42, -42, 84),
        (-42, 42, -84),
    ]
)
def test_sub(a, b, res):
    from myfuncs import sub

    assert sub(a, b) == res


# %% [markdown]
# But this time we learn from our mistakes, and we run the test **BEFORE** committing:

# %%
# ! pytest -v

# %% [markdown]
# Great! Everything works!
#
# Let's add and commit these changes:
# %%
# ! git add test_myfuncs.py

# %%
# ! git commit -m "add test function for sub"

# %% [markdown]
# Check the diff contained in this last commit:
# %%
# ! git log -1 -p

# %% [markdown]
# ## Working with remote repositories

# %% [markdown]
# In this section, we are going to add remotes to our project.
#
# First of all, we shouldn't at this point have any remote configured i our local working copy:
# %%
# ! git remote
# %% [markdown]
# ### Add a remote located somewhere else in the filesystem
#
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
# %% [markdown]
# The project has a new remote called `origin` which translates in these two lines:

# %%
# ! git remote -v

# %% [markdown]
# For a same remote, Git makes a difference between fetch/pull and push. Most of the time, these two lines will be identical.
#
# Push your `main` branch and enable upstream tracking in the meantime:
# %%
# ! git push origin main -u
# %% [markdown]
# Check that the `main` branch is now referenced on the remote repository:
# %%
# ! git remote show origin
# %% [markdown]
# In another directory, clone a repository hosted on gitlab:
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
# Again we have a remote called `origin`, but the URL is not a path in our filesystem but the URL of the repository on GitLab.

# %% [markdown]
# ## Manipulating branches

# %% [markdown]
# Move back to the `dummymaths` directory and list your local and remote branches:
# %%
# %cd ../dummymaths
# ! git branch
# %% [markdown]
# By default, [git branch](https://git-scm.com/docs/git-branch) only lists the local branches. If you want to also list remote branches, you can add the `--all` option:

# %%
# ! git branch --all

# %% [markdown]
# ### Create a new branch and work on it
#
# Create a branch called `multiply` and list the branches again:
# %%
# ! git branch multiply
# ! git branch
# %% [markdown]
# Current branch is still `main` but there's a new `multiply` branch listed. Also note how immediate it is to create a new branch.
#
# We can now switch to the `multiply` branch with the [git switch](https://git-scm.com/docs/git-switch) command, and list the local branches again:
# %%
# ! git switch multiply
# ! git branch
# %% [markdown]
# Now let's display the history of commits on both branches:
# %%
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# ```{note}
# You can also try with a graphical tool, such as `gitk` using `gitk --all`.
# ```
#
# Let's add a new multiply function to the `myfuncs.py` module:
# %%
# %%writefile -a myfuncs.py

def multiply(a, b):
    """Multiply a by b."""
    return a * b


# %% [markdown]
# Commit the changes above, they should end up in the `multiply` branch:
# %%
# ! git commit -am "myfuncs: add the multiply function"
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# The `multiply` branch is now one commit ahead of `main`.
#
# Now switch back to the `main` branch:
# %%
# ! git switch main

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
# As we can see, the branches `main` and `multiply` are diverging.
#
# Let's see how we can merge them in a single branch.


# %% [markdown]
# ## Merging branches
#
# In this section, we are going to create several branches and merge them in the `main` branch.
#
# ### The fast-forward merge
#
# Let's start with a new `divide` branch in which we are planning to write a function implementing division:

# %%
# ! git branch divide
# %%
# ! git switch -c todo

# %% [markdown]
# we can list the different branches of the project:

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
# The `todo` branch is now one commit ahead of the `main` branch.
#
# Let's switch back to `main` and merge the `todo` branch in `main`:
# %%
# ! git switch main
# %%
# ! git merge todo

# %% [markdown]
# We are in a very simple case where Git can perform a "fast-forward" merge. 
#
# ```{note}
# A fast-forward merge can occur when there is a linear path from the current branch tip to the target branch. Instead of “actually” merging the branches, Git only moves (i.e., “fast forward”) the current branch tip up to the target branch tip.
#
# You can read more about fast-forward merge [here](https://www.atlassian.com/git/tutorials/using-branches/git-merge).
# ```
#
# In this case, Git will just "fast-forward" `main` to `todo`:
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
# ! git switch divide
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
# Now try to merge the `divide` branch in the `main` branch:
# %%
# ! git switch main
# %%
# ! git merge divide --no-edit

# %% [markdown]
# Git is telling us that there is a conflict and that it cannot perform the merge by itself. It requests us to decide how we want to handle each conflict.
#
# In this case, Git is telling us that the problem is located in the file `myfuncs`. It is also telling us that we should "fix" the conflict and then commit the result.
#     
# Fair enough, but how do we "fix" the conflict ?
#
# Let's take a look at the problematic file:
# %%
# ! cat myfuncs.py

# %% [markdown]
# As you can see, Git is showing us the conflicting portions of the code. In our case, the `main` branch contains the `multiply` function while the `divide` branch contains the `divide` function.
#
# Solving the conflict in this situation is quite easy for us because we know that we want to keep both functions. However, there is no way for Git to infer that, and this is why we have to step in and decide.
#
# Since we want to keep both functions, we simply need to remove the lines added by Git, save the file, and add it:

# %%
# ! sed -i '' -e 's#<<<<<<< HEAD##' myfuncs.py
# ! sed -i '' -e 's#>>>>>>> divide##' myfuncs.py
# ! sed -i '' -e 's#=======##' myfuncs.py

# %% [markdown]
# ```{note}
# Do not mind the code in the above cell. Usually, you would do that by opening the file in your IDE and manually edit it.
# ```
#
# Let's verify what we just did:

# %%
# ! cat myfuncs.py

# %% [markdown]
# Seems clean enough ! Let's add it and continue the merge process:

# %%
# ! git add myfuncs.py

# %%
# ! git commit --no-edit

# %% [markdown]
# Awesome! The merge was successful and we can verify that with `git log`:

# %%
# ! git log --decorate --graph --oneline --all

# %% [markdown]
# ### Synchronize your local copy with the remote(s)
#
# Once all features are merged, it's time to sync your local `main` branch with the remote repository:

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
# In this section, we will learn another way of merging branches without actually performing merge operation, a process called `rebasing`.
#
# Let's again extend the `dummymaths` Python project with a `power` function.
#
# Check that your current branch is `main` (remember that git status is your friend, use it anytime you are unsure of the state of the project):

# %%
# ! git status
# %% [markdown]
# Create a new `power` branch and switch to it:
# %%
# ! git switch -c power
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
# ! git switch main

# %%
# ! echo "This is the new README file" > README.md

# %%
# ! git add README.md
# ! git commit -m "README: remove TODO section"
# ! git log --decorate --graph --oneline --all
# %% [markdown]
# At this point, the `main` and `power` branches have diverged.
#
# We already know how to perform a merge but here we want to perform a rebase. For that, we switch back to the `power` branch, and rebase it on top of the `main` branch:
# %%
# ! git switch power
# %%
# ! git rebase main

# %%
# ! git log --decorate --graph --oneline --all

# %% [markdown]
# Alright! The `power` branch is now ahead of `main` and the two branches aren't diverging: this is a case where Git can perform a fast-forward merge!
#
# Let's do that: merge the `power` branch into `main`:
# %%
# ! git switch main
# %%
# ! git merge power --no-edit

# %%
# ! git log --decorate --graph --oneline --all

# %% [markdown]
# Great! The `main` branch is now up-to-date with the `power` branch!
#
# Finally push `main` and delete `power`:
# %%
# ! git push origin main
# ! git branch -d power

# %% [markdown]
# ## Using bisect to find bugs
#
# This final section presents a less known, but extremely powerful, Git command: [git bisect](https://git-scm.com/docs/git-bisect).
#
# ### Realize that there is a bug
#
# For some reason, we decide to run our test suite:

# %%
# ! pytest -v
# %% [markdown]
# Looks like our new habit of always running our tests before committing didn't last long...
#
# We clearly made a mistake somewhere, and even if the problem is obvious in such a simple example, we will use [git bisect](https://git-scm.com/docs/git-bisect) to find the commit that introduced the problem. In some more complex cases, that can help understand what is the origin of the bug.
#
# Before starting bisect, you must find a commit that works.
#
# We know one: the one that contains the "add function test". Let's use `git log` with some special parameters to find it:
# %%
# ! git log --oneline --grep "tests: add test function for add"
# %% [markdown]
# The reply contains the short hash of the matching commit that you'll use as
# good commit.
# %% [markdown]
# Let's now start bisecting:
# %%
# ! git bisect start
# %% [markdown]
# We know the current commit is bad because we watched our test suite fail a couple cells before:

# %%
# ! git bisect bad

# %% [markdown]
# You now have to manually switch to the commit that we think is good, test it and tell git:
# %%
# Change the hash value to the one in the output of the cell above
# ! git checkout d5d3c04
# %% [markdown]
# Let's run our tests. We should have all tests passing:

# %%
# ! pytest -v

# %% [markdown]
# Alright, seems to be the case. Let's tell Git:

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
# ! git show 3adeb7c
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
