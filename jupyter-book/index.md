# Overview

This practical session will give an introduction to code and data versioning.
The first part will guide you through the usage of GIT for code versioning,
while the second part will explain how to use GIT with DVC in order to perform
code and data versioning.

The objective of the tutorial is to gain knowledge on: 

- how to use GIT in a basic way in order to properly version your code
- how to use DVC with GIT in order to version your data
- how to describe your expriments as configuration files to make them easily reproducible

## Running interactively the notebooks

To run interactively the content of this book you have two options: run it locally or use Colab.

::::{tab-set}

:::{tab-item} Run in Colab
* When the content of the page is interactive, hover over the rocket icon 
  <i class="fa fa-rocket" aria-hidden="true"></i>
  at the top of the page an click "Colab" to open a cloud version of the same page.
  Colab notebooks are very similar to jupyter notebooks and the content
  can be executed cell by cell by clicking Ctrl-Enter (or Cmd-Enter).

* You need to login with a Google account and authorize to link with github.

* Remember to choose a runtime with GPU (Runtime menu -> *"Change runtime
  type"*). 
:::

:::{tab-item} Run Locally
* Clone the repository:
```
git clone https://github.com/aramis-lab/NOW-2023
cd NOW-2023
git checkout student
```

* Create a dedicated environment
```
conda env create -f environment.yml
conda activate now
```

* Install the dependencies
```
pip install -r ./requirements.txt
pip install -r ./jupyter-book/requirements.txt
```

* Launch jupyterlab or jupyter notebook
```
jupyter lab
```
A new browser window will open, choose the correponding notebook from the folder `notebooks`.
:::

::::


```{admonition} Prerequisite
Programming knowledge in Python, basics usage of PyTorch ([see
here](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)).
```

```{tableofcontents}
```
