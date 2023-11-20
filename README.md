# NOW-2023

Repository for the [2023 NeuroScience Open Workshop](https://open-neuro.org/).

The online version, published as a Jupyter book can be browsed [here](https://aramislab.paris.inria.fr/workshops/NOW/2023/).

The notebooks rely on the [now-2023](https://now-2023.readthedocs.io/en/latest/index.html) library.

## Quick start

If you want to follow the tutorial online, there is nothing to do, just go the
[jupyter book](https://aramislab.paris.inria.fr/workshops/NOW/2023/) and run
the interactive notebooks on Collab.

If you want to follow on your local machine, we strongly encourage you to build
a virtual environment (for example with conda). You can then install the requirements:

```bash
$ conda env create -f environment.yml
$ conda activate now
$ pip install -r requirements.txt
$ pip install -r ./jupyter-book/requirements.txt
```

## Schedule

09:30-10:00 - [Introduction and setup](https://aramislab.paris.inria.fr/workshops/NOW/2023/notebooks/preamble.html)
10:00-10:15 - [Introduction to Git](https://aramislab.paris.inria.fr/workshops/NOW/2023/notebooks/code_versioning_intro.html)
10:15-11:00 - [Version your code with Git](https://aramislab.paris.inria.fr/workshops/NOW/2023/notebooks/code_versionning.html)
11:00-11:15 - Break
11:15-11:30 - [Introduction to DVC](https://aramislab.paris.inria.fr/workshops/NOW/2023/notebooks/data_versionning_intro.html)
11:30-12:00 - [Version code and data with DVC](https://aramislab.paris.inria.fr/workshops/NOW/2023/notebooks/data_versionning.html)
12:00-12:30 - [Version your experiments as data pipelines](https://aramislab.paris.inria.fr/workshops/NOW/2023/notebooks/data_versionning_2.html)

## Contributing

### Build  all the notebooks

Once your changes are done in the scripts (the `src` files) run at the root folder:

```
make
```

This command will recreate automatically the notebooks and clean the outputs.

### Build the jupyter-book

You need to install the requirements:

```
$ pip install -r jupyter-book/requirements.txt
```

Then:

```
$ cd jupyter-book
$ make
```

The HTML files are then in `jupyter-book/_build/html/`.

## External ressources

- [Git training performed by the Paris SED](https://gitlab.inria.fr/git-tutorial/git-tutorial)
- [Reproducible research notebook by the Turing Way](https://the-turing-way.netlify.app/index.html)
- [GIN Tonic](https://gin-tonic.netlify.app/)

