<h1 align="center">Encoding Models</h1>

<p align="center">A project reproducing endocing models published by <a href="https://github.com/HuthLab/deep-fMRI-dataset"><i>LeBel et al. 2023</i></a>.</p>
<p align="center">We documented our results in this <a href="https://kristijanarmeni.github.io/encoders_report/"><i>Report</i></a>.</p>

<p align="center">
<a href="https://gabrielkp.com/enc/"><img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

---

## Setup

```sh
# setup up conda environment
conda create -n enc python=3.9
conda activate enc

# install package
pip install .

# install git-annex
https://handbook.datalad.org/en/latest/intro/installation.html

# download the data (one of the below)
python src/encoders/download_data.py # subject 2 & few stories
python src/encoders/download_data.py --stories all # subject 2 & all stories
python src/encoders/download_data.py --stories all --subjects all # all subjects & all stories

# you can also install the path into a custom dir
python src/encoders/download_data.py --data_dir /path/to/custom/dir


# Setup the config with the editor of your choice
nano config.yaml
# If you want to generate plots, make sure to install inkscape
# and set its path in INKSCAPE_DIR (see below)
```

### Setup pycortex (for visualization)

1. Find out the location of the pycortex config:
   In the ipython terminal:

```py
import cortex
cortex.options.usercfg
```

2. Open the config file (should be a `*/options.cfg`)

3. Modify the filestore entry to point towards `ds003020/derivative/pycortex-db`

4. To use `quickshow` install inkscape: https://inkscape.org/release/inkscape-1.3.2/

5. Make sure inkscape is available in the terminal. [Instructions Mac](https://stackoverflow.com/a/22085247)

## Development setup

1. [Install poetry](https://python-poetry.org/docs/#installation)
2. Run following commands:

```sh
# setup up conda environment (optional)
conda create -n enc python=3.9
conda activate enc

# install dependencies
poetry install

# install pre-commit
pre-commit install

# run pre-commit against all files once
pre-commit run --all-files

# download the data (one of the below)
python src/encoders/download_data.py # subject 2 & few stories

# Setup the config with the editor of your choice
nano config.yaml
```

## Team

- [Gabriel Kressing Palacios](https://gabrielkp.com/)
- Gio Li
- [Kristijan Armeni](https://www.kristijanarmeni.net/)


## License
