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
# setup up conda environment and install git-annex
conda create -n enc python=3.9 conda-forge::git-annex
conda activate enc

# install dependencies
pip install -r requirements.txt

# install pre-commit
pre-commit install

# run pre-commit against all files once
pre-commit run --all-files

# Copy the config
cp config.example.yaml config.yaml
```

### Download data

```sh
# download empty dataset (ideally not within this repository)
cd <separate-data-dir>
datalad clone https://github.com/OpenNeuroDatasets/ds003020.git
cd ds003020

# Edit config.yaml such that your data path points to the ds003020 repository

# Stories: https://www.nature.com/articles/s41597-023-02437-z/tables/1
datalad get derivative/preprocessed_data/UTS02/alternateithicatom.hf5
datalad get derivative/preprocessed_data/UTS02/souls.hf5
datalad get derivative/preprocessed_data/UTS02/avatar.hf5
datalad get derivative/preprocessed_data/UTS02/legacy.hf5
datalad get derivative/preprocessed_data/UTS02/odetostepfather.hf5
datalad get derivative/TextGrids/alternateithicatom.TextGrid
datalad get derivative/TextGrids/souls.TextGrid
datalad get derivative/TextGrids/avatar.TextGrid
datalad get derivative/TextGrids/legacy.TextGrid
datalad get derivative/TextGrids/odetostepfather.TextGrid
datalad get stimuli/alternateithicatom.wav
datalad get stimuli/souls.wav
datalad get stimuli/avatar.wav
datalad get stimuli/legacy.wav
datalad get stimuli/odetostepfather.wav
datalad get derivative/pycortex-db/UTS02/
datalad get derivative/english1000sm.hf5
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


## Team

- [Gabriel Kressing Palacios](https://gabrielkp.com/)
- Gio Li
- [Kristijan Armeni](https://www.kristijanarmeni.net/)


## License
