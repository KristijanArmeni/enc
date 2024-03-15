# enc

The encoders

## Setup

```sh
# setup up virtual env
conda create -n enc python=3.9
conda activate enc

# Install git-annex for data download (does not have to be with brew)
brew install git-annex
brew services start git-annex

# install dependencies
pip install -r requirements.txt

# install pre-commit
pre-commit install

# run pre-commit against all files once
pre-commit run --all-files

# Copy th
cp config.example.yaml config.yaml

# download empty dataset (but not in the repository)
cd <separate-data-dir>
datalad clone https://github.com/OpenNeuroDatasets/ds003020.git
cd ds003020

# Edit config.yaml such that your data path points to the ds003020 repository

# Stories: https://www.nature.com/articles/s41597-023-02437-z/tables/1
# download 5 files
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
```
