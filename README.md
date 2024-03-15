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

# download empty dataset (but not in the repository)
cd <separate-data-dir>
datalad clone https://github.com/OpenNeuroDatasets/ds003020.git
cd ds003020
# download 5 files
datalad get derivative/preprocessed_data/UTS02/X
```
