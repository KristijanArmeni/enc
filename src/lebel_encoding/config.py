import os
from os.path import dirname, join

REPO_DIR = join(dirname(dirname(dirname(os.path.abspath(__file__)))))
EM_DATA_DIR = join(dirname(REPO_DIR), "enc_data", "ds003020", "derivative")
DATA_DIR = join(dirname(REPO_DIR), "enc_data")
