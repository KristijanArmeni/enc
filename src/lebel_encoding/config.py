import os
from os.path import dirname, join

from encoders.utils import load_config

DATA_DIR = os.path.dirname(os.path.normpath(load_config()["DATA_DIR"]))
