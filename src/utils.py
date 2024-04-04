import logging
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).parent.parent
FORMAT = "[%(levelname)s] %(name)s.%(funcName)s - %(message)s"

logging.basicConfig(format=FORMAT)


def get_logger(
    name=__name__,
    log_level=logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Initializes command line logger."""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if log_file is not None:
        formatter = logging.Formatter(FORMAT)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_config():

    with open(ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config
