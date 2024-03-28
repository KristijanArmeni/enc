from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def load_config():

    with open(ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config
