import argparse
import shutil
from pathlib import Path
from typing import Optional, Union

import yaml
from datalad.api import clone, get  # type: ignore

from encoders.utils import get_logger, load_config

log = get_logger(__name__)

DATASET_URL = "https://github.com/OpenNeuroDatasets/ds003020.git"


def download_data(
    data_dir: Optional[str], stories: str, subjects: Union[str, list[str]]
):
    """Downloads data needed to run code into data_dir.

    Parameters
    ----------
    data_dir

    stories

    """

    # 0. parameters
    data_dir = data_dir or "ds003020"
    if not isinstance(subjects, list):
        subjects = [subjects]

    if "all" in subjects:
        subjects = [
            "UTS01",
            "UTS02",
            "UTS03",
            "UTS04",
            "UTS05",
            "UTS06",
            "UTS07",
            "UTS08",
        ]

    # 1. Clone
    if Path(data_dir).exists():
        log.info(f"{data_dir} already exists. Skipping cloning.")
    else:
        clone(source=DATASET_URL, path=data_dir)

    # 2. Download relevant data
    if stories == "all":
        log.info("Downloading all data, this can take a while.")
        get(
            dataset=data_dir,
            path=Path(data_dir, "derivative/english1000sm.hf5"),
        )
        for subject in subjects:
            get(
                dataset=data_dir,
                path=Path(data_dir, "derivative/TextGrids"),
            )
            get(
                dataset=data_dir,
                path=Path(data_dir, "stimuli"),
            )
            get(
                dataset=data_dir,
                path=Path(data_dir, f"derivative/pycortex-db/{subject}"),
            )
            get(
                dataset=data_dir,
                path=Path(data_dir, f"derivative/preprocessed_data/{subject}"),
            )
    else:
        log.info("Downloading three stories")
        story_names = ["souls", "alternateithicatom", "avatar"]

        get(
            dataset=data_dir,
            path=Path(data_dir, "derivative/english1000sm.hf5"),
        )

        for story_name in story_names:
            for subject in subjects:
                get(
                    dataset=data_dir,
                    path=Path(data_dir, f"derivative/TextGrids/{story_name}.TextGrid"),
                )
                get(
                    dataset=data_dir,
                    path=Path(data_dir, f"stimuli/{story_name}.wav"),
                )
                get(
                    dataset=data_dir,
                    path=Path(data_dir, f"derivative/pycortex-db/{subject}"),
                )
                get(
                    dataset=data_dir,
                    path=Path(
                        data_dir,
                        f"derivative/preprocessed_data/{subject}/{story_name}.hf5",
                    ),
                )

    # After download update the config data_dir
    if not Path("config.yaml").exists():
        shutil.copy("config.example.yaml", "config.yaml")
        log.info("Created new config file")

    config = load_config()
    config["DATA_DIR"] = data_dir

    with open("config.yaml", "w") as f_out:
        yaml.dump(config, f_out)
    log.info(f"Updated config.yaml to DATA_DIR={data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=("Path into which data is downloaded."),
    )
    parser.add_argument(
        "--stories",
        nargs="+",
        choices=["all", "few"],
        default=["few"],
        help="Amount of stories to download, will download all by default, or 3 if 'few' is selected.",
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        type=str,
        default=["UTS02"],
        help="List of subject identifier. Can be 'all' for all subjects.",
        choices=[
            "all",
            "UTS01",
            "UTS02",
            "UTS03",
            "UTS04",
            "UTS05",
            "UTS06",
            "UTS07",
            "UTS08",
        ],
    )
    args = parser.parse_args()

    download_data(args.data_dir, args.stories, args.subject)
