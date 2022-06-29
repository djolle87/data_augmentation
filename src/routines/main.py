from pathlib import Path
import sys, os

sys.path.append(os.getcwd())

from src.data_aug.utils import read_yaml_config
from src.routines.batch_processing import BatchAugmentation

if __name__ == "__main__":
    config = read_yaml_config(Path("src/config/config.yml"))

    batch_job = BatchAugmentation(config=config)
    batch_job.run()
