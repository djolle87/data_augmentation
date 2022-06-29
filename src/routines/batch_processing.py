import logging
import os
from datetime import datetime
from pathlib import Path

from src.routines.augmentation import run_parallel_augmentation, run_sequential_augmentation


class BatchAugmentation:
    """
    A class to represent batch augmentation job (either sequential or parallel augmentation job.
    """

    def __init__(self, config: dict):
        """

        Args:
            config: Configuration.
        """
        self._config = config
        self._approach = config["augmentation_job"]["approach"]
        self._export = self._config["augmentation_job"]["export"]
        self._timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        if (not os.environ.get('IMPORT_DATA_PATH')) and (not os.environ.get('EXPORT_DATA_PATH')):
            self._root_import_path = Path(config["augmentation_job"]["paths"]["import_path"])
            self._root_import_dir_name = self._root_import_path.name
            self._root_export_path = Path(config["augmentation_job"]["paths"]["export_path"]).joinpath(
                self._timestamp).joinpath(self._root_import_dir_name)
        else:
            self._root_import_path = Path(os.environ.get('IMPORT_DATA_PATH'))
            self._root_import_dir_name = self._root_import_path.name
            self._root_export_path = Path(os.environ.get('EXPORT_DATA_PATH')).joinpath(
                self._timestamp).joinpath(self._root_import_dir_name)

        logging.basicConfig(format='%(asctime)s | %(name)-32s | %(levelname)s | %(message)s',
                            level=os.environ.get("LOGLEVEL", "INFO"))

        self.logger = logging.getLogger(__name__)

    def run(self):
        """
        This is a run method to launch batch data augmentation.
        Returns: None.

        """
        self.logger.info(
            f"Starting batch augmentation using {'sequence' if self._approach == 'SEQ' else 'parallel'} approach.")
        # Create export dirs
        os.makedirs(self._root_export_path)

        # Traversing
        for dir_path, subdir_list, file_list in os.walk(self._root_import_path):
            self.logger.info(2 * "_______________________________________________________________")
            self.logger.info(f"Input directory: {dir_path}")

            # Check if this is root dir
            is_root = os.path.samefile(self._root_import_path, dir_path)

            if not is_root:
                tmp_path = dir_path.split(str(self._root_import_path))[1][1:]
                tmp_output_path = self._root_export_path.joinpath(Path(tmp_path))
            else:
                tmp_output_path = self._root_export_path

            # File traversing
            for file_name in file_list:

                # Take only .wav files
                if file_name.lower().endswith(".wav"):
                    input_file_path = Path(dir_path).joinpath(file_name)
                    self.logger.info(f"\tFile:\t{file_name}")

                    # Run augmentation
                    if self._approach == "SEQ":
                        sequence = self._config["augmentation_approach"]["sequential"]["sequence"]
                        run_sequential_augmentation(input_signal_path=Path(input_file_path),
                                                    config=self._config,
                                                    sequence=sequence,
                                                    export=self._export,
                                                    export_path=tmp_output_path)
                    elif self._approach == "PAR":
                        methods = self._config["augmentation_approach"]["parallel"]["methods"]
                        n_methods = self._config["augmentation_approach"]["parallel"]["n_methods"]
                        run_parallel_augmentation(input_signal_path=Path(input_file_path),
                                                  config=self._config,
                                                  methods=methods,
                                                  n_methods=n_methods,
                                                  export=self._export,
                                                  export_path=tmp_output_path)
