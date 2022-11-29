import logging
from pathlib import Path
from random import sample

import librosa

from src.data_aug.audio_augmentation import Augmenter
from src.data_aug.utils import generate_file_name, export_audio_file, print_plot_play


def run_sequential_augmentation(input_signal_path: Path, config: dict, sequence: list, export: bool,
                                export_path: Path = None) -> tuple:
    """
    This function applies sequence of augmentation methods on a given input signal.
    Args:
        input_signal_path: Path to input signal (time series data).
        config: Config file.
        sequence: List (sequence) of augmentation methods.
        export: Save outputs (True/False)
        export_path: Export path.

    Returns: Augmented signal and augmented filename.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\t\tStarting sequence augmentation: {sequence}")
    input_signal, _ = librosa.load(input_signal_path, sr=None)
    file_name = input_signal_path.name.lower()
    file_report = {"FILE_NAME": file_name}

    aug_file_name = file_name.split(".wav")[0]

    y = input_signal.copy()
    for method in sequence:
        aug = Augmenter(input_signal=y, file_name=file_name, config=config)
        tmp_config = config["augmentation_method"][method]
        if tmp_config:
            y = getattr(aug, method)(**tmp_config)
        else:
            y = getattr(aug, method)()

        file_report.update(aug._report)
        aug_file_name = aug_file_name + generate_file_name(file_name, config, method).split(".wav")[0].split(
            file_name.split(".wav")[0])[1]

    aug_file_name = aug_file_name + ".wav"

    if export:
        if not export_path:
            export_path = config["augmentation_job"]["paths"]["export_path"]
            if not Path(export_path).exists():
                raise Exception("You need to specify correct path for data export in the config.")
        export_audio_file(audio_signal=y, file_name=aug_file_name, path=str(export_path))
    logger.info(f"\t\tFinished sequence augmentation.")
    logger.info(2 * "_______________________________________________________________")

    return y, aug_file_name, file_report


def run_parallel_augmentation(input_signal_path: Path, config: dict, methods: list, n_methods: int, export: bool,
                              export_path: Path = None):
    """
    This function applies parallel data augmentation on a given input signal.
    Args:
        input_signal_path: Path to input signal (time series data).
        config: Config.
        methods: List of audio augmentation methods.
        n_methods: Number of random methods picked from a list of methods.
        export: Save outputs (True/False)
        export_path: Export path.

    Returns: List of augmented time series.
    """
    logger = logging.getLogger(__name__)
    assert len(methods) >= n_methods, "n_samples is greater than number of picked methods"

    input_signal, _ = librosa.load(input_signal_path, sr=None)
    file_name = input_signal_path.name
    file_report = {"FILE_NAME": file_name}

    sample_sequence = sample(methods, n_methods)
    logger.info(f"\t\tStarting parallel augmentation: {sample_sequence}")

    for method in sample_sequence:
        y = input_signal.copy()
        aug = Augmenter(input_signal=y, file_name=file_name, config=config)
        tmp_config = config["augmentation_method"][method]
        if tmp_config:
            y = getattr(aug, method)(**tmp_config)
        else:
            y = getattr(aug, method)()
        file_report.update(aug._report)
        aug_file_name = generate_file_name(file_name, config, method)

        if export:
            if not export_path:
                export_path = config["augmentation_job"]["paths"]["export_path"]
                if not Path(export_path).exists():
                    raise Exception("You need to specify correct path for data export in the config.")
            export_audio_file(audio_signal=y, file_name=aug_file_name, path=str(export_path))
        else:
            print_plot_play(x=y, sr=22050, title="Augmented signal", text=aug_file_name)

    return file_report

    logger.info(f"\t\tFinished parallel augmentation.")
    logger.info(2 * "_______________________________________________________________")
