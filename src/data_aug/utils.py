import os
import numpy as np
import yaml
from pathlib import Path
import logging
from matplotlib import pyplot as plt
import soundfile as sf
import IPython.display as ipd


def read_yaml_config(path: Path) -> dict:
    """
    This function reads a yaml config file.
    Args:
        path: Path to config file.

    Returns: Config file as a dictionary.

    """

    if not path.exists():
        raise Exception(
            f"Path '{path}' to configuration file doesn't exist."
        )

    with open(path, "r") as file:
        if yaml.__version__ < '5.1':
            config = yaml.safe_load(file)
        else:
            config = yaml.load(file, Loader=yaml.Loader)

    return config


def generate_file_name(original_name: str, config: dict, method: str) -> str:
    """
    This function generates file name including file suffix.
    Args:
        original_name: Original input file name.
        config: Config file as a dictionary.
        method: Augmentation method.

    Returns: File name as a string.

    """
    generate_suffixes = config["file_suffixes"]["generate_suffixes"]

    if generate_suffixes:
        is_detailed = config["file_suffixes"]["detailed_suffixes"]
        file_name = original_name.split(".")[0]
        suffix = generate_suffix(config, method, is_detailed)
        return file_name + suffix + ".wav"
    else:
        return original_name


def generate_suffix(config: dict, method: str, is_detailed: True) -> str:
    """
    This function creates suffix.
    Args:
        config: Configuration file as a dictionary.
        method: Augmentation method.
        is_detailed: A flag to represent detailed or simple suffix.

    Returns: Generated suffix as a string.

    """
    simple_suffix = config["file_suffixes"][method]["suffix"]

    long_to_short_str = {
        "rate": "R",
        "sr": "SR",
        "bins_per_octave": "BPO",
        "n_steps": "NS",
        "gain_dB": "G",
        "snr_db": "SNR",
        "shift_pct": "SPCT"
    }

    if not is_detailed:
        return simple_suffix
    else:
        aug_settings = config["augmentation_method"][method]

        detailed_suffix = ""
        for item in aug_settings.items():
            method_suffix = long_to_short_str[item[0]]
            value = item[1]
            detailed_suffix = method_suffix + str(value)

        return simple_suffix + "_" + detailed_suffix


def print_plot_play(x: np.ndarray, sr: int, text: str = ''):
    """
    This function plots the wave frame and creates player
    Args:
        x: Input signal represented as a numpy array.
        sr: Sampling rate.
        text: Text to print

    Returns: None

    """

    print('%s Fs = %d, x.shape = %s, x.dtype = %s' % (text, sr, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=sr))


def export_audio_file(audio_signal: np.ndarray, file_name: str, path: str, sr: int = 22050):
    """
    This function saves augmented file to a given path.
    Args:
        audio_signal: Augmented audio file represented as a numpy array.
        file_name: File name for an augmented file.
        path: Export path.
        sr: Sampling rate

    Returns: None

    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(path):
        os.makedirs(path)

    file_path = f"{path}/{file_name}"
    sf.write(file=file_path, data=audio_signal, samplerate=sr)
    logger.info(f"\t\t\tSaved file: {file_path}")
