import logging

import librosa
import numpy as np


class Augmenter:
    """
    A class to represent audio augmenter. Currently, supported augmentation methods: time_stretch,
    pitch_shift, volume_control, time_shift and add_noise.
    """

    def __init__(self, input_signal: np.ndarray, file_name: str, config: dict = None):
        """

        Args:
            input_signal: An audio input signal as an array.
            file_name: File name of a signal.
            config: Configuration.
        """
        self._input_signal = input_signal
        self._file_name = file_name
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"\t\tStarting augmentation for file: {self._file_name}")

    def time_stretch(self, rate=None) -> np.ndarray:
        """
        This method speeds up or slows down input signal by given rate.
        Args:
            rate: Stretch factor (Real positive scalar). If rate>1 the signal is speed up.
            If rate<1 the signal is slowed down.

        Returns: Audio time series stretched by the specified rate.
        """
        self._logger.info(f"\t\t\tStarting time stretching...")
        if not (self._config or rate):
            raise Exception(f"You need to pass either a config or an argument.")

        y = self._input_signal.copy()

        if self._config:
            rate = self._config["augmentation_method"]["time_stretch"]["rate"]
            self._logger.info(f"\t\t\tUsing config value rate={rate}.")
        else:
            self._logger.info(f"\t\t\tUsing manually set value rate={rate}.")

        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        self._logger.info(f"\t\t\tFinished time stretching.")

        return y_stretched

    def pitch_shift(self, sr: int, bins_per_octave: int, n_steps: int) -> np.ndarray:
        """
        This method shifts pitch of input signal by specified number of steps.
        Args:
            sr: Sampling rate
            bins_per_octave: Bins per octave
            n_steps: Number of fractional steps to shift.
            (A step is equal to a semitone if bins_per_octave is set to 12.)

        Returns: The pitch-shifted audio time-series.
        """
        self._logger.info(f"\t\t\tStarting pitch shifting...")
        if not (self._config or sr or bins_per_octave or n_steps):
            raise Exception(f"You need to pass either a config or arguments.")

        y = self._input_signal.copy()

        if self._config:
            sr = self._config["augmentation_method"]["pitch_shift"]["sr"]
            bins_per_octave = self._config["augmentation_method"]["pitch_shift"]["bins_per_octave"]
            n_steps = self._config["augmentation_method"]["pitch_shift"]["n_steps"]
            self._logger.info(
                f"\t\t\tUsing following config values: sr={sr}, "
                f"bins_per_octave={bins_per_octave} and n_steps={n_steps}.")
        else:
            self._logger.info(
                f"\t\t\tUsing manually set values: sr={sr}, bins_per_octave={bins_per_octave} and n_steps={n_steps}.")
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, bins_per_octave=bins_per_octave, n_steps=n_steps)
        self._logger.info(f"\t\t\tFinished pitch shifting.")

        return y_shifted

    def volume_control(self, gain_dB: int) -> np.ndarray:
        """
        This method controls volume of input signal.
        Args:
            gain_dB: Volume gain in dB.
    
        Returns: The audio time-series with increased or decreased volume.
        """
        self._logger.info(f"\t\t\tStarting volume control...")
        if not (self._config or gain_dB):
            raise Exception(f"You need to pass either a config or an argument.")

        y = self._input_signal.copy()

        if self._config:
            gain_dB = self._config["augmentation_method"]["volume_control"]["gain_dB"]
            self._logger.info(f"\t\t\tUsing config values for gain_dB: {gain_dB}.")
        else:
            self._logger.info(f"\t\t\tUsing manually set value: gain_dB={gain_dB}.")
        volume_ratio = 10 ** (gain_dB / 20)
        y_vol_ctrl = y * volume_ratio
        self._logger.info(f"\t\t\tFinished volume control.")

        return y_vol_ctrl

    def time_shift(self, shift_pct: int) -> np.ndarray:
        """
        This method shifts input signal in time by given percentage
        Args:
            shift_pct: Time shift in percents.

        Returns: Time shifted audio time-series.
        """
        self._logger.info(f"\t\t\tStarting time shifting...")
        if not (self._config or shift_pct):
            raise Exception(f"You need to pass either a config or an argument.")

        y = self._input_signal.copy()

        if self._config:
            shift_pct = self._config["augmentation_method"]["time_shift"]["shift_pct"]
            self._logger.info(f"\t\t\tUsing config values for shift_pct: {shift_pct}.")
        else:
            self._logger.info(f"\t\t\tUsing manually set value: shift_pct={shift_pct}.")
        n_shift = int(shift_pct / 100 * y.shape[0])
        if n_shift > 0:
            y_shift = np.pad(y, (n_shift, 0), mode="constant")
        else:
            y_shift = np.pad(y, (0, -n_shift), mode="constant")
        self._logger.info(f"\t\t\tFinished time shifting.")

        return y_shift

    def add_noise(self, snr_db: int) -> np.ndarray:
        """
        This method adds noise by specified snr.
        Args:
            snr_db: Signal-Noise-Ratio (SNR)

        Returns: The audio time-series with added white noise.
        """
        self._logger.info(f"\t\t\tStarting adding noise...")
        if not (self._config or snr_db):
            raise Exception(f"You need to pass either a config or an argument.")

        y = self._input_signal.copy()

        if self._config:
            snr_db = self._config["augmentation_method"]["add_noise"]["snr_db"]
            self._logger.info(f"\t\t\tUsing config values for snr_db: {snr_db}.")
        else:
            self._logger.info(f"\t\t\tUsing manually set value snr_db: {snr_db}.")
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr_db / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()

        y_noise = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        self._logger.info(f"\t\t\tFinished adding noise.")
        return y_noise
