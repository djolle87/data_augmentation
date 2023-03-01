import librosa
import logging
import numpy as np
import pyroomacoustics as pra
import random
import scipy.signal as sg


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
        self._report = {}

    def time_stretch(self, rate=None, p: float = 0) -> np.ndarray:
        """
        This method speeds up or slows down input signal by given rate.
        Args:
            rate: Stretch factor (Real positive scalar). If rate>1 the signal is speed up.
            If rate<1 the signal is slowed down.
            p: Probability of applying time stretch.

        Returns: Audio time series stretched by the specified rate.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping applying time stretching...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting time stretching...")
        if not (self._config or rate):
            raise Exception(f"You need to pass either a config or an argument.")

        y = self._input_signal.copy()

        if self._config:
            rate = self._config["augmentation_method"]["time_stretch"]["rate"]
            if isinstance(rate, list):
                rate = random.randint(rate[0] * 10, rate[1] * 10) / 10
            self._logger.info(f"\t\t\tUsing config value rate={rate}.")
        else:
            self._logger.info(f"\t\t\tUsing manually set value rate={rate}.")

        self._report["TIME_STRETCH"] = {"RATE": rate}
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        self._logger.info(f"\t\t\tFinished time stretching.")

        return y_stretched

    def pitch_shift(self, sr: int, bins_per_octave: int, n_steps, p: float = 0) -> np.ndarray:
        """
        This method shifts pitch of input signal by specified number of steps.
        Args:
            sr: Sampling rate
            bins_per_octave: Bins per octave
            n_steps: Number of fractional steps to shift.
            (A step is equal to a semitone if bins_per_octave is set to 12.)
            p: Probability of applying pitch shifting.

        Returns: The pitch-shifted audio time-series.
        """
        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping pitch shifting...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting pitch shifting...")
        if not (self._config or sr or bins_per_octave or n_steps):
            raise Exception(f"You need to pass either a config or arguments.")

        y = self._input_signal.copy()

        if self._config:
            sr = self._config["augmentation_method"]["pitch_shift"]["sr"]
            bins_per_octave = self._config["augmentation_method"]["pitch_shift"]["bins_per_octave"]
            n_steps = self._config["augmentation_method"]["pitch_shift"]["n_steps"]
            if isinstance(n_steps, list):
                n_steps = random.randint(n_steps[0], n_steps[1])
            self._logger.info(
                f"\t\t\tUsing following config values: sr={sr}, "
                f"bins_per_octave={bins_per_octave} and n_steps={n_steps}.")
        else:
            self._logger.info(
                f"\t\t\tUsing manually set values: sr={sr}, bins_per_octave={bins_per_octave} and n_steps={n_steps}.")
        self._report["PITCH_SHIFT"] = {"SR": sr, "BINS_PER_OCTAVE": bins_per_octave, "N_STEPS": n_steps}
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, bins_per_octave=bins_per_octave, n_steps=n_steps)
        self._logger.info(f"\t\t\tFinished pitch shifting.")

        return y_shifted

    def volume_control(self, gain_dB: int, p: float = 0) -> np.ndarray:
        """
        This method controls volume of input signal.
        Args:
            gain_dB: Volume gain in dB.
            p: Probability of applying volume control.
    
        Returns: The audio time-series with increased or decreased volume.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping applying volume control...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting volume control...")
        if not (self._config or gain_dB):
            raise Exception(f"You need to pass either a config or an argument.")

        y = self._input_signal.copy()

        if self._config:
            gain_dB = self._config["augmentation_method"]["volume_control"]["gain_dB"]
            if isinstance(gain_dB, list):
                gain_dB = random.randint(gain_dB[0], gain_dB[1])
            self._logger.info(f"\t\t\tUsing config values for gain_dB: {gain_dB}.")
        else:
            self._logger.info(f"\t\t\tUsing manually set value: gain_dB={gain_dB}.")

        self._report["VOLUME_CONTROL"] = {"GAIN_DB": gain_dB}
        volume_ratio = 10 ** (gain_dB / 20)
        y_vol_ctrl = y * volume_ratio
        self._logger.info(f"\t\t\tFinished volume control.")

        return y_vol_ctrl

    def time_shift(self, shift_pct: int, p: float = 0) -> np.ndarray:
        """
        This method shifts input signal in time by given percentage
        Args:
            shift_pct: Time shift in percents.
            p: Probability of applying time shifting.

        Returns: Time shifted audio time-series.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping time shifting...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting time shifting...")
        if not (self._config or shift_pct):
            raise Exception(f"You need to pass either a config or an argument.")

        y = self._input_signal.copy()

        if self._config:
            shift_pct = self._config["augmentation_method"]["time_shift"]["shift_pct"]
            if isinstance(shift_pct, list):
                shift_pct = round(random.uniform(shift_pct[0], shift_pct[1]), 1)
            self._logger.info(f"\t\t\tUsing config values for shift_pct: {shift_pct}.")
        else:
            self._logger.info(f"\t\t\tUsing manually set value: shift_pct={shift_pct}.")

        self._report["TIME_SHIFT"] = {"SHIFT_PCT": shift_pct}
        n_shift = int(shift_pct / 100 * y.shape[0])
        if n_shift > 0:
            y_shift = np.pad(y, (n_shift, 0), mode="constant")
        else:
            y_shift = np.pad(y, (0, -n_shift), mode="constant")
        self._logger.info(f"\t\t\tFinished time shifting.")

        return y_shift

    def add_noise(self, snr_db: int, p: float = 0) -> np.ndarray:
        """
        This method adds noise by specified snr.
        Args:
            snr_db: Signal-Noise-Ratio (SNR)
            p: Probability of adding noise.

        Returns: The audio time-series with added white noise.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping adding noise...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting adding noise...")
        if not (self._config or snr_db):
            raise Exception(f"You need to pass either a config or an argument.")

        y = self._input_signal.copy()
        if self._config:
            snr_db = self._config["augmentation_method"]["add_noise"]["snr_db"]
            if isinstance(snr_db, list):
                snr_db = random.randint(snr_db[0], snr_db[1])
            self._logger.info(f"\t\t\tUsing config values for snr_db: {snr_db}.")
        else:
            self._logger.info(f"\t\t\tUsing manually set value snr_db: {snr_db}.")
        self._report["ADD_NOISE"] = {"SNR_DB": snr_db}
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr_db / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()

        y_noise = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        self._logger.info(f"\t\t\tFinished adding noise.")
        return y_noise

    def polarity_inversion(self, p: float = 0) -> np.ndarray:
        """
        This method performs polarity inversion.
        Args:
            p: Probability of applying polarity inversion.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping applying polarity inversion...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting polarity inversion...")
        y = self._input_signal.copy()
        self._report["POLARITY_INVERSION"] = True
        y_inverse = -1 * y
        self._logger.info(f"\t\t\tFinished polarity inversion.")

        return y_inverse

    def add_reverb(self, sr: int, delay: float, rt60: float, room_dim: np.ndarray, src_pos: np.ndarray,
                   mic_pos: np.ndarray, p: float = 0) -> np.ndarray:
        """
        This method adds reverberation to a given signal.
        Args:
            sr: Sampling frequency
            delay: A time delay until the source signal starts in the simulation
            rt60: Desired RT60 (time it takes to go from full amplitude to 60 db decay) in seconds
            room_dim: Room dimensions (L, W, H)
            src_pos: Source (Speaker) position (L, W, H)
            mic_pos: Microphone position (L, W, H)
            p: Probability of adding reverb.

        Returns: The audio time-series with added reverberation.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping adding reverb...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting adding reverb...")

        if not (self._config or sr or rt60 or room_dim or src_pos or mic_pos):
            raise Exception(f"You need to pass either a config or arguments.")

        if self._config:
            sr = self._config["augmentation_method"]["add_reverb"]["sr"]
            delay = self._config["augmentation_method"]["add_reverb"]["delay"]
            rt60 = self._config["augmentation_method"]["add_reverb"]["rt60"]
            room_dim = self._config["augmentation_method"]["add_reverb"]["room_dim"]
            src_pos = self._config["augmentation_method"]["add_reverb"]["src_pos"]
            mic_pos = self._config["augmentation_method"]["add_reverb"]["mic_pos"]
            self._logger.info(
                f"\t\t\tUsing config values for sr: {sr}, delay: {delay}, rt60: {rt60}, room_dim: {room_dim}, src_pos: {src_pos}, mic_pos: {mic_pos}.")
        else:
            self._logger.info(
                f"\t\t\tUsing manually set values for sr: {sr}, delay: {delay}, rt60: {rt60}, room_dim: {room_dim}, src_pos: {src_pos}, mic_pos: {mic_pos}.")

        y = self._input_signal.copy()
        self._report["ADD_REVERB"] = {"DELAY": delay, "RT60": rt60, "ROOM_DIM": room_dim,
                                      "SRC_POS": src_pos, "MIC_POS": mic_pos}

        # Room properties
        e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

        # Create a room
        room = pra.ShoeBox(
            room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order
        )

        # Add sources
        room.add_source(src_pos, signal=y, delay=delay)

        # Add receiver
        mic_loc = np.c_[mic_pos]

        # Place the mic in the room
        room.add_microphone_array(mic_loc)

        # Simulate reverberation
        room.simulate()
        y_reverb = room.mic_array.signals[0]

        return y_reverb

    def bandpass_filter(self, lowcut, highcut, sr, filter_order=5, p: float = 0):
        """
        This method performs band pass filtering.
        Args:
            lowcut: Lowcut frequency
            highcut: Highcut frequency
            sr: Sampling frequency
            filter_order: Filter order
            p: Probability of applying bandpass filtering.

        Returns: The audio time-series filtered by bandpass filter.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping applying bandpass filtering...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting bandpass filtering...")

        if not (self._config or lowcut or highcut or sr or filter_order):
            raise Exception(f"You need to pass either a config or arguments.")

        if self._config:
            lowcut = self._config["augmentation_method"]["bandpass_filter"]["lowcut"]
            highcut = self._config["augmentation_method"]["bandpass_filter"]["highcut"]
            filter_order = self._config["augmentation_method"]["bandpass_filter"]["filter_order"]
            sr = self._config["augmentation_method"]["bandpass_filter"]["sr"]
            self._logger.info(
                f"\t\t\tUsing config values for lowcut: {lowcut}, highcut: {highcut}, filter order: {filter_order}, sr: {sr}.")
        else:
            self._logger.info(
                f"\t\t\tUsing manually set values for lowcut: {lowcut}, highcut: {highcut}, filter order: {filter_order}, sr: {sr}.")

        y = self._input_signal.copy()
        self._report["BANDPASS_FILTERING"] = {"LOW_CUT": lowcut, "HIGH_CUT": highcut, "SR": sr,
                                              "FILTER_ORDER": filter_order}

        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = sg.butter(filter_order, [low, high], btype='band')

        y_filtered = sg.lfilter(b, a, y)
        self._logger.info(f"\t\t\tFinished bandpass filtering...")
        return y_filtered

    def lowpass_filter(self, lowcut, sr, order=5, p: float = 0):
        """
        This method performs low pass filtering.
        Args:
            lowcut: Lowcut frequency
            sr: Sampling frequency
            order: Filter order
            p: Probability of applying lowpass filtering.

        Returns: The audio time-series filtered by lowpass filter.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping lowpass filtering...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting lowpass filtering...")

        if not (self._config or lowcut or sr or order):
            raise Exception(f"You need to pass either a config or arguments.")

        if self._config:
            lowcut = self._config["augmentation_method"]["bandpass_filter"]["lowcut"]
            order = self._config["augmentation_method"]["bandpass_filter"]["filter_order"]
            sr = self._config["augmentation_method"]["bandpass_filter"]["sr"]
            self._logger.info(
                f"\t\t\tUsing config values for lowcut: {lowcut}, filter order: {order}, sr: {sr}.")
        else:
            self._logger.info(
                f"\t\t\tUsing manually set values for lowcut: {lowcut}, filter order: {order}, sr: {sr}.")

        y = self._input_signal.copy()
        self._report["LOWPASS_FILTERING"] = {"LOW_CUT": lowcut, "SR": sr, "FILTER_ORDER": order}

        b, a = sg.butter(order, lowcut, fs=sr, btype='low', analog=False)
        y_filtered = sg.lfilter(b, a, y)
        self._logger.info(f"\t\t\tFinished lowpass filtering...")
        return y_filtered

    def highpass_filter(self, highcut, sr, order=5, p: float = 0):
        """
        This method performs hith pass filtering.
        Args:
            highcut: Lowcut frequency
            sr: Sampling frequency
            order: Filter order
            p: Probability of applying highpass filtering.

        Returns: The audio time-series filtered by lowpass filter.
        """

        p = get_random_probability(p)
        if p == 0:
            self._logger.info(f"\t\t\tSkipping highpass filtering...")
            return self._input_signal

        self._logger.info(f"\t\t\tStarting highpass filtering...")

        if not (self._config or highcut or sr or order):
            raise Exception(f"You need to pass either a config or arguments.")

        if self._config:
            order = self._config["augmentation_method"]["bandpass_filter"]["filter_order"]
            sr = self._config["augmentation_method"]["bandpass_filter"]["sr"]
            self._logger.info(
                f"\t\t\tUsing config values for highcut: {highcut}, filter order: {order}, sr: {sr}.")
        else:
            self._logger.info(
                f"\t\t\tUsing manually set values for highcut: {highcut}, filter order: {order}, sr: {sr}.")

        y = self._input_signal.copy()
        self._report["HIGH_FILTERING"] = {"HIGH_CUT": highcut, "SR": sr, "FILTER_ORDER": order}

        nyq = 0.5 * sr
        normal_cutoff = highcut / nyq
        b, a = sg.butter(order, normal_cutoff, btype='high', analog=False)
        y_filtered = sg.filtfilt(b, a, y)
        return y_filtered


def get_random_probability(p: float):
    if p == 1:
        return 1
    elif p == 0:
        return 0
    else:
        tmp_p = random.uniform(0, 1)
        return 1 if tmp_p > p else 0
