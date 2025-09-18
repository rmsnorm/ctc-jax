# Utility for computing MFCC features for a wav file

import dataclasses
import enum
import json

import numpy as np
import scipy as sp


class WindowType(enum.Enum):
    HAMMING = 0
    HANNING = 1


@dataclasses.dataclass
class MfccConfig:
    """Config for MFCC computation."""

    # sampling rate
    fs: int
    # windowing
    window_ms: int
    window_shift_ms: int
    window_type: WindowType
    # spectrum
    fft_window_len: int
    # mel config
    num_mel_filters: int
    low_freq_hz: int
    compute_delta: bool
    compute_delta_delta: bool

    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return cls(
            fs=data["fs"],
            window_ms=data["window_ms"],
            window_shift_ms=data["window_shift_ms"],
            window_type=data["window_type"],
            fft_window_len=data["fft_window_len"],
            num_mel_filters=data["num_mel_filters"],
            low_freq_hz=data["low_freq_hz"],
            compute_delta=data["compute_delta"],
            compute_delta_delta=data["compute_delta_delta"],
        )


PI = np.pi


def raised_cosine_window(a0: float, n: int):
    """General form for raised cosine windows."""
    window = np.zeros(n) + a0 - (1 - a0) * np.cos(2 * PI * np.arange(n) / (n * 1.0))
    return window


def hanning_window(n):
    """Hanning window."""
    return raised_cosine_window(0.5, n)


def hamming_window(n):
    """Hamming window."""
    return raised_cosine_window(0.54, n)


def hz2mel(f):
    """Convert Hz to Mel"""
    return 2595.0 * np.log10(1 + f / 700.0)


def mel2hz(m):
    """Convert Mel to Hz"""
    return 700.0 * (10 ** (m / 2595.0) - 1)


class MfccComputer:
    """Class for computing MFCC."""

    def __init__(self, cfg: MfccConfig):
        self.cfg = cfg

        self.window_len = int(self.cfg.fs * self.cfg.window_ms / 1000.0)
        self.window_shift = int(self.cfg.fs * self.cfg.window_shift_ms / 1000.0)

        if self.cfg.window_type == WindowType.HAMMING:
            self.window = hamming_window(self.window_len)
        else:
            self.window = hanning_window(self.window_len)

        # initialize mel-filter-bank
        self.mel_filter_bank_matrix = self._compute_mel_filter_bank_matrix()

    def _mel_filter_bank_centers(self):
        high_freq_hz = self.cfg.fs // 2

        low_freq_mel = hz2mel(self.cfg.low_freq_hz)
        high_freq_mel = hz2mel(high_freq_hz)

        diff_mel = high_freq_mel - low_freq_mel

        filter_bank_centers = []

        num_filters = self.cfg.num_mel_filters
        for i in range(num_filters + 2):
            m = low_freq_mel + diff_mel / (num_filters) * i
            filter_bank_centers.append(mel2hz(m))

        return filter_bank_centers

    def _compute_mel_filter_bank_matrix(self):
        filter_bank_centers = self._mel_filter_bank_centers()

        filter_bank_coeff = []

        fft_window_len = self.cfg.fft_window_len
        fs = self.cfg.fs
        freqs = np.fft.fftfreq(fft_window_len + 1, 1 / fs)[: (fft_window_len + 1) // 2]

        for i in range(1, len(filter_bank_centers) - 1):
            fbc = filter_bank_centers[i] * 1.0
            fbc_prev = filter_bank_centers[i - 1] * 1.0
            fbc_next = filter_bank_centers[i + 1] * 1.0
            coeffs = []
            for f in freqs:
                if f < fbc_prev:
                    r = 0
                elif f >= fbc_prev and f < fbc:
                    r = (f - fbc_prev) / (fbc - fbc_prev)
                elif f > fbc and f <= fbc_next:
                    r = (fbc_next - f) / (fbc_next - fbc)
                else:
                    r = 0
                coeffs.append(r)
            filter_bank_coeff.append(coeffs)

        return np.array(filter_bank_coeff)

    def _compute_spectrum(self, x: np.array):
        # pre-emphasis
        x = x[1:] - 0.97 * x[:-1]

        # window_len = self.window_len
        window_shift = self.window_shift

        x_windowed = np.lib.stride_tricks.sliding_window_view(x, self.window.shape)[
            ::window_shift
        ]
        x_windowed = x_windowed * self.window[np.newaxis, :]

        x_windowed_fft = np.fft.fft(x_windowed, self.cfg.fft_window_len, axis=-1)
        x_windowed_psd = (
            1
            / (self.cfg.fs * self.cfg.fft_window_len)
            * np.absolute(x_windowed_fft) ** 2
        )

        psd_one_sided = x_windowed_psd[:, : (self.cfg.fft_window_len + 1) // 2] * 2

        freqs = np.fft.fftfreq(self.cfg.fft_window_len + 1, 1 / self.cfg.fs)[
            : (self.cfg.fft_window_len + 1) // 2
        ]

        num_frames = psd_one_sided.shape[0]
        frame_times = np.arange(num_frames) * (self.cfg.window_shift_ms / 1000.0)

        return psd_one_sided, freqs, frame_times

    def __call__(self, x: np.array):
        psd, _, _ = self._compute_spectrum(x)

        log_mel_psd = np.log10(psd @ self.mel_filter_bank_matrix.T)

        # compute the DCT (which is equivalent to the IFFT of psd.)
        mel_cepstrum = sp.fft.dct(log_mel_psd, type=2, axis=-1)

        mfcc_13 = mel_cepstrum[:, :13]
        if self.cfg.compute_delta:
            mfcc_delta = (mfcc_13[2:] - mfcc_13[:-2]) / 2.0
            if self.cfg.compute_delta_delta:
                mfcc_delta_delta = (mfcc_delta[2:] - mfcc_delta[:-2]) / 2.0
                mfcc = np.stack(
                    (mfcc_13[4:], mfcc_delta[2:], mfcc_delta_delta), axis=-1
                ).reshape(mfcc_delta_delta.shape[0], 39)
                return mfcc
            mfcc = np.stack((mfcc_13[2:], mfcc_delta), axis=-1).reshape(
                mfcc_delta.shape[0], 26
            )
            return mfcc
        return mfcc_13
