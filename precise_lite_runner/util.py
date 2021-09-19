import hashlib
import platform
import wave
from math import exp, log, sqrt, pi
from os.path import join, dirname, abspath
from subprocess import Popen
from typing import *

import numpy as np
import wavio

from precise_lite_runner.params import pr


class ThresholdDecoder:
    """
    Decode raw network output into a relatively linear threshold using
    This works by estimating the logit normal distribution of network
    activations using a series of averages and standard deviations to
    calculate a cumulative probability distribution
    """

    def __init__(self, mu_stds: Tuple[Tuple[float, float]], center=0.5,
                 resolution=200, min_z=-4, max_z=4):
        self.min_out = int(min(mu + min_z * std for mu, std in mu_stds))
        self.max_out = int(max(mu + max_z * std for mu, std in mu_stds))
        self.out_range = self.max_out - self.min_out
        self.cd = np.cumsum(self._calc_pd(mu_stds, resolution))
        self.center = center

    def decode(self, raw_output: float) -> float:
        if raw_output == 1.0 or raw_output == 0.0:
            return raw_output
        if self.out_range == 0:
            cp = int(raw_output > self.min_out)
        else:
            ratio = (asigmoid(raw_output) - self.min_out) / self.out_range
            ratio = min(max(ratio, 0.0), 1.0)
            cp = self.cd[int(ratio * (len(self.cd) - 1) + 0.5)]
        if cp < self.center:
            return 0.5 * cp / self.center
        else:
            return 0.5 + 0.5 * (cp - self.center) / (1 - self.center)

    def encode(self, threshold: float) -> float:
        threshold = 0.5 * threshold / self.center
        if threshold < 0.5:
            cp = threshold * self.center * 2
        else:
            cp = (threshold - 0.5) * 2 * (1 - self.center) + self.center
        ratio = np.searchsorted(self.cd, cp) / len(self.cd)
        return sigmoid(self.min_out + self.out_range * ratio)

    def _calc_pd(self, mu_stds, resolution):
        points = np.linspace(self.min_out, self.max_out,
                             resolution * self.out_range)
        return np.sum([pdf(points, mu, std) for mu, std in mu_stds],
                      axis=0) / (resolution * len(mu_stds))


class InvalidAudio(ValueError):
    pass


def sigmoid(x):
    """Sigmoid squashing function for scalars"""
    return 1 / (1 + exp(-x))


def asigmoid(x):
    """Inverse sigmoid (logit) for scalars"""
    return -log(1 / x - 1)


def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    if std == 0:
        return 0
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(
        -(x - mu) ** 2 / (2 * std ** 2))


def chunk_audio(audio: np.ndarray, chunk_size: int) -> Generator[
    np.ndarray, None, None]:
    for i in range(chunk_size, len(audio), chunk_size):
        yield audio[i - chunk_size:i]


def buffer_to_audio(buffer: bytes) -> np.ndarray:
    """Convert a raw mono audio byte string to numpy array of floats"""
    return np.fromstring(buffer, dtype='<i2').astype(np.float32,
                                                     order='C') / 32768.0


def audio_to_buffer(audio: np.ndarray) -> bytes:
    """Convert a numpy array of floats to raw mono audio"""
    return (audio * 32768).astype('<i2').tostring()


def load_audio(file: Any) -> np.ndarray:
    """
    Args:
        file: Audio filename or file object
    Returns:
        samples: Sample rate and audio samples from 0..1
    """

    try:
        wav = wavio.read(file)
    except (EOFError, wave.Error):
        wav = wavio.Wav(np.array([[]], dtype=np.int16), 16000, 2)
    if wav.data.dtype != np.int16:
        raise InvalidAudio('Unsupported data type: ' + str(wav.data.dtype))
    if wav.rate != pr.sample_rate:
        raise InvalidAudio('Unsupported sample rate: ' + str(wav.rate))

    data = np.squeeze(wav.data)
    return data.astype(np.float32) / float(np.iinfo(data.dtype).max)


def save_audio(filename: str, audio: np.ndarray):
    save_audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    wavio.write(filename, save_audio, pr.sample_rate,
                sampwidth=pr.sample_depth, scale='none')


def play_audio(filename: str):
    """
    Args:
        filename: Audio filename
    """

    player = 'play' if platform.system() == 'Darwin' else 'aplay'
    Popen([player, '-q', filename])


def activate_notify():
    audio = 'data/activate.wav'
    audio = abspath(dirname(abspath(__file__)) + '/../' + audio)

    play_audio(audio)


def glob_all(folder: str, filt: str) -> List[str]:
    """Recursive glob"""
    import os
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, filt):
            matches.append(os.path.join(root, filename))
    return matches


def find_wavs(folder: str) -> Tuple[List[str], List[str]]:
    """Finds wake-word and not-wake-word wavs in folder"""
    return (glob_all(join(folder, 'wake-word'), '*.wav'),
            glob_all(join(folder, 'not-wake-word'), '*.wav'))


def calc_sample_hash(inp: np.ndarray, outp: np.ndarray) -> str:
    md5 = hashlib.md5()
    md5.update(inp.tostring())
    md5.update(outp.tostring())
    return md5.hexdigest()
