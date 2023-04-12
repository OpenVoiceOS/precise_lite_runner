import hashlib

import numpy as np
from sonopy import mfcc_spec, mel_spec

from precise_lite_runner.params import params, Vectorizer
from precise_lite_runner.util import InvalidAudio

inhibit_t = 0.4
inhibit_dist_t = 1.0
inhibit_hop_t = 0.1

vectorizers = {
    Vectorizer.mels: lambda x: mel_spec(
        x, params.sample_rate, (params.window_samples, params.hop_samples),
        num_filt=params.n_filt, fft_size=params.n_fft
    ),
    Vectorizer.mfccs: lambda x: mfcc_spec(
        x, params.sample_rate, (params.window_samples, params.hop_samples),
        num_filt=params.n_filt, fft_size=params.n_fft, num_coeffs=params.n_mfcc
    ),
    Vectorizer.speechpy_mfccs: lambda x: __import__('speechpy').feature.mfcc(
        x, params.sample_rate, params.window_t, params.hop_t, params.n_mfcc, params.n_filt,
        params.n_fft
    )
}


def vectorize_raw(audio: np.ndarray) -> np.ndarray:
    """Turns audio into feature vectors, without clipping for length"""
    if len(audio) == 0:
        raise InvalidAudio('Cannot vectorize empty audio!')
    return vectorizers[params.vectorizer](audio)


def add_deltas(features: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(features)
    for i in range(1, len(features)):
        deltas[i] = features[i] - features[i - 1]

    return np.concatenate([features, deltas], -1)


def vectorize(audio: np.ndarray) -> np.ndarray:
    """
    Args:
        audio: Audio verified to be of `sample_rate`

    Returns:
        array<float>: Vector representation of audio
    """
    if len(audio) > params.max_samples:
        audio = audio[-params.max_samples:]
    features = vectorize_raw(audio)
    if len(features) < params.n_features:
        features = np.concatenate([
            np.zeros((params.n_features - len(features), features.shape[1])),
            features
        ])
    if len(features) > params.n_features:
        features = features[-params.n_features:]

    return features


def vectorize_delta(audio: np.ndarray) -> np.ndarray:
    return add_deltas(vectorize(audio))


def vectorization_md5_hash(pr):
    """Hash all the fields related to audio vectorization"""
    keys = sorted(pr.__dict__)
    print(keys)
    keys.remove('threshold_config')
    keys.remove('threshold_center')
    return hashlib.md5(
        str([pr.__dict__[i] for i in keys]).encode()
    ).hexdigest()
