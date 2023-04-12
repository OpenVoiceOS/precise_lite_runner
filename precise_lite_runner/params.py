from math import floor
from dataclasses import dataclass


class Vectorizer:
    mels = 1
    mfccs = 2
    speechpy_mfccs = 3


@dataclass
class ListenerParams:
    window_t: float = 0.1
    hop_t: float = 0.05
    buffer_t: float = 1.5
    sample_rate: int = 16000
    sample_depth: int = 2
    n_mfcc: int = 13
    n_filt: int = 20
    n_fft: int = 512
    use_delta: bool = False
    vectorizer: int = Vectorizer.mfccs
    threshold_config: tuple = ((6, 4),)
    threshold_center: float = 0.2


    @property
    def buffer_samples(self):
        samples = int(self.sample_rate * self.buffer_t + 0.5)
        return self.hop_samples * (samples // self.hop_samples)

    @property
    def n_features(self):
        return 1 + int(floor((self.buffer_samples - self.window_samples) / self.hop_samples))

    @property
    def window_samples(self):
        return int(self.sample_rate * self.window_t + 0.5)

    @property
    def hop_samples(self):
        return int(self.sample_rate * self.hop_t + 0.5)

    @property
    def max_samples(self):
        return int(self.buffer_t * self.sample_rate)

    @property
    def feature_size(self):
        num_features = {
            Vectorizer.mfccs: self.n_mfcc,
            Vectorizer.mels: self.n_filt,
            Vectorizer.speechpy_mfccs: self.n_mfcc
        }[self.vectorizer]
        if self.use_delta:
            num_features *= 2
        return num_features


# Global listener parameters
params = ListenerParams()

