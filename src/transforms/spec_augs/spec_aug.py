import torchaudio
from torch import nn


class SpecAug(nn.Module):
    def __init__(self, freq=20, time=100):
        super().__init__()
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq),
            torchaudio.transforms.TimeMasking(time),
        )

    def __call__(self, log_spec):
        return self.specaug(log_spec)
