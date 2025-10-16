import torch
import torchaudio
from torch import Tensor, nn


class Bedroom(nn.Module):
    def __init__(self):
        super().__init__()
        self.rir, self.rir_sr = torchaudio.load("h001_Bedroom_65.wav")

    def __call__(self, data: Tensor):
        left_pad = right_pad = self.rir.shape[-1] - 1

        # Since torch.conv do cross-correlation (not convolution) we need to flip kernel
        flipped_rir = self.rir.squeeze().flip(0)

        audio = torch.Functional.pad(data, [left_pad, right_pad]).view(1, 1, -1)
        convolved_audio = torch.conv1d(audio, flipped_rir.view(1, 1, -1)).squeeze()

        # peak normalization
        if convolved_audio.abs().max() > 1:
            convolved_audio /= convolved_audio.abs().max()

        return convolved_audio
