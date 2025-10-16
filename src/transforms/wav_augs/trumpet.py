import librosa
import torch
import torchaudio
from torch import Tensor, nn


class Trumpet(nn.Module):
    def __init__(self, level=20):
        super().__init__()
        filename = librosa.ex("trumpet")
        noise, _ = librosa.load(filename)
        self.noise = torch.from_numpy(noise)
        self.level = level

    def __call__(self, data: Tensor):
        noise = self.noise.to(data.device)
        if noise.shape[-1] < data.shape[-1]:
            rep = data.shape[-1] // noise.shape[-1] + 1
            noise = noise.repeat(rep)[... : data.shape[-1]]
        else:
            noise = noise[... : data.shape[-1]]
        noize_energy = torch.norm(noise)
        audio_energy = torch.norm(data)

        alpha = (audio_energy / noize_energy) * (10 ** (-self.level / 20))

        augumented_wav = data + alpha * noise

        return torch.clamp(augumented_wav, -1, 1)
