import librosa
import torch
import torchaudio
from torch import Tensor, nn


class Bedroom(nn.Module):
    def __init__(self, level=20):
        super().__init__()
        filename = librosa.ex("trumpet")
        self.noise, self.sr = librosa.load(filename)
        self.level = Tensor([level])

    def __call__(self, data: Tensor):
        noize_energy = torch.norm(torch.from_numpy(self.noise))
        audio_energy = torch.norm(data)

        alpha = (audio_energy / noize_energy) * torch.pow(self.level / 20)

        clipped_wav = data[..., : self.noise.shape[0]]

        augumented_wav = clipped_wav + alpha * torch.from_numpy(self.noise)

        return torch.clamp(augumented_wav, -1, 1)
