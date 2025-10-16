import torch_audiomentations
from torch import Tensor, distributions, nn


class Gaussian(nn.Module):
    def __init__(self, mean=0, var=0.05):
        super().__init__()
        self._aug = distributions.Normal(mean, var)

    def __call__(self, data: Tensor):
        return data + self._aug.sample(data.size())
