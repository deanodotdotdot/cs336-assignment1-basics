import torch
from einops import einsum
import numpy as np

class Linear(torch.nn.modules.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None, dtype: torch.dtype | None = None):
        super().__init__()

        empty = torch.empty([out_features, in_features],device=device, dtype=dtype)

        self.sigma = np.sqrt(2 / (in_features + out_features))
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.nn.init.trunc_normal_(empty, 0, self.sigma, -3 * self.sigma, 3 * self.sigma)
        self.weight = torch.nn.Parameter(weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dot_product = einsum(self.weight, x, 'd_out d_in, ... d_in  -> ... d_out')
        return dot_product