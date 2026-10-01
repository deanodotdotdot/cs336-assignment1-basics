# Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear
# transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear
# module, except for not having a bias argument or parameter. We recommend the following
# interface:
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

        w = torch.nn.init.trunc_normal_(empty, 0, self.sigma, -3 * self.sigma, 3 * self.sigma)
        self.w = torch.nn.Parameter(w)
        
    # Apply the linear transformation to the input.
    # Make sure to:
    
    # construct and store your parameter as 𝑊 (not 𝑊 ⊤), putting it in an nn.Parameter
    # To test your Linear module, implement the test adapter at [adapters.run_linear] . The adapter
    # should load the given weights into your Linear module. You ca


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dot_product = einsum(self.w, x, 'd_out d_in, ... d_in  -> ... d_out')
        return dot_product