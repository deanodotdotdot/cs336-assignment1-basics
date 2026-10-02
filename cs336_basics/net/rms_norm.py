import torch
from einops import reduce
import numpy as np

class RMSNorm(torch.nn.modules.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None=None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.ones([d_model], device=device, dtype=dtype))

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        in_dtype = x_in.dtype
        x = x_in.to(torch.float32)

        inner = (x * x)
        rms = torch.sqrt(reduce(inner, '... d_model -> ...', 'mean') + self.eps ) 
        rms = rms.unsqueeze(-1) 
   
        result = (x / rms) * self.weight
        return result.to(in_dtype)