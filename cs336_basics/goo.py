from cs336_basics.net.linear import Linear
from tests.adapters import run_linear
from einops import einsum
import torch

if __name__ == "__main__":
    W = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    x = torch.tensor([1.0, 2.0, 3.0])
    temp = torch.randn(5, 4)


    #  Try an x of shape (4, 3) in goo.py

    print(W.shape)
    print(x.shape)

    out = run_linear(3, 2, W, x)
    # y.forward(x)
    print(temp)  # Should print [7, 5]

    # 1. Index with a single integer
    print(temp[1])

    # 2. Index with a 1D tensor of a few IDs
    idx_1d = torch.tensor([0, 2, 4])
    print(temp[idx_1d])

    # 3. Index with a 2D tensor of IDs
    idx_2d = torch.tensor([[0, 1], [2, 3]])
    print(temp[idx_2d])
