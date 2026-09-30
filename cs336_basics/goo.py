from cs336_basics.net.linear import Linear
from tests.adapters import run_linear
from einops import einsum
import torch

if __name__ == "__main__":
    W = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    x = torch.tensor([1.0, 2.0, 3.0])


    #  Try an x of shape (4, 3) in goo.py

    print(W.shape)
    print(x.shape)

    out = run_linear(3, 2, W, x)
    # y.forward(x)
    print(out)  # Should print [7, 5]
