import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None=None, dtype: torch.dtype | None = None):
        super().__init__()
        empty = torch.empty([num_embeddings, embedding_dim], device=device, dtype=dtype)

        weight = torch.nn.init.trunc_normal_(empty, 0, 1, -3, 3)
        self.weight = torch.nn.Parameter(weight)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]