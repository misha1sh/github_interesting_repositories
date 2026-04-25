import torch
import torch.nn as nn
import torch.nn.functional as F


class RepoEmbeddingModel(nn.Module):
    def __init__(self, nomic_dim: int = 768, embed_dim: int = 6, hidden_dim: int = 0):
        super().__init__()
        in_dim = nomic_dim + 1
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.net = nn.Linear(in_dim, embed_dim)

    def forward(self, x_nomic: torch.Tensor, x_stars: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_nomic, x_stars], dim=-1)
        return F.normalize(self.net(x), dim=-1)
