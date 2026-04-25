import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class RepoEncoder(nn.Module):
    def __init__(self, nomic_dim: int = 768, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.nomic_proj = nn.Linear(nomic_dim, hidden_dim)
        self.star_proj = nn.Linear(1, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_nomic: torch.Tensor, x_stars: torch.Tensor) -> torch.Tensor:
        x = self.nomic_proj(x_nomic) + self.star_proj(x_stars)
        return self.dropout(x)


def _make_conv(conv_type: str, hidden_dim: int, heads: int = 1):
    if conv_type == "sage":
        return SAGEConv((-1, -1), hidden_dim)
    elif conv_type == "gat":
        return GATConv((-1, -1), hidden_dim // heads, heads=heads, add_self_loops=False)
    raise ValueError(f"Unknown conv_type: {conv_type}")


class BipartiteSAGE(nn.Module):
    """
    Variable-depth heterogeneous GNN on user<->repo bipartite graph.

    With num_hops=K, message passing alternates K times:
      hop 1: repo -> user
      hop 2: user -> repo
      hop 3: repo -> user
      ...
    Odd hops: repo -> user.  Even hops: user -> repo.
    """

    def __init__(self, hidden_dim: int = 64, num_hops: int = 3, dropout: float = 0.1,
                 conv_type: str = "sage", gat_heads: int = 4, use_layernorm: bool = False):
        super().__init__()
        self.num_hops = num_hops
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_hops):
            self.convs.append(_make_conv(conv_type, hidden_dim, gat_heads))
            self.norms.append(nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity())

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_repo, x_user, edge_index_r2u, edge_index_u2r):
        h_repo = x_repo
        h_user = x_user

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            is_last = (i == self.num_hops - 1)
            if i % 2 == 0:
                h_user_new = conv((h_repo, h_user), edge_index_r2u)
                h_user_new = norm(h_user_new)
                if not is_last:
                    h_user_new = self.dropout(F.relu(h_user_new))
                h_user = h_user_new
            else:
                h_repo_new = conv((h_user, h_repo), edge_index_u2r)
                h_repo_new = norm(h_repo_new)
                if not is_last:
                    h_repo_new = self.dropout(F.relu(h_repo_new))
                h_repo = h_repo_new

        return h_user, h_repo


class InductivePinSage(nn.Module):
    def __init__(self, nomic_dim: int = 768, hidden_dim: int = 64, num_hops: int = 3,
                 dropout: float = 0.1, conv_type: str = "sage", gat_heads: int = 4,
                 use_layernorm: bool = False, normalize_output: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.normalize_output = normalize_output
        self.repo_encoder = RepoEncoder(nomic_dim, hidden_dim, dropout=dropout)
        self.gnn = BipartiteSAGE(hidden_dim, num_hops, dropout, conv_type, gat_heads, use_layernorm)

    def forward(self, x_nomic, x_stars, edge_index_u2r, edge_index_r2u, num_users):
        x_repo = self.repo_encoder(x_nomic, x_stars)
        x_user = self._init_user_features(x_repo, edge_index_u2r, num_users)
        h_user, h_repo = self.gnn(x_repo, x_user, edge_index_r2u, edge_index_u2r)
        if self.normalize_output:
            h_user = F.normalize(h_user, dim=-1)
            h_repo = F.normalize(h_repo, dim=-1)
        return h_user, h_repo

    def _init_user_features(self, x_repo, edge_index_u2r, num_users):
        user_idx = edge_index_u2r[0]
        repo_idx = edge_index_u2r[1]
        x_user = torch.zeros(num_users, self.hidden_dim, device=x_repo.device)
        counts = torch.zeros(num_users, 1, device=x_repo.device)
        x_user.scatter_add_(0, user_idx.unsqueeze(1).expand(-1, self.hidden_dim), x_repo[repo_idx])
        counts.scatter_add_(0, user_idx.unsqueeze(1), torch.ones_like(user_idx, dtype=torch.float).unsqueeze(1))
        mask = counts.squeeze(1) > 0
        x_user[mask] = x_user[mask] / counts[mask]
        return x_user

    def predict_scores(self, h_user, h_repo):
        return (h_user * h_repo).sum(dim=-1)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()
