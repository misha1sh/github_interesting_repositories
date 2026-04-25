import torch
import torch.nn as nn
import torch.nn.functional as F


class TagEmbeddingModel(nn.Module):
    def __init__(self, num_tags: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.tag_embeddings = nn.Embedding(num_tags, embed_dim)
        
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.net = None
        
        nn.init.normal_(self.tag_embeddings.weight, std=0.01)
    
    def forward(self, tag_indices: torch.Tensor, tag_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tag_indices: [batch_size, max_tags] - padded tag indices
            tag_weights: [batch_size, max_tags] - weights for each tag (0 for padding)
        
        Returns:
            [batch_size, embed_dim] - normalized embeddings
        """
        tag_embs = self.tag_embeddings(tag_indices)
        
        if self.net is not None:
            tag_embs = self.net(tag_embs)
        
        weighted_embs = tag_embs * tag_weights.unsqueeze(-1)
        summed = weighted_embs.sum(dim=1)
        
        counts = tag_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        averaged = summed / counts
        
        return F.normalize(averaged, dim=-1)
