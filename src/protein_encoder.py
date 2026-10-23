import torch, torch.nn as nn
class ProteinEncoder(nn.Module):
    """Residue embedding + residual 1D CNN -> per-residue features [B,L,D] and mask [B,L]."""
    def __init__(s, d=128, layers=3, k=7):
        super().__init__(); s.emb = nn.Embedding(21, d, padding_idx=0)
        s.convs = nn.ModuleList(nn.Conv1d(d, d, k, padding=k//2) for _ in range(layers)); s.norm = nn.LayerNorm(d)
    def forward(s, ids):
        mask = ids > 0; x = s.emb(ids)
        for c in s.convs: x = x + torch.relu(c(x.transpose(1, 2))).transpose(1, 2)
        return s.norm(x) * mask.unsqueeze(-1), mask
