import torch.nn as nn
class CrossAttention(nn.Module):
    """Atoms (queries) attend to protein residues (keys/values)."""
    def __init__(s, d=128, h=4, drop=0.1):
        super().__init__(); s.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True); s.norm = nn.LayerNorm(d)
    def forward(s, drug, prot, prot_mask):
        out, w = s.attn(drug, prot, prot, key_padding_mask=~prot_mask)
        return s.norm(drug + out), w
