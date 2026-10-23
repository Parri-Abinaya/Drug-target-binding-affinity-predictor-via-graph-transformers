import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch
class SpatialLayer(nn.Module):
    """Self-attention whose logits get a learned per-head bias from RBF-expanded interatomic distances."""
    def __init__(s, d, h, nrbf=16, drop=0.1, rmax=12.0):
        super().__init__(); s.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        s.bias = nn.Linear(nrbf, h); s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Dropout(drop), nn.Linear(2*d, d))
        s.register_buffer('mu', torch.linspace(0, rmax, nrbf)); s.gamma = (nrbf / rmax) ** 2
    def forward(s, x, dist, pad):
        B, N, _ = x.shape
        b = s.bias(torch.exp(-s.gamma * (dist.unsqueeze(-1) - s.mu) ** 2)).permute(0, 3, 1, 2)  # B,H,N,N
        b = b.masked_fill(pad[:, None, None, :], -1e4).reshape(-1, N, N)
        x = s.n1(x + s.attn(x, x, x, attn_mask=b, need_weights=False)[0])
        return s.n2(x + s.ff(x))
class GraphTransformer3D(nn.Module):
    """Bond-aware GINE layer (topology) -> spatially-biased transformer layers (3D geometry)."""
    def __init__(s, atom_dim, bond_dim, d=128, h=4, layers=3):
        super().__init__(); s.inp = nn.Linear(atom_dim, d)
        s.gine = GINEConv(nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)), edge_dim=bond_dim)
        s.layers = nn.ModuleList(SpatialLayer(d, h) for _ in range(layers))
    def forward(s, data):
        x = F.relu(s.gine(s.inp(data.x), data.edge_index, data.edge_attr))
        x, mask = to_dense_batch(x, data.batch); pos, _ = to_dense_batch(data.pos, data.batch)
        dist = torch.cdist(pos, pos)
        for l in s.layers: x = l(x, dist, ~mask)
        return x, mask
