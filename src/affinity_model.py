import torch, torch.nn as nn
from common import ATOM_DIM, BOND_DIM
from protein_encoder import ProteinEncoder
from graph_transformer_3d import GraphTransformer3D
from cross_attention import CrossAttention
def mpool(x, m): return (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
class AffinityModel(nn.Module):
    def __init__(s, d=128, h=4, layers=3, drop=0.1):
        super().__init__(); s.cfg = dict(d=d, h=h, layers=layers, drop=drop)
        s.drug = GraphTransformer3D(ATOM_DIM, BOND_DIM, d, h, layers); s.prot = ProteinEncoder(d)
        s.cross = CrossAttention(d, h, drop)
        s.head = nn.Sequential(nn.Linear(3*d, 2*d), nn.GELU(), nn.Dropout(drop), nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, 1))
    def forward(s, data):
        dx, dm = s.drug(data); px, pm = s.prot(data.prot); c, _ = s.cross(dx, px, pm)
        return s.head(torch.cat([mpool(c, dm), mpool(dx, dm), mpool(px, pm)], -1)).squeeze(-1)
