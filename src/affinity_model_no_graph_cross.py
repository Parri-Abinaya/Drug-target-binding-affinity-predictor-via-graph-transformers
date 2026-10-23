import torch
import torch.nn as nn

from common import ATOM_DIM, BOND_DIM
from protein_encoder import ProteinEncoder
from graph_transformer_3d import GraphTransformer3D
from cross_attention import CrossAttention
from smiles_encoder import SMILESEncoder, encode_smiles


def mpool(x, mask):
    mask = mask.to(x.dtype)
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(
        1, keepdim=True
    ).clamp(min=1)


class AblationNoGraphCross(nn.Module):
    """Multimodal model without graph-to-protein cross-attention."""

    def __init__(self, d=128, h=4, layers=3, drop=0.1):
        super().__init__()
        self.cfg = dict(d=d, h=h, layers=layers, drop=drop)

        self.drug = GraphTransformer3D(ATOM_DIM, BOND_DIM, d, h, layers)
        self.smiles = SMILESEncoder(d=d, heads=h, dropout=drop)
        self.prot = ProteinEncoder(d)

        self.smiles_cross = CrossAttention(d, h, drop)

        # Four pooled feature vectors: graph, SMILES context,
        # SMILES tokens, and protein.
        self.head = nn.Sequential(
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(self, data):
        dx, dm = self.drug(data)
        px, pm = self.prot(data.prot)

        smiles_list = data.smiles
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        token_ids = torch.stack([
            encode_smiles(s) for s in smiles_list
        ]).to(dx.device)

        sx, sm = self.smiles(token_ids)
        smiles_context, _ = self.smiles_cross(sx, px, pm)

        features = torch.cat([
            mpool(dx, dm),
            mpool(smiles_context, sm),
            mpool(sx, sm),
            mpool(px, pm),
        ], dim=-1)

        return self.head(features).squeeze(-1)
