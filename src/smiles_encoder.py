"""Character-level SMILES Transformer encoder."""

import torch
import torch.nn as nn


# Character vocabulary for common SMILES notation.
SMILES_CHARS = (
    "#%()*+-./0123456789=@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[]\\"
    "abcdefghijklmnopqrstuvwxyz"
    ":"
)

PAD_ID = 0
UNK_ID = 1

SMILES_VOCAB = {
    char: index + 2
    for index, char in enumerate(SMILES_CHARS)
}

VOCAB_SIZE = len(SMILES_VOCAB) + 2
MAX_SMILES_LEN = 256


def encode_smiles(smiles, max_length=MAX_SMILES_LEN):
    """Convert a SMILES string into fixed-length token IDs."""
    ids = [SMILES_VOCAB.get(char, UNK_ID) for char in str(smiles)]
    ids = ids[:max_length]

    if len(ids) < max_length:
        ids.extend([PAD_ID] * (max_length - len(ids)))

    return torch.tensor(ids, dtype=torch.long)


class SMILESEncoder(nn.Module):
    """Transformer encoder for molecular SMILES strings."""

    def __init__(self, d=128, heads=4, layers=2, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE, d, padding_idx=PAD_ID
        )

        self.position = nn.Embedding(MAX_SMILES_LEN, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=4 * d,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
        )

        self.norm = nn.LayerNorm(d)

    def forward(self, token_ids):
        """
        Args:
            token_ids: LongTensor [batch_size, sequence_length]

        Returns:
            features: FloatTensor [batch_size, sequence_length, d]
            mask: BoolTensor [batch_size, sequence_length]
        """
        token_ids = token_ids[:, :MAX_SMILES_LEN]
        batch_size, seq_len = token_ids.shape

        mask = token_ids.ne(PAD_ID)

        positions = torch.arange(
            seq_len, device=token_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        x = self.embedding(token_ids) + self.position(positions)

        x = self.transformer(
            x,
            src_key_padding_mask=~mask,
        )

        return self.norm(x), mask
