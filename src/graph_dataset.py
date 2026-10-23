"""Stage B2: graph + protein dataset for PyG DataLoader."""
import os, glob, torch, pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from common import *
def load_graphs():
    g = {}
    for f in sorted(glob.glob(os.path.join(GRAPHS, 'graphs_*.pt'))): g.update(torch.load(f, weights_only=False))
    return g
class AffinityDataset(Dataset):
    def __init__(self, csv, mtype=None, graphs=None, max_rows=None, seed=42):
        df = pd.read_csv(csv)
        if mtype: df = df[df.mtype == mtype]
        self.graphs = graphs if graphs is not None else load_graphs()
        df = df[df.smiles.isin(self.graphs)]
        if max_rows and len(df) > max_rows: df = df.sample(max_rows, random_state=seed)
        self.df = df.reset_index(drop=True); self._p = {}
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]; g = self.graphs[r.smiles]
        if r.seq not in self._p: self._p[r.seq] = torch.tensor([encode_protein(r.seq)], dtype=torch.long)
        return Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, pos=g.pos,
                    prot=self._p[r.seq], smiles=r.smiles, y=torch.tensor([r.label], dtype=torch.float))

