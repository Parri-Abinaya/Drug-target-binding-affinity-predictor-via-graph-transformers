"""Shared paths, featurization, metrics."""
import os, re, glob, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
def P(*a): return os.path.join(ROOT, *a)
RAW, PROC, GRAPHS, MODELS = P('data','raw'), P('data','processed'), P('data','graphs'), P('models')
METRICS, PLOTS, LOGS = P('results','metrics'), P('results','plots'), P('logs')
for d in (RAW, PROC, GRAPHS, MODELS, METRICS, PLOTS, LOGS): os.makedirs(d, exist_ok=True)

AA = 'ACDEFGHIKLMNPQRSTVWY'; AA_IDX = {a: i+1 for i, a in enumerate(AA)}
MAX_PROT, MAX_ATOMS = 1000, 80
ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
ATOM_DIM, BOND_DIM = 23, 6
MTYPES = {'ki': 'Ki', 'ic50': 'IC50', 'kd': 'Kd', 'ec50': 'EC50'}
VIRAL_KW = ('virus', 'viral', 'hiv', 'sars', 'hepatitis', 'influenza', 'dengue', 'zika', 'ebola', 'herpes')

def is_viral(org): return any(k in str(org).lower() for k in VIRAL_KW)
def find_col(cols, *keys):
    for c in cols:
        if all(k.lower() in c.lower() for k in keys): return c
def find_tsv():
    f = sorted(glob.glob(os.path.join(RAW, '*.tsv')))
    if not f: raise FileNotFoundError(f'No .tsv in {RAW}')
    return f[-1]
def read_tsv_chunks(path, size=100_000, **kw):
    return pd.read_csv(path, sep='\t', chunksize=size, dtype=str, quoting=3,
                       on_bad_lines='skip', low_memory=False, **kw)
def encode_protein(seq):
    ids = [AA_IDX.get(c, 0) for c in seq[:MAX_PROT]]
    return ids + [0] * (MAX_PROT - len(ids))

def metrics(y, p):
    from scipy.stats import pearsonr, spearmanr
    y, p = np.asarray(y, float), np.asarray(p, float)
    r = lambda f: float(f(y, p)[0]) if len(y) > 2 and np.std(p) > 0 and np.std(y) > 0 else float('nan')
    return dict(n=int(len(y)), rmse=float(np.sqrt(np.mean((y-p)**2))), mae=float(np.mean(np.abs(y-p))),
                pearson=r(pearsonr), spearman=r(spearmanr))

def _onehot(v, choices): return [float(v == c) for c in choices]
def smiles_to_data(smi, seed=42):
    """SMILES -> PyG Data with ETKDG 3D coords (heavy atoms only). None on failure."""
    import torch
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = max(Chem.GetMolFrags(m, asMols=True), key=lambda x: x.GetNumAtoms())
    if m.GetNumAtoms() > MAX_ATOMS or m.GetNumAtoms() < 2: return None
    mh = Chem.AddHs(m)
    if AllChem.EmbedMolecule(mh, randomSeed=seed) != 0 and \
       AllChem.EmbedMolecule(mh, randomSeed=seed, useRandomCoords=True) != 0: return None
    try: AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
    except Exception: pass
    mh = Chem.RemoveHs(mh)
    pos = mh.GetConformer().GetPositions()
    H = Chem.HybridizationType
    x = []
    for a in mh.GetAtoms():
        z = a.GetAtomicNum()
        x.append(_onehot(z if z in ATOMS else -1, ATOMS + [-1]) + _onehot(min(a.GetDegree(), 5), range(6))
                 + [a.GetFormalCharge() / 2, float(a.GetIsAromatic())]
                 + _onehot(a.GetHybridization(), [H.SP, H.SP2, H.SP3, H.UNSPECIFIED]) + [min(a.GetTotalNumHs(), 4) / 4])
    BT = Chem.BondType
    ei, ea = [], []
    for b in mh.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = _onehot(b.GetBondType(), [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]) + [float(b.GetIsConjugated()), float(b.IsInRing())]
        ei += [[i, j], [j, i]]; ea += [f, f]
    if not ei: return None
    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.tensor(ei).t().contiguous(),
                edge_attr=torch.tensor(ea, dtype=torch.float), pos=torch.tensor(pos, dtype=torch.float))
