"""Usage: python predict_affinity.py --tag model_Ki --smiles "CC(=O)Oc1ccccc1C(=O)O" --seq MKT..."""
import argparse, torch
from torch_geometric.data import Data, Batch
from common import *
from evaluate_bindingdb import load_model
def predict(tag, smiles, seq, dev='cpu'):
    m, ck = load_model(tag, dev); g = smiles_to_data(smiles)
    if g is None: raise ValueError('Could not build a 3D graph (invalid SMILES or > %d heavy atoms)' % MAX_ATOMS)
    seq = ''.join(c for c in seq.upper() if c in AA)
    if len(seq) < 20: raise ValueError('Protein sequence too short / invalid')
    g.prot = torch.tensor([encode_protein(seq)])
    with torch.no_grad(): p = float(m(Batch.from_data_list([g]).to(dev)) + ck['mean'])
    return dict(mtype=ck['mtype'], pAffinity=p, nM=10 ** (9 - p))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--smiles', required=True); ap.add_argument('--seq', required=True)
    a = ap.parse_args(); print(predict(a.tag, a.smiles, a.seq))
