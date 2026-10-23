"""Baselines (Morgan FP + AA/dipeptide composition) vs proposed model. Usage: python compare_models.py --mtype Ki --tag model_Ki"""
import os, json, argparse, itertools, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from common import *
from graph_dataset import AffinityDataset
RDLogger.DisableLog('rdApp.*')
ap = argparse.ArgumentParser(); ap.add_argument('--mtype', default='Ki'); ap.add_argument('--tag', default='model_Ki'); ap.add_argument('--max-train', type=int, default=20000)
a = ap.parse_args(); DI = {x+y: i for i, (x, y) in enumerate(itertools.product(AA, AA))}
def feats(df):
    fp = {s: np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 1024)) for s in df.smiles.unique()}
    pc = {}
    for s in df.seq.unique():
        v = np.zeros(420); 
        for c in s: v[AA.index(c)] += 1
        for i in range(len(s)-1): v[20 + DI[s[i:i+2]]] += 1
        pc[s] = v / len(s)
    return np.hstack([np.stack(df.smiles.map(fp)), np.stack(df.seq.map(pc))])
tr = AffinityDataset(os.path.join(PROC, 'train.csv'), a.mtype, max_rows=a.max_train).df; te = AffinityDataset(os.path.join(PROC, 'test.csv'), a.mtype).df
Xtr, Xte = feats(tr), feats(te); rows = []
for name, m in [('Ridge (FP+composition)', Ridge(10.0)), ('RandomForest (FP+composition)', RandomForestRegressor(200, n_jobs=-1, random_state=42))]:
    m.fit(Xtr, tr.label); rows.append(dict(model=name, **metrics(te.label, m.predict(Xte))))
f = os.path.join(METRICS, f'{a.tag}_test_metrics.json')
if os.path.exists(f): rows.append(dict(model='3D Graph Transformer + cross-attn', **json.load(open(f))))
r = pd.DataFrame(rows); r.to_csv(os.path.join(METRICS, 'comparison.csv'), index=False); print(r.to_string(index=False))
