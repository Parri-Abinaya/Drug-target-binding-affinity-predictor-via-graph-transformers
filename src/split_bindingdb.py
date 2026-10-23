"""Stage A5: scaffold-grouped split (no ligand scaffold shared across splits) + leakage checks.
Usage: python split_bindingdb.py [--subset N] [--seed 42]"""
import os, sys, json, numpy as np, pandas as pd
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from common import *
a = sys.argv; seed = int(a[a.index('--seed')+1]) if '--seed' in a else 42
df = pd.read_csv(os.path.join(PROC, 'bindingdb_labels.csv'))
if '--subset' in a: df = df.sample(min(int(a[a.index('--subset')+1]), len(df)), random_state=seed).reset_index(drop=True)
def scaf(s):
    try: return MurckoScaffoldSmiles(smiles=s) or s
    except Exception: return s
u = pd.Series(df.smiles.unique()); sm = dict(zip(u, u.apply(scaf))); df['scaffold'] = df.smiles.map(sm)
groups = df.groupby('scaffold').size().sample(frac=1, random_state=seed)
groups = pd.concat([groups[groups > 0.01*len(df)], groups[groups <= 0.01*len(df)]])  # big scaffolds go to train first
assign, c = {}, 0
for sc, n in groups.items():
    assign[sc] = 'train' if c < 0.8*len(df) else ('val' if c < 0.9*len(df) else 'test'); c += n
df['split'] = df.scaffold.map(assign)
for s in ('train', 'val', 'test'): df[df.split == s].drop(columns='split').to_csv(os.path.join(PROC, f'{s}.csv'), index=False)
S = {s: set(df[df.split == s].scaffold) for s in ('train', 'val', 'test')}
assert not (S['train'] & S['test']) and not (S['train'] & S['val']) and not (S['val'] & S['test']), 'scaffold leakage!'
pairs = {s: set(zip(df[df.split == s].smiles, df[df.split == s].seq)) for s in S}
assert not (pairs['train'] & pairs['test']), 'pair leakage!'
info = dict(seed=seed, sizes=df.split.value_counts().to_dict(), test_proteins_unseen_in_train=int(len(set(df[df.split=='test'].seq) - set(df[df.split=='train'].seq))))
json.dump(info, open(os.path.join(LOGS, 'split_info.json'), 'w'), indent=1); print(info, '\nLeakage checks passed.')
