"""Stage A3: long-format labels. pAff = 9 - log10(nM). Measurement type + qualifier retained; types NOT mixed.
Usage: python prepare_affinity_labels.py [--keep-censored]"""
import os, re, sys, numpy as np, pandas as pd
from common import *
keep_cens = '--keep-censored' in sys.argv
rx = re.compile(r'^\s*([<>]?)\s*=?\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*$')
def parse(v):
    m = rx.match(v) if isinstance(v, str) else None
    return (m.group(1) or '=', float(m.group(2))) if m else (None, np.nan)
parts = []
for ch in pd.read_csv(os.path.join(PROC, 'bindingdb_clean.csv'), chunksize=200_000, dtype=str):
    for col, name in MTYPES.items():
        d = ch.dropna(subset=[col])
        if d.empty: continue
        q = d[col].apply(parse)
        o = d[['smiles', 'seq', 'target', 'organism']].copy()
        o['mtype'] = name; o['qualifier'] = q.str[0]; o['value_nM'] = q.str[1]; parts.append(o)
df = pd.concat(parts, ignore_index=True).dropna(subset=['value_nM'])
n0 = len(df)
df = df[(df.value_nM >= 0.01) & (df.value_nM <= 1e7)]
if not keep_cens: df = df[df.qualifier == '=']
agg = df.groupby(['smiles', 'seq', 'mtype', 'qualifier'], as_index=False).agg(
    target=('target', 'first'), organism=('organism', 'first'), value_nM=('value_nM', 'median'), n_meas=('value_nM', 'size'))
agg['label'] = 9 - np.log10(agg.value_nM)
agg.to_csv(os.path.join(PROC, 'bindingdb_labels.csv'), index=False)
print(f'parsed {n0:,} -> {len(df):,} filtered -> {len(agg):,} unique (ligand, protein, type) labels')
print(agg.mtype.value_counts().to_string())
