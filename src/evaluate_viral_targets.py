"""Held-out viral evaluation from test predictions. Usage: python evaluate_viral_targets.py --tag model_Ki [--min-n 30]"""
import os, argparse, pandas as pd
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--min-n', type=int, default=30); a = ap.parse_args()
p = pd.read_csv(os.path.join(METRICS, f'{a.tag}_test_predictions.csv')); p['viral'] = p.organism.apply(is_viral); v = p[p.viral]
if v.empty: raise SystemExit('No viral rows in the test split.')
rows = [dict(target='ALL VIRAL', **metrics(v.label, v.pred))]
for t, g in v.groupby('target'):
    if len(g) >= a.min_n: rows.append(dict(target=t, **metrics(g.label, g.pred)))
r = pd.DataFrame(rows); r.to_csv(os.path.join(METRICS, f'{a.tag}_viral_metrics.csv'), index=False); print(r.to_string(index=False))
