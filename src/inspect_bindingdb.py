"""Stage A1: inspect the raw TSV in chunks."""
import os, sys, pandas as pd
from common import *
path = find_tsv(); print('File:', path, f'{os.path.getsize(path)/1e6:.0f} MB')
rows, miss, cols = 0, None, None
for ch in read_tsv_chunks(path, 200_000):
    if cols is None: cols = list(ch.columns); miss = pd.Series(0, index=cols)
    rows += len(ch); miss += ch.isna().sum()
    if len(sys.argv) > 1 and rows >= int(sys.argv[1]): break   # optional row cap for quick looks
print('Rows read:', rows, '| Columns:', len(cols))
rep = pd.DataFrame({'column': cols, 'missing': miss.values, 'missing_pct': (miss.values/rows*100).round(1)})
rep.to_csv(os.path.join(LOGS, 'inspect_report.csv'), index=False)
print(rep.to_string(index=False))
for k in [('ligand smiles',), ('target chain sequence',), ('ki (nm)',), ('ic50 (nm)',), ('kd (nm)',), ('ec50 (nm)',)]:
    print(k[0], '->', find_col(cols, *k))
