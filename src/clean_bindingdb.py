"""Stage A2: keep valid SMILES + sequences + at least one affinity value. Usage: python clean_bindingdb.py [max_rows]"""
import os, re, sys, pandas as pd
from rdkit import Chem, RDLogger
from common import *
RDLogger.DisableLog('rdApp.*')
maxr = int(sys.argv[1]) if len(sys.argv) > 1 else None
out = os.path.join(PROC, 'bindingdb_clean.csv'); 
if os.path.exists(out): os.remove(out)
seq_ok = re.compile(f'^[{AA}]{{20,}}$'); cache = {}
def ok(s):
    if s not in cache: cache[s] = Chem.MolFromSmiles(s) is not None
    return cache[s]
read = kept = 0; first = True; stats = dict(header_rows=0, no_smiles_seq=0, bad_seq=0, bad_smiles=0, no_affinity=0)
for ch in read_tsv_chunks(find_tsv()):
    if first:
        c = ch.columns
        m = dict(smiles=find_col(c, 'ligand smiles'), seq=find_col(c, 'target chain sequence'), target=find_col(c, 'target name'),
                 organism=find_col(c, 'target source organism'), ki=find_col(c, 'ki (nm)'), ic50=find_col(c, 'ic50 (nm)'),
                 kd=find_col(c, 'kd (nm)'), ec50=find_col(c, 'ec50 (nm)'))
        print('Column map:', m)
    d = ch[[v for v in m.values()]].copy(); d.columns = list(m.keys()); read += len(d)
    n = len(d); d = d[d.smiles != 'Ligand SMILES']; stats['header_rows'] += n - len(d)
    n = len(d); d = d.dropna(subset=['smiles', 'seq']); stats['no_smiles_seq'] += n - len(d)
    d['seq'] = d.seq.str.upper().str.replace(r'\s', '', regex=True)
    n = len(d); d = d[d.seq.apply(lambda s: bool(seq_ok.match(s)))]; stats['bad_seq'] += n - len(d)
    n = len(d); d = d.dropna(subset=['ki', 'ic50', 'kd', 'ec50'], how='all'); stats['no_affinity'] += n - len(d)
    n = len(d); d = d[d.smiles.apply(ok)]; stats['bad_smiles'] += n - len(d)
    d.to_csv(out, mode='a', header=first, index=False); first = False; kept += len(d)
    print(f'read {read:,} kept {kept:,}', end='\r')
    if maxr and read >= maxr: break
print(f'\nDone. read={read:,} kept={kept:,}', stats)
pd.Series({'read': read, 'kept': kept, **stats}).to_csv(os.path.join(LOGS, 'clean_report.csv'))
