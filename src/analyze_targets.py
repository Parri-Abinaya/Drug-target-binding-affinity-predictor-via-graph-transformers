"""Stage A4: per-target coverage and viral subsets."""
import os, pandas as pd
from common import *
df = pd.read_csv(os.path.join(PROC, 'bindingdb_labels.csv'))
df['viral'] = df.organism.apply(is_viral)
t = df.groupby(['seq', 'mtype'], as_index=False).agg(target=('target', 'first'), organism=('organism', 'first'),
        viral=('viral', 'first'), n_measurements=('label', 'size'), n_ligands=('smiles', 'nunique')).sort_values('n_measurements', ascending=False)
t.to_csv(os.path.join(PROC, 'target_summary.csv'), index=False)
print('unique ligands:', df.smiles.nunique(), '| unique proteins:', df.seq.nunique(), '| organisms:', df.organism.nunique())
print('viral rows:', int(df.viral.sum()), '| viral proteins:', df[df.viral].seq.nunique())
print(t[t.viral].head(15)[['target', 'organism', 'mtype', 'n_measurements']].to_string(index=False))
