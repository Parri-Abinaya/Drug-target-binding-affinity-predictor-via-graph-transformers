"""Stage B3: sanity-check cached graphs."""
import os, torch, numpy as np
from common import *
from graph_dataset import load_graphs
g = load_graphs(); print('graphs:', len(g))
failed = os.path.join(LOGS, 'failed_conformers.txt')
if os.path.exists(failed): print('failed conformers logged:', sum(1 for l in open(failed) if l.strip()))
n = np.array([d.num_nodes for d in g.values()]); e = np.array([d.edge_index.size(1) for d in g.values()])
print(f'atoms mean/max {n.mean():.1f}/{n.max()} | directed edges mean {e.mean():.1f}')
bad = 0
for s, d in g.items():
    ok = d.x.size(1) == ATOM_DIM and d.edge_attr.size(1) == BOND_DIM and d.pos.shape == (d.num_nodes, 3) \
         and torch.isfinite(d.pos).all() and d.edge_index.max() < d.num_nodes
    bad += not ok
print('malformed graphs:', bad)
b = d.edge_index; dist = (d.pos[b[0]] - d.pos[b[1]]).norm(dim=1)
print(f'sample bond lengths (A): {dist.min():.2f}-{dist.max():.2f} (expect ~1.0-1.6)')
