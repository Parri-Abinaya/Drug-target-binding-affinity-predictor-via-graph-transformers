"""Usage: python train_bindingdb.py --mtype Ki --epochs 20 --batch 16 [--max-train 5000] [--tag name]"""
import os, json, argparse, time, torch, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from common import *
from graph_dataset import AffinityDataset, load_graphs
from affinity_model import AffinityModel
ap = argparse.ArgumentParser()
ap.add_argument('--mtype', default='Ki'); ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--max-train', type=int); ap.add_argument('--patience', type=int, default=5)
ap.add_argument('--d', type=int, default=128); ap.add_argument('--tag'); ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args(); tag = a.tag or f'model_{a.mtype}'; torch.manual_seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; g = load_graphs()
tr, va = (AffinityDataset(os.path.join(PROC, f'{s}.csv'), a.mtype, g, a.max_train if s == 'train' else None) for s in ('train', 'val'))
print(f'train {len(tr)} | val {len(va)} | device {dev}'); assert len(tr) and len(va), 'empty split: build graphs / check mtype'
mean = float(tr.df.label.mean())
tl, vl = DataLoader(tr, a.batch, shuffle=True), DataLoader(va, a.batch)
model = AffinityModel(d=a.d).to(dev); opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-2)
def run(loader, train):
    model.train(train); tot = n = 0
    for b in loader:
        b = b.to(dev)
        with torch.set_grad_enabled(train):
            loss = F.mse_loss(model(b), b.y.view(-1) - mean)
        if train: opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
        tot += loss.item() * b.num_graphs; n += b.num_graphs
    return tot / n
hist, best, bad = [], 1e9, 0
for ep in range(a.epochs):
    t = time.time(); trl, vll = run(tl, True), run(vl, False); hist.append(dict(epoch=ep, train_mse=trl, val_mse=vll))
    print(f'ep {ep:02d} train {trl:.3f} val {vll:.3f} ({time.time()-t:.0f}s)', flush=True)
    if vll < best:
        best, bad = vll, 0
        torch.save(dict(state=model.state_dict(), cfg=model.cfg, mean=mean, mtype=a.mtype), os.path.join(MODELS, f'{tag}.pt'))
    else:
        bad += 1
        if bad >= a.patience: print('early stop'); break
json.dump(dict(args=vars(a), history=hist), open(os.path.join(METRICS, f'{tag}_history.json'), 'w'), indent=1)
