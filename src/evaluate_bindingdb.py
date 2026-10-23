"""Usage: python evaluate_bindingdb.py --tag model_Ki [--split test]"""
import os, json, argparse, torch, numpy as np
from torch_geometric.loader import DataLoader
from common import *
from graph_dataset import AffinityDataset
from affinity_model import AffinityModel
def load_model(tag, dev):
    ck = torch.load(os.path.join(MODELS, f'{tag}.pt'), map_location=dev, weights_only=False)
    m = AffinityModel(**ck['cfg']).to(dev); m.load_state_dict(ck['state']); return m.eval(), ck
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--split', default='test')
    ap.add_argument('--batch', type=int, default=32); a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; m, ck = load_model(a.tag, dev)
    ds = AffinityDataset(os.path.join(PROC, f'{a.split}.csv'), ck['mtype']); pred = []
    with torch.no_grad():
        for b in DataLoader(ds, a.batch): pred.append((m(b.to(dev)) + ck['mean']).cpu().numpy())
    ds.df['pred'] = np.concatenate(pred); res = metrics(ds.df.label, ds.df.pred); print(res)
    ds.df.to_csv(os.path.join(METRICS, f'{a.tag}_{a.split}_predictions.csv'), index=False)
    json.dump(res, open(os.path.join(METRICS, f'{a.tag}_{a.split}_metrics.json'), 'w'), indent=1)
