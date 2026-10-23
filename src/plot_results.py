"""Usage: python plot_results.py --tag model_Ki"""
import os, json, argparse, pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
h = pd.DataFrame(json.load(open(os.path.join(METRICS, f'{a.tag}_history.json')))['history'])
plt.figure(); plt.plot(h.epoch, h.train_mse, label='train'); plt.plot(h.epoch, h.val_mse, label='val'); plt.xlabel('epoch'); plt.ylabel('MSE'); plt.legend()
plt.savefig(os.path.join(PLOTS, f'{a.tag}_loss.png'), dpi=150); plt.close()
p = pd.read_csv(os.path.join(METRICS, f'{a.tag}_test_predictions.csv')); m = metrics(p.label, p.pred)
plt.figure(figsize=(5, 5)); plt.scatter(p.label, p.pred, s=4, alpha=.4); lo, hi = p.label.min(), p.label.max(); plt.plot([lo, hi], [lo, hi], 'r--')
plt.xlabel('actual pAffinity'); plt.ylabel('predicted'); plt.title(f"RMSE {m['rmse']:.2f} | r {m['pearson']:.2f}")
plt.savefig(os.path.join(PLOTS, f'{a.tag}_pred_vs_actual.png'), dpi=150); plt.close()
plt.figure(); plt.hist(p.pred - p.label, bins=60); plt.xlabel('error (pred - actual)'); plt.savefig(os.path.join(PLOTS, f'{a.tag}_errors.png'), dpi=150)
print('plots saved to', PLOTS)
