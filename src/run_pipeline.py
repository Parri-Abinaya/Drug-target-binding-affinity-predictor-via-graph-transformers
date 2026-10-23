"""Run all stages. Smoke test: python run_pipeline.py --max-rows 200000 --subset 20000 --epochs 2
Full: python run_pipeline.py --epochs 20"""
import sys, argparse, subprocess
ap = argparse.ArgumentParser(); ap.add_argument('--max-rows', type=int); ap.add_argument('--subset', type=int); ap.add_argument('--mtype', default='Ki')
ap.add_argument('--epochs', default='20'); ap.add_argument('--max-train', type=int); ap.add_argument('--skip', nargs='*', default=[]); a = ap.parse_args()
tag = f'model_{a.mtype}'; opt = lambda f, v: [f, str(v)] if v else []
steps = [('inspect', ['inspect_bindingdb.py'] + ([str(a.max_rows)] if a.max_rows else [])),
         ('clean', ['clean_bindingdb.py'] + ([str(a.max_rows)] if a.max_rows else [])),
         ('labels', ['prepare_affinity_labels.py']), ('targets', ['analyze_targets.py']),
         ('split', ['split_bindingdb.py'] + opt('--subset', a.subset)),
         ('graphs', ['build_3d_graphs.py', '--mtype', a.mtype]), ('inspect_graphs', ['inspect_3d_graphs.py']),
         ('train', ['train_bindingdb.py', '--mtype', a.mtype, '--epochs', a.epochs, '--tag', tag] + opt('--max-train', a.max_train)),
         ('evaluate', ['evaluate_bindingdb.py', '--tag', tag]), ('compare', ['compare_models.py', '--mtype', a.mtype, '--tag', tag]),
         ('plots', ['plot_results.py', '--tag', tag]), ('viral', ['evaluate_viral_targets.py', '--tag', tag])]
for name, cmd in steps:
    if name in a.skip: continue
    print(f'\n=== {name} ==='); r = subprocess.run([sys.executable] + cmd)
    if r.returncode and name not in ('viral', 'inspect'): sys.exit(f'Stage {name} failed')
