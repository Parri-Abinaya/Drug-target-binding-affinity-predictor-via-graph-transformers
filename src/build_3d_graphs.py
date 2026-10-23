import os
import argparse
import torch
import pandas as pd
from multiprocessing import Pool, freeze_support
from common import *

def build_one(smi):
    try:
        g = smiles_to_data(smi)
        return smi, g, None
    except Exception as e:
        return smi, None, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mtype', default='Ki')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=500)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()

    files = [os.path.join(PROC, f'{s}.csv') for s in ('train', 'val', 'test')]
    df = pd.concat(
        [pd.read_csv(f, usecols=['smiles', 'mtype']) for f in files],
        ignore_index=True
    )
    smis = sorted(df.loc[df.mtype == a.mtype, 'smiles'].dropna().unique())
    if a.limit:
        smis = smis[:a.limit]

    existing = set()
    next_id = 0
    for f in os.listdir(GRAPHS):
        if f.startswith('graphs_') and f.endswith('.pt'):
            path = os.path.join(GRAPHS, f)
            data = torch.load(path, map_location='cpu', weights_only=False)
            existing.update(data.keys())
            try:
                next_id = max(next_id, int(f[7:12]) + 1)
            except ValueError:
                pass
            del data

    pending = [s for s in smis if s not in existing]
    total = len(pending)
    print(
        f'Type: {a.mtype} | Target: {len(smis):,} | '
        f'Already available: {len(smis)-total:,} | Remaining: {total:,} | '
        f'Workers: {a.workers}',
        flush=True
    )

    if total == 0:
        print('Nothing left to build.', flush=True)
        return

    batch = {}
    failed = []
    processed = 0
    saved = 0

    def save_batch():
        nonlocal batch, next_id, saved
        if batch:
            out = os.path.join(GRAPHS, f'graphs_{next_id:05d}.pt')
            torch.save(batch, out)
            saved += len(batch)
            print(f'Saved {out} ({len(batch)} graphs)', flush=True)
            next_id += 1
            batch = {}

    pool = Pool(processes=a.workers)
    try:
        for smi, g, error in pool.imap_unordered(build_one, pending, chunksize=4):
            processed += 1
            if g is None:
                failed.append(smi)
                if error:
                    print(f'Failed molecule: {error}', flush=True)
            else:
                batch[smi] = g

            if len(batch) >= a.batch_size:
                save_batch()

            if processed % 500 == 0 or processed == total:
                print(
                    f'Processed {processed:,}/{total:,} '
                    f'({100*processed/total:.1f}%) | '
                    f'Saved this run: {saved:,} | Failed: {len(failed):,}',
                    flush=True
                )

        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print('Stopping workers; saving completed graphs...', flush=True)
        pool.terminate()
        pool.join()
    except Exception:
        pool.terminate()
        pool.join()
        raise
    finally:
        save_batch()
        if failed:
            with open(os.path.join(LOGS, 'failed_conformers.txt'), 'a') as f:
                f.write('\n'.join(failed) + '\n')

    print(f'Finished. Saved {saved:,} new graphs.', flush=True)

if __name__ == '__main__':
    freeze_support()
    main()
