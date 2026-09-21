from __future__ import annotations
import argparse, json, csv
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-root',type=Path,default=Path('results')); ap.add_argument('--split',choices=['val','test'],default='test'); ap.add_argument('--output',type=Path,default=None); a=ap.parse_args()
    rows=[]
    for rep in ['direct','global_residual','local_residual']:
        p=a.results_root/rep/a.split/'summary.json'
        if not p.exists(): print('missing:',p); continue
        d=json.loads(p.read_text()); s=d['methods']['rom']; rows.append({'representation':rep,**s})
    if not rows: raise SystemExit('no result summaries found')
    out=a.output or (a.results_root/f'final_{a.split}_comparison.csv'); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('='*100); print(f'FINAL {a.split.upper()} ROM COMPARISON'); print('='*100)
    for r in rows: print(r)
    print('wrote:',out)

if __name__=='__main__': main()
