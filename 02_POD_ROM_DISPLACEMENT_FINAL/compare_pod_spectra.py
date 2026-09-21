from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from pod_common import decode_scalar_string


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--direct',type=Path,required=True); ap.add_argument('--global-residual',dest='global_residual',type=Path,required=True); ap.add_argument('--local-residual',dest='local_residual',type=Path,required=True)
    ap.add_argument('--output-csv',type=Path,default=Path('results/pod_spectrum_comparison.csv'))
    a=ap.parse_args(); items=[('direct',a.direct),('global_residual',a.global_residual),('local_residual',a.local_residual)]
    rows=[]
    print('='*100); print('POD SPECTRUM COMPARISON'); print('='*100)
    for expected,p in items:
        z=np.load(p,allow_pickle=False); rep=decode_scalar_string(z['representation']); sv=np.asarray(z['singular_values'],dtype=np.float64); total=float(z['total_energy']); cum=np.cumsum(sv**2)/total
        if rep!=expected: raise RuntimeError(f'{p}: representation={rep}, expected={expected}')
        row={'representation':rep,'total_centered_energy':total}
        print('\n',rep)
        for level in (0.90,0.99,0.999,0.9999):
            hit=np.flatnonzero(cum>=level); r=int(hit[0])+1 if hit.size else -1; row[f'rank_{100*level:.2f}pct']=r; print(f' {100*level:7.3f}% -> {r if r>0 else ">q"} modes')
        for r in (1,2,4,8,16,32,64,128,256):
            if r<=len(cum): row[f'energy_rank{r}']=float(cum[r-1]); print(f' rank={r:3d} energy={cum[r-1]:.10f}')
        rows.append(row)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    a.output_csv.parent.mkdir(parents=True,exist_ok=True)
    with a.output_csv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    print('\nwrote:',a.output_csv)

if __name__=='__main__': main()
