from __future__ import annotations
import argparse,csv,json
from pathlib import Path

GNN_ORDER=[
("exp01_mgn_h80_x10_direct_2step","MGN-Direct"),
("exp02_mgnt_h128_x10_direct_2step","MGN-T-Direct"),
("exp03_mgnt_h128_x10_global_anaresid_2step","Global-ANARESID"),
("exp04_mgnt_h128_x10_local_anaresid_2step","Local-ANARESID"),
]
POD_ORDER=[("direct","POD-Direct"),("global_residual","POD-Global"),("local_residual","POD-Local")]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--suite',choices=['ood15','mesh15'],required=True); ap.add_argument('--gnn-root',type=Path,default=Path('external_results')); ap.add_argument('--pod-project',type=Path,default=None); ap.add_argument('--out-dir',type=Path,default=Path('external_results/summary')); a=ap.parse_args()
 rows=[]
 for eid,label in GNN_ORDER:
  p=a.gnn_root/a.suite/eid/'summary.json'
  if not p.is_file(): continue
  d=json.loads(p.read_text()); s=d['aggregate']; rows.append({'family':'GNN','model':label,'id':eid,**s})
 if a.pod_project:
  reps=POD_ORDER if a.suite=='ood15' else [('local_residual','POD-Local')]
  for rep,label in reps:
   p=a.pod_project/'results_external'/a.suite/rep/'summary.json'
   if not p.is_file(): continue
   d=json.loads(p.read_text()); s=d.get('rom',d.get('aggregate',{})); rows.append({'family':'POD','model':label,'id':rep,**s})
 a.out_dir.mkdir(parents=True,exist_ok=True); out=a.out_dir/f'{a.suite}_model_summary.csv'
 if rows:
  keys=[]
  for r in rows:
   for k in r:
    if k not in keys: keys.append(k)
  with out.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
 print(f'\n{a.suite.upper()} SUMMARY')
 for r in rows:
  mean=r.get('mean_free_mm',float('nan')); final=r.get('final_free_mean_mm',r.get('final_free_mm',float('nan'))); mx=r.get('max_free_mm',r.get('worst_max_free_mm',float('nan')))
  print(f"{r['model']:20s} mean={mean:.6f} final={final:.6f} max={mx:.6f}")
 print('wrote:',out)
if __name__=='__main__': main()
