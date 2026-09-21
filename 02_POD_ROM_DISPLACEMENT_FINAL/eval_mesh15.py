from __future__ import annotations
import argparse,csv,json
from pathlib import Path

from evaluate_rom import aggregate
from pod_external_common import eval_case, load_frozen_rom, map_npz_by_id

MESH_GROUPS=[(4,183,184),(3,185,186),(139,174,175),(15,187,188),(128,178,179)]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base-data-dir',type=Path,required=True); ap.add_argument('--external-data-dir',type=Path,required=True); ap.add_argument('--basis',type=Path,required=True); ap.add_argument('--dataset',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--representation',default='local_residual',choices=['direct','global_residual','local_residual']); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
 base=map_npz_by_id(a.base_data_dir); ext=map_npz_by_id(a.external_data_dir); frozen=load_frozen_rom(a.basis,a.dataset,a.checkpoint,a.representation); a.out_dir.mkdir(parents=True,exist_ok=True)
 cases=[]
 for src,fine,coarse in MESH_GROUPS:
  if src not in base: raise RuntimeError(f'baseline sample{src:03d} missing from Base150')
  if fine not in ext or coarse not in ext: raise RuntimeError(f'mesh variants missing for source {src}')
  cases += [(src,fine,0.80,ext[fine]),(src,src,1.00,base[src]),(src,coarse,1.25,ext[coarse])]
 method_rows={k:[] for k in ('roundtrip','pod_oracle','rom')}; csv_rows=[]
 print('='*110); print('FROZEN POD-ROM MESH-ROBUSTNESS15 EVALUATION'); print('representation:',a.representation,'rank:',frozen['rank'],'device:',frozen['device']); print('5 geometries x mesh size 0.80/1.00/1.25 mm; 1.00 mm baselines reused from Base150'); print('='*110)
 for i,(src,eid,h,p) in enumerate(cases,1):
  c,met=eval_case(p,frozen)
  for method,row in met.items():
   rr={**row,'case':p.name,'eval_sample_id':eid,'source_sample_id':src,'mesh_size_mm':h,'method':method}; method_rows[method].append(rr); csv_rows.append(rr)
  r=met['rom']; print(f"[{i:02d}/15] src={src:03d} eval={eid:04d} h={h:.2f} ROM mean={r['mean_free_mm']:.6f} final={r['final_free_mm']:.6f} max={r['max_free_mm']:.6f}",flush=True)
 summaries={k:aggregate(v) for k,v in method_rows.items()}; paired=[]
 rom_rows=method_rows['rom']
 for src,_,_ in MESH_GROUPS:
  rr={float(r['mesh_size_mm']):r for r in rom_rows if int(r['source_sample_id'])==src}; b=rr[1.0]
  for h in (0.8,1.25):
   x=rr[h]; paired.append({'source_sample_id':src,'mesh_size_mm':h,'delta_mean_free_mm':x['mean_free_mm']-b['mean_free_mm'],'ratio_mean_free':x['mean_free_mm']/max(b['mean_free_mm'],1e-30),'delta_final_free_mm':x['final_free_mm']-b['final_free_mm'],'ratio_final_free':x['final_free_mm']/max(b['final_free_mm'],1e-30),'delta_max_free_mm':x['max_free_mm']-b['max_free_mm']})
 with (a.out_dir/'case_metrics.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
 with (a.out_dir/'paired_vs_1mm.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(paired[0].keys())); w.writeheader(); w.writerows(paired)
 out={'suite':'mesh15','representation':a.representation,'rank':frozen['rank'],'normalization_source':'frozen Train120 only','metrics_contract':'free tube nodes; frames 1..180; case-balanced','roundtrip':summaries['roundtrip'],'pod_oracle':summaries['pod_oracle'],'rom':summaries['rom'],'paired_vs_1mm':paired}
 (a.out_dir/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print('\nROM SUMMARY:',json.dumps(summaries['rom'],indent=2)); print('wrote:',a.out_dir/'summary.json')
if __name__=='__main__': main()
