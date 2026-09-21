from __future__ import annotations
import argparse,csv,json
from pathlib import Path

from evaluate_rom import aggregate
from pod_external_common import eval_case, load_frozen_rom, map_npz_by_id

OOD15_IDS=[156,157,158,159,162,163,166,167,168,169,171,172,180,181,182]
OOD_LABELS={156:"D_low",157:"D_low",158:"D_high",159:"D_high",162:"tD_high",163:"tD_high",166:"RD_high",167:"RD_high",168:"D_low+tD_high",169:"D_low+RD_high",171:"D_high+RD_high",172:"tD_high+RD_high",180:"D_high_mild",181:"tD_high_mild",182:"RD_high_mild"}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--external-data-dir',type=Path,required=True); ap.add_argument('--basis',type=Path,required=True); ap.add_argument('--dataset',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--representation',required=True,choices=['direct','global_residual','local_residual']); ap.add_argument('--out-dir',type=Path,required=True); a=ap.parse_args()
 ext=map_npz_by_id(a.external_data_dir); missing=[i for i in OOD15_IDS if i not in ext]
 if missing: raise RuntimeError(f'missing OOD15 ids: {missing}')
 frozen=load_frozen_rom(a.basis,a.dataset,a.checkpoint,a.representation); a.out_dir.mkdir(parents=True,exist_ok=True)
 method_rows={k:[] for k in ('roundtrip','pod_oracle','rom')}; csv_rows=[]
 print('='*110); print('FROZEN POD-ROM GEOMETRY-OOD15 EVALUATION'); print('representation:',a.representation,'rank:',frozen['rank'],'device:',frozen['device']); print('IMPORTANT: basis, coefficient model, geom scaling and y_scale are all frozen from Train120'); print('='*110)
 for i,sid in enumerate(OOD15_IDS,1):
  c,met=eval_case(ext[sid],frozen)
  for method,row in met.items():
   rr={**row,'case':ext[sid].name,'sample_id':sid,'method':method,'ood_label':OOD_LABELS[sid]}; method_rows[method].append(rr); csv_rows.append(rr)
  r=met['rom']; print(f"[{i:02d}/15] sample={sid:04d} {OOD_LABELS[sid]:18s} ROM mean={r['mean_free_mm']:.6f} final={r['final_free_mm']:.6f} max={r['max_free_mm']:.6f}",flush=True)
 summaries={k:aggregate(v) for k,v in method_rows.items()}
 with (a.out_dir/'case_metrics.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
 out={'suite':'ood15','representation':a.representation,'rank':frozen['rank'],'normalization_source':'frozen Train120 only','metrics_contract':'free tube nodes; frames 1..180; case-balanced','roundtrip':summaries['roundtrip'],'pod_oracle':summaries['pod_oracle'],'rom':summaries['rom']}
 (a.out_dir/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print('\nROM SUMMARY:',json.dumps(summaries['rom'],indent=2)); print('wrote:',a.out_dir/'summary.json')
if __name__=='__main__': main()
