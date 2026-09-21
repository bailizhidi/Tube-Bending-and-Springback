from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from pod_common import canonical_representation, decode_scalar_string, fixed_to_native, list_split_files, load_case, representation_native_to_displacement


def case_metrics(pred,true,free_mask):
    # Match GNN formal evaluation: predicted frames 1..180, free tube only.
    pred=np.asarray(pred,dtype=np.float32)[1:,free_mask,:]
    true=np.asarray(true,dtype=np.float32)[1:,free_mask,:]
    diff=pred-true; err=np.linalg.norm(diff,axis=-1)
    return {
      'mean_free_mm':float(err.mean()),
      'final_free_mm':float(err[-1].mean()),
      'p95_free_mm':float(np.percentile(err,95)),
      'p99_free_mm':float(np.percentile(err,99)),
      'max_free_mm':float(err.max()),
      'rel_l2_pct':float(100*np.sqrt(np.sum(diff.astype(np.float64)**2)/max(np.sum(true.astype(np.float64)**2),1e-30))),
    }


def aggregate(rows):
    return {
      'mean_free_mm':float(np.mean([r['mean_free_mm'] for r in rows])),
      'final_free_mm':float(np.mean([r['final_free_mm'] for r in rows])),
      'mean_case_p95_mm':float(np.mean([r['p95_free_mm'] for r in rows])),
      'mean_case_p99_mm':float(np.mean([r['p99_free_mm'] for r in rows])),
      'worst_max_free_mm':float(np.max([r['max_free_mm'] for r in rows])),
      'mean_case_rel_l2_pct':float(np.mean([r['rel_l2_pct'] for r in rows])),
      'worst_case':rows[int(np.argmax([r['max_free_mm'] for r in rows]))]['case'],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--basis',type=Path,required=True)
    ap.add_argument('--representation',required=True,choices=['direct','global_residual','local_residual'])
    ap.add_argument('--split',choices=['val','test'],default='val'); ap.add_argument('--ranks',type=int,nargs='+',default=[1,2,4,8,16,32,64,128,256])
    ap.add_argument('--output-csv',type=Path,default=None)
    a=ap.parse_args(); rep=canonical_representation(a.representation)
    b=np.load(a.basis,allow_pickle=False); basis_rep=decode_scalar_string(b['representation'])
    if basis_rep!=rep: raise RuntimeError(f'basis representation={basis_rep}, requested={rep}')
    mean=np.asarray(b['mean'],dtype=np.float32); modes=np.asarray(b['modes'],dtype=np.float32); sv=np.asarray(b['singular_values'],dtype=np.float64); total=float(b['total_energy'])
    n_s,n_phi=[int(v) for v in b['grid']]; ranks=[r for r in a.ranks if 1<=r<=modes.shape[0]]
    files=list_split_files(a.data_dir,a.split); cum=np.cumsum(sv**2)/total
    print('='*120); print('POD ORACLE | GNN-ALIGNED DISPLACEMENT METRICS'); print('='*120)
    print('representation:',rep); print('split:',a.split,'cases:',len(files),'grid:',f'{n_s}x{n_phi}x3','ranks:',ranks)
    all_rows={r:[] for r in ranks}; rt_rows=[]
    for ci,p in enumerate(files,1):
        c=load_case(p,n_s,n_phi,rep); T=c.U_true.shape[0]; X=c.representation_fixed.reshape(T,-1); coeff=(X-mean[None,:])@modes.T
        rt_field=fixed_to_native(c.representation_fixed,c.grid_ids,c.s_src,c.phi_src)
        rt_U=representation_native_to_displacement(c,rt_field,rep); rr=case_metrics(rt_U,c.U_true,c.free_mask); rr['case']=p.name; rt_rows.append(rr)
        for r in ranks:
            rec=(mean[None,:]+coeff[:,:r]@modes[:r]).reshape(T,n_s,n_phi,3).astype(np.float32)
            rec_native=fixed_to_native(rec,c.grid_ids,c.s_src,c.phi_src); pred_U=representation_native_to_displacement(c,rec_native,rep)
            row=case_metrics(pred_U,c.U_true,c.free_mask); row['case']=p.name; row['rank']=r; all_rows[r].append(row)
        print(f'[{ci:02d}/{len(files):02d}] {p.name} done',flush=True)
    print('\nROUNDTRIP FLOOR'); print(aggregate(rt_rows))
    print('\nPOD ORACLE SUMMARY')
    print(f"{'rank':>6} {'train_energy':>13} {'mean_free':>12} {'final_free':>12} {'case_p95':>12} {'worst_max':>12} {'case_RelL2%':>13} {'worst_case'}")
    output=[]
    for r in ranks:
        s=aggregate(all_rows[r]); output.append({'rank':r,'train_energy':float(cum[r-1]),**s})
        print(f"{r:6d} {cum[r-1]:13.9f} {s['mean_free_mm']:12.6f} {s['final_free_mm']:12.6f} {s['mean_case_p95_mm']:12.6f} {s['worst_max_free_mm']:12.6f} {s['mean_case_rel_l2_pct']:13.6f} {s['worst_case']}")
    if a.output_csv:
        a.output_csv.parent.mkdir(parents=True,exist_ok=True)
        with a.output_csv.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(output[0].keys())); w.writeheader(); w.writerows(output)
        print('wrote:',a.output_csv)

if __name__=='__main__': main()
