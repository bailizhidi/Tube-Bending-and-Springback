from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from pod_common import canonical_representation, decode_scalar_string, list_split_files, load_case


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--basis',type=Path,required=True)
    ap.add_argument('--representation',required=True,choices=['direct','global_residual','local_residual']); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--max-rank',type=int,default=32)
    a=ap.parse_args(); rep=canonical_representation(a.representation)
    b=np.load(a.basis,allow_pickle=False); basis_rep=decode_scalar_string(b['representation'])
    if basis_rep!=rep: raise RuntimeError(f'basis representation={basis_rep}, requested={rep}')
    mean=np.asarray(b['mean'],dtype=np.float32); modes=np.asarray(b['modes'],dtype=np.float32); n_s,n_phi=[int(v) for v in b['grid']]
    max_rank=min(a.max_rank,modes.shape[0]); modes_r=modes[:max_rank]
    coeff_rows=[]; geom_rows=[]; tau_rows=[]; angle_rows=[]; frame_rows=[]; case_idx_rows=[]; split_rows=[]
    case_names=[]; case_splits=[]; case_geoms=[]; counter=0
    print('='*110); print('EXTRACT POD COEFFICIENT DATASET'); print('='*110); print('representation:',rep,'max_rank:',max_rank)
    for split in ['train','val']:
        files=list_split_files(a.data_dir,split)
        for i,p in enumerate(files,1):
            c=load_case(p,n_s,n_phi,rep); T=c.U_true.shape[0]; X=c.representation_fixed.reshape(T,-1); coeff=(X-mean[None,:])@modes_r.T
            coeff_rows.append(coeff.astype(np.float32)); geom_rows.append(np.repeat(c.geom[None,:],T,axis=0)); tau_rows.append(c.tau[:,None]); angle_rows.append(c.angle_deg[:,None])
            frame_rows.append(np.arange(T,dtype=np.int32)); case_idx_rows.append(np.full(T,counter,dtype=np.int32)); split_rows.append(np.full(T,split,dtype='U5'))
            case_names.append(p.name); case_splits.append(split); case_geoms.append(c.geom); counter+=1
            print(f'[{split:5s} {i:03d}/{len(files):03d}] {p.name}',flush=True)
    coeffs=np.concatenate(coeff_rows); geom=np.concatenate(geom_rows); tau=np.concatenate(tau_rows); angle=np.concatenate(angle_rows); frame=np.concatenate(frame_rows); case_index=np.concatenate(case_idx_rows); split=np.concatenate(split_rows)
    train_case_geoms=np.asarray([g for g,s in zip(case_geoms,case_splits) if s=='train'],dtype=np.float64); geom_mean=train_case_geoms.mean(0).astype(np.float32); geom_std=train_case_geoms.std(0).astype(np.float32)
    if np.any(geom_std<=0): raise RuntimeError(f'zero geometry std: {geom_std}')
    tr=split=='train'; y_scale=float(np.max(np.abs(coeffs[tr]))); 
    if not np.isfinite(y_scale) or y_scale<=0: raise RuntimeError(f'invalid y_scale={y_scale}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,coeffs=coeffs,geom=geom,tau=tau,angle_deg=angle,frame=frame,case_index=case_index,split=split,
        case_names=np.asarray(case_names,dtype='U128'),case_splits=np.asarray(case_splits,dtype='U5'),case_geoms=np.asarray(case_geoms,dtype=np.float32),
        geom_mean=geom_mean,geom_std=geom_std,y_scale=np.float32(y_scale),max_rank=np.int32(max_rank),basis=np.asarray(str(a.basis)),representation=np.asarray(rep),grid=np.asarray([n_s,n_phi],dtype=np.int32))
    print('snapshots:',coeffs.shape[0],'coeff shape:',coeffs.shape,'y_scale:',y_scale,'wrote:',a.output)

if __name__=='__main__': main()
