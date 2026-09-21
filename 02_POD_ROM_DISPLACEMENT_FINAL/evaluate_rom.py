from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
from pod_common import canonical_representation, decode_scalar_string, fixed_to_native, list_split_files, load_case, representation_native_to_displacement
from train_rom_mlp import ROMMLP, make_features


def metrics(pred,true,free):
    pred=np.asarray(pred,dtype=np.float32)[1:,free,:]; true=np.asarray(true,dtype=np.float32)[1:,free,:]
    diff=pred-true; err=np.linalg.norm(diff,axis=-1)
    flat=int(np.argmax(err)); fi,ni=np.unravel_index(flat,err.shape)
    return {'mean_free_mm':float(err.mean()),'final_free_mm':float(err[-1].mean()),'p95_free_mm':float(np.percentile(err,95)),'p99_free_mm':float(np.percentile(err,99)),'max_free_mm':float(err.max()),'max_pred_frame':int(fi+1),'max_free_local_index':int(ni),'rel_l2_pct':float(100*np.sqrt(np.sum(diff.astype(np.float64)**2)/max(np.sum(true.astype(np.float64)**2),1e-30)))}


def aggregate(rows):
    im=int(np.argmax([r['max_free_mm'] for r in rows])); return {'mean_free_mm':float(np.mean([r['mean_free_mm'] for r in rows])),'final_free_mm':float(np.mean([r['final_free_mm'] for r in rows])),'mean_case_p95_mm':float(np.mean([r['p95_free_mm'] for r in rows])),'mean_case_p99_mm':float(np.mean([r['p99_free_mm'] for r in rows])),'worst_max_free_mm':float(rows[im]['max_free_mm']),'mean_case_rel_l2_pct':float(np.mean([r['rel_l2_pct'] for r in rows])),'worst_case':rows[im]['case']}


def predict(model,geom,tau,gm,gs,device):
    G=np.repeat(geom[None,:],len(tau),axis=0).astype(np.float32); X,_=make_features(G,tau[:,None],gm,gs); out=[]
    with torch.no_grad():
        for i in range(0,len(X),4096): out.append(model(torch.from_numpy(X[i:i+4096]).to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--basis',type=Path,required=True); ap.add_argument('--dataset',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--representation',required=True,choices=['direct','global_residual','local_residual']); ap.add_argument('--split',choices=['val','test'],default='val'); ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args(); rep=canonical_representation(a.representation)
    b=np.load(a.basis,allow_pickle=False); br=decode_scalar_string(b['representation']); d=np.load(a.dataset,allow_pickle=False); dr=decode_scalar_string(d['representation']); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cr=str(ck.get('representation',''))
    if not (br==dr==cr==rep):
        raise RuntimeError(f'representation mismatch basis={br} dataset={dr} checkpoint={cr} requested={rep}')

    output_space=str(ck.get('coefficient_output_space',''))
    if output_space!='normalized_by_global_train_max':
        raise RuntimeError(
            f'unsupported coefficient_output_space={output_space!r}; '
            'expected normalized_by_global_train_max'
        )
    mean=np.asarray(b['mean'],dtype=np.float32); modes=np.asarray(b['modes'],dtype=np.float32); n_s,n_phi=[int(v) for v in b['grid']]; rank=int(ck['rank']); modes_r=modes[:rank]
    gm=np.asarray(ck['geom_mean'],dtype=np.float32); gs=np.asarray(ck['geom_std'],dtype=np.float32); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ROMMLP(int(ck['n_inputs']),rank,int(ck['width']),int(ck['depth'])); model.load_state_dict(ck['state_dict']); model.to(device).eval()
    files=list_split_files(a.data_dir,a.split); a.out_dir.mkdir(parents=True,exist_ok=True)
    method_rows={'roundtrip':[],'pod_oracle':[],'rom':[]}; case_csv=[]
    print('='*120); print('POD-ROM FINAL DISPLACEMENT EVALUATION'); print('='*120); print('representation:',rep,'split:',a.split,'cases:',len(files),'rank:',rank,'device:',device)
    for ci,p in enumerate(files,1):
        c=load_case(p,n_s,n_phi,rep); T=c.U_true.shape[0]; X=c.representation_fixed.reshape(T,-1); coeff_true=(X-mean[None,:])@modes_r.T
        coeff_pred_norm=predict(model,c.geom,c.tau,gm,gs,device)
        coeff_pred=coeff_pred_norm*np.float32(ck['y_scale'])
        rt_native=fixed_to_native(c.representation_fixed,c.grid_ids,c.s_src,c.phi_src); rt_U=representation_native_to_displacement(c,rt_native,rep)
        oracle_fixed=(mean[None,:]+coeff_true@modes_r).reshape(T,n_s,n_phi,3); oracle_native=fixed_to_native(oracle_fixed.astype(np.float32),c.grid_ids,c.s_src,c.phi_src); oracle_U=representation_native_to_displacement(c,oracle_native,rep)
        rom_fixed=(mean[None,:]+coeff_pred@modes_r).reshape(T,n_s,n_phi,3); rom_native=fixed_to_native(rom_fixed.astype(np.float32),c.grid_ids,c.s_src,c.phi_src); rom_U=representation_native_to_displacement(c,rom_native,rep)
        for name,pred in [('roundtrip',rt_U),('pod_oracle',oracle_U),('rom',rom_U)]:
            row=metrics(pred,c.U_true,c.free_mask); row.update({'case':p.name,'sample_id':c.sample_id,'method':name}); method_rows[name].append(row); case_csv.append(row)
        rr=method_rows['rom'][-1]; print(f"[{ci:02d}/{len(files):02d}] {p.name} ROM mean={rr['mean_free_mm']:.6f} final={rr['final_free_mm']:.6f} max={rr['max_free_mm']:.6f}",flush=True)
    summaries={name:aggregate(rows) for name,rows in method_rows.items()}
    print('\nSUMMARY | case-balanced, free tube, frames 1..180')
    for name,s in summaries.items(): print(name,s)
    with (a.out_dir/'case_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(case_csv[0].keys())); w.writeheader(); w.writerows(case_csv)
    (a.out_dir/'summary.json').write_text(json.dumps({'representation':rep,'split':a.split,'rank':rank,'metrics_contract':'free tube nodes; frames 1..180; case-balanced','methods':summaries},indent=2))
    print('wrote:',a.out_dir/'summary.json')

if __name__=='__main__': main()
