"""Recompute TrainN-only edge and target-increment statistics after data freeze.

IMPORTANT: --resume 0 is the formal-paper setting. Per-sample caches include the
packed-source size/mtime fingerprint so repaired sample007/138 cannot silently
reuse obsolete sufficient statistics.
"""
from __future__ import annotations
import argparse,json,os,hashlib
from pathlib import Path
import numpy as np, torch
from packed_dataset import safe_torch_load,sample_id_from_cache_path
from target_space import exact_state
SUBSETS=(40,60,80,120)

def suff(a):
    a=a.double(); axes=tuple(range(a.ndim-1))
    return {'count':int(np.prod(a.shape[:-1])),'sum':a.sum(dim=axes).cpu().numpy(),'sumsq':a.square().sum(dim=axes).cpu().numpy(),'min':a.amin(dim=axes).cpu().numpy(),'max':a.amax(dim=axes).cpu().numpy()}
def agg(records,ids,key,clamp=False):
    by={r['sample_id']:r for r in records}; xs=[by[i][key] for i in ids]; c=sum(x['count'] for x in xs); s=sum(np.asarray(x['sum']) for x in xs); ss=sum(np.asarray(x['sumsq']) for x in xs); m=s/c; std=np.sqrt(np.maximum(ss/c-m*m,0.0));
    if clamp: std[std<1e-8]=1.0
    return c,m,std,np.min(np.stack([x['min'] for x in xs]),0),np.max(np.stack([x['max'] for x in xs]),0)
def fingerprint(path:Path):
    st=path.stat(); return int(st.st_size),int(st.st_mtime_ns)
def data_fingerprint(records,ids):
    by={int(r['sample_id']):r for r in records}
    payload='\n'.join(f"{sid}:{by[sid]['source_size']}:{by[sid]['source_mtime_ns']}" for sid in ids)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workspace',type=Path,default=Path('dataeff_workspace')); ap.add_argument('--device',default='cuda',choices=['cuda','cpu']); ap.add_argument('--theta-final-deg',type=float,default=180.0); ap.add_argument('--resume',type=int,choices=[0,1],default=0); args=ap.parse_args()
    ws=args.workspace.resolve(); order=json.loads((ws/'subsets/train_order_maximin.json').read_text())['ordered_train_sample_ids']; order=[int(x) for x in order]
    view=ws/'cache_views/N120'; names=[x.strip() for x in (view/'train_manifest.txt').read_text().splitlines() if x.strip()]; paths={sample_id_from_cache_path(x):view/x for x in names}; device=torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    per=ws/'stats/per_sample_final'; per.mkdir(parents=True,exist_ok=True); allrec=[]; shared=None
    for i,sid in enumerate(order,1):
        path=paths[sid]; size,mtime=fingerprint(path.resolve()); fp=per/f'sample_{sid:04d}.npz'; rec=None
        if args.resume and fp.exists():
            z=np.load(fp,allow_pickle=False)
            if int(z['source_size'])==size and int(z['source_mtime_ns'])==mtime:
                rec={'sample_id':sid,'sig':str(z['sig'].reshape(())),'source_size':size,'source_mtime_ns':mtime,'edge':{k:z[f'e_{k}'] for k in ('sum','sumsq','min','max')},'direct':{k:z[f'd_{k}'] for k in ('sum','sumsq','min','max')},'global_residual':{k:z[f'g_{k}'] for k in ('sum','sumsq','min','max')},'local_residual':{k:z[f'l_{k}'] for k in ('sum','sumsq','min','max')}}
                for key,prefix in [('edge','e'),('direct','d'),('global_residual','g'),('local_residual','l')]: rec[key]['count']=int(z[f'{prefix}_count'])
        if rec is None:
            p=safe_torch_load(str(path)); sig=str(p['cache_signature']); edge=suff(p['mesh_edge_attr_static_raw'].float()); p['mesh_pos']=p['mesh_pos'].to(device); p['is_tube_node']=p['is_tube_node'].to(device); p['world_pos']=p['world_pos'].to(device); vals={}
            for mode in ('direct','global_residual','local_residual'):
                states=[]
                for t in range(int(p['num_time_steps'])): states.append(exact_state(p,t,device,mode,args.theta_final_deg))
                stacked=torch.stack(states,0); d=stacked[1:]-stacked[:-1]; tube=p['is_tube_node'].reshape(-1).bool().to(device); vals[mode]=suff(d[:,tube,:]); del states,stacked,d,tube
            rec={'sample_id':sid,'sig':sig,'source_size':size,'source_mtime_ns':mtime,'edge':edge,**vals}
            save={'source_size':size,'source_mtime_ns':mtime,'sig':np.asarray(sig)}
            for key,prefix in [('edge','e'),('direct','d'),('global_residual','g'),('local_residual','l')]:
                x=rec[key]; save[f'{prefix}_count']=x['count']; [save.__setitem__(f'{prefix}_{k}',x[k]) for k in ('sum','sumsq','min','max')]
            np.savez(fp,**save)
        shared=shared or rec['sig']
        if rec['sig']!=shared: raise RuntimeError('mixed cache signatures')
        allrec.append(rec); print(f'[{i:03d}/120] sample={sid:04d}')
        if device.type=='cuda': torch.cuda.empty_cache()
    for n in SUBSETS:
        tag=f'N{n:03d}'; ids=order[:n]; fp_digest=data_fingerprint(allrec,ids); sd=ws/'stats'/tag; sd.mkdir(parents=True,exist_ok=True)
        c,m,s,mn,mx=agg(allrec,ids,'edge',True)
        norm={'format':'final_bending_edge_stats_v1','feature_contract':'ANARESID_X10_shared_raw_packed_v2','shared_cache_signature':shared,'train_sample_ids':ids,'edge_count':c,'edge_mean':m.tolist(),'edge_std':s.tolist(),'edge_min':mn.tolist(),'edge_max':mx.tolist(),'normalization_source':f'Train{n} only','data_fingerprint':fp_digest}
        (sd/'normalization_stats.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
        td=ws/'target_stats'; td.mkdir(exist_ok=True)
        for mode in ('direct','global_residual','local_residual'):
            c,m,s,mn,mx=agg(allrec,ids,mode,False)
            if np.any(s<=0) or not np.all(np.isfinite(s)): raise RuntimeError(f'{tag} {mode} invalid std={s}')
            raw={'format':'final_bending_target_increment_stats_v1','prediction_mode':mode,'train_sample_ids':ids,'count':c,'theta_final_deg':args.theta_final_deg,'mean':m.tolist(),'std':s.tolist(),'min':mn.tolist(),'max':mx.tolist(),'normalization_source':f'Train{n} only','data_fingerprint':fp_digest}
            (td/f'{mode}_{tag}.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
    print('FINAL TrainN-only statistics complete; use these files for all formal experiments.')
if __name__=='__main__': main()
