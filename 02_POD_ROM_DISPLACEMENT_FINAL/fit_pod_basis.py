from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch
from pod_common import canonical_representation, dataset_signature, list_split_files, load_case


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--representation', required=True, choices=['direct','global_residual','local_residual'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--n-s', type=int, default=64)
    ap.add_argument('--n-phi', type=int, default=28)
    ap.add_argument('--n-modes', type=int, default=256)
    ap.add_argument('--niter', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    a=ap.parse_args(); rep=canonical_representation(a.representation)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    train_files=list_split_files(a.data_dir,'train')
    if len(train_files)!=120: raise RuntimeError(f'expected 120 train files, got {len(train_files)}')
    first=load_case(train_files[0],a.n_s,a.n_phi,rep)
    T=first.U_true.shape[0]; dof=a.n_s*a.n_phi*3; n_snap=len(train_files)*T
    X=np.empty((n_snap,dof),dtype=np.float32)
    print('='*110); print('POD BASIS FIT'); print('='*110)
    print('representation :',rep); print('train cases    :',len(train_files)); print('frames/case    :',T)
    print('snapshots      :',n_snap); print('fixed grid     :',f'{a.n_s}x{a.n_phi}x3'); print('snapshot dof   :',dof)
    print('matrix GiB     :',f'{X.nbytes/1024**3:.3f}'); print('dataset sig    :',dataset_signature(train_files))
    t0=time.time(); cur=0
    for i,p in enumerate(train_files,1):
        c=first if i==1 else load_case(p,a.n_s,a.n_phi,rep)
        if c.U_true.shape[0]!=T: raise RuntimeError(f'frame mismatch: {p}')
        X[cur:cur+T]=c.representation_fixed.reshape(T,-1); cur+=T
        if i==1 or i%10==0 or i==len(train_files): print(f'loaded {i:3d}/{len(train_files)} train cases',flush=True)
    mean=X.mean(axis=0,dtype=np.float64).astype(np.float32)
    centered=X-mean[None,:]
    total_energy=float(np.einsum('ij,ij->',centered,centered,dtype=np.float64))
    if total_energy<=0 or not np.isfinite(total_energy): raise RuntimeError(f'invalid total energy {total_energy}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device         :',device)
    if device.type=='cuda': print('gpu            :',torch.cuda.get_device_name(0))
    q=min(a.n_modes,min(centered.shape)-1)
    A=torch.from_numpy(centered).to(device)
    s0=time.time(); Uq,S,V=torch.pca_lowrank(A,q=q,center=False,niter=a.niter); del Uq,A
    if device.type=='cuda': torch.cuda.synchronize()
    sv=S.detach().cpu().numpy().astype(np.float32); modes=V.detach().cpu().numpy().T.astype(np.float32)
    cum=np.cumsum(sv.astype(np.float64)**2)/total_energy
    print('SVD time min   :',f'{(time.time()-s0)/60:.2f}')
    print('CUMULATIVE TRAIN ENERGY')
    energy_ranks={}
    for level in (0.90,0.99,0.999,0.9999):
        hit=np.flatnonzero(cum>=level); r=int(hit[0])+1 if hit.size else None; energy_ranks[str(level)]=r
        print(f' {100*level:7.3f}% -> {r if r is not None else f">{q}"} modes')
    for r in (1,2,4,8,16,32,64,128,256):
        if r<=q: print(f' rank={r:3d} energy={cum[r-1]:.10f}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,mean=mean,modes=modes,singular_values=sv,total_energy=np.float64(total_energy),
        grid=np.asarray([a.n_s,a.n_phi],dtype=np.int32),n_frames=np.int32(T),n_train_cases=np.int32(len(train_files)),
        representation=np.asarray(rep),dataset_signature=np.asarray(dataset_signature(train_files)),seed=np.int32(a.seed),niter=np.int32(a.niter))
    print('wrote          :',a.output); print('total time min :',f'{(time.time()-t0)/60:.2f}')

if __name__=='__main__': main()
