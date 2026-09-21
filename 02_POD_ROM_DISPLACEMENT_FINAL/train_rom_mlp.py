from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pod_common import canonical_representation, decode_scalar_string


class ROMMLP(nn.Module):
    def __init__(self,n_inputs:int,n_modes:int,width:int=128,depth:int=3):
        super().__init__(); layers=[]; d=n_inputs
        for _ in range(depth): layers += [nn.Linear(d,width),nn.GELU()]; d=width
        layers.append(nn.Linear(d,n_modes)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


def make_features(geom,tau,geom_mean,geom_std):
    geom=np.asarray(geom,dtype=np.float32); tau=np.asarray(tau,dtype=np.float32).reshape(-1,1); geom_mean=np.asarray(geom_mean,dtype=np.float32); geom_std=np.asarray(geom_std,dtype=np.float32)
    gz=(geom-geom_mean[None,:])/geom_std[None,:]; t=tau
    X=np.concatenate([gz,t,t**2,t**3,np.sin(np.pi*t),np.cos(np.pi*t),np.sin(2*np.pi*t),np.cos(2*np.pi*t)],axis=1)
    names=['D_outer_z','t_over_D_z','R_over_D_z','tau','tau2','tau3','sin_pi_tau','cos_pi_tau','sin_2pi_tau','cos_2pi_tau']
    return X.astype(np.float32),names


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--representation',required=True,choices=['direct','global_residual','local_residual']); ap.add_argument('--rank',type=int,default=16); ap.add_argument('--width',type=int,default=128); ap.add_argument('--depth',type=int,default=3)
    ap.add_argument('--epochs',type=int,default=800); ap.add_argument('--batch-size',type=int,default=512); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight-decay',type=float,default=0.0); ap.add_argument('--patience',type=int,default=120); ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args(); rep=canonical_representation(a.representation)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    d=np.load(a.dataset,allow_pickle=False); ds_rep=decode_scalar_string(d['representation'])
    if ds_rep!=rep: raise RuntimeError(f'dataset representation={ds_rep}, requested={rep}')
    coeff=np.asarray(d['coeffs'],dtype=np.float32); geom=np.asarray(d['geom'],dtype=np.float32); tau=np.asarray(d['tau'],dtype=np.float32); split=np.asarray(d['split']).astype(str); gm=np.asarray(d['geom_mean'],dtype=np.float32); gs=np.asarray(d['geom_std'],dtype=np.float32)
    if a.rank>coeff.shape[1]: raise RuntimeError(f'rank {a.rank}>{coeff.shape[1]}')
    X,names=make_features(geom,tau,gm,gs); Y=coeff[:,:a.rank]; tr=split=='train'; va=split=='val'
    y_scale=float(np.max(np.abs(Y[tr])))
    if not np.isfinite(y_scale) or y_scale<=0:
        raise RuntimeError(f'invalid y_scale {y_scale}')

    # Network predicts globally normalized POD coefficients.
    # A single scalar preserves the relative weighting among POD modes,
    # while keeping all representations in a numerically comparable output range.
    Y_norm=(Y / np.float32(y_scale)).astype(np.float32)

    Xtr=torch.from_numpy(X[tr])
    Ytr=torch.from_numpy(Y_norm[tr])
    Xva=torch.from_numpy(X[va])
    Yva=torch.from_numpy(Y_norm[va])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=ROMMLP(X.shape[1],a.rank,a.width,a.depth).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    print('='*110); print('TRAIN PROCESS-ENHANCED POD-ROM MLP'); print('='*110)
    print('representation:',rep,'device:',device,'rank:',a.rank,'inputs:',X.shape[1],'train rows:',int(tr.sum()),'val rows:',int(va.sum()),'params:',sum(p.numel() for p in model.parameters()),'y_scale:',y_scale)
    best=float('inf'); best_epoch=-1; best_state=None; bad=0; n=Xtr.shape[0]
    for epoch in range(a.epochs):
        model.train(); perm=torch.randperm(n); total=0.0
        for i in range(0,n,a.batch_size):
            idx=perm[i:i+a.batch_size]; xb=Xtr[idx].to(device,non_blocking=True); yb=Ytr[idx].to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); pred=model(xb); loss=nn.functional.mse_loss(pred,yb); loss.backward(); opt.step(); total += float(loss.detach().cpu())*len(idx)
        sched.step(); model.eval(); preds=[]
        with torch.no_grad():
            for i in range(0,Xva.shape[0],4096): preds.append(model(Xva[i:i+4096].to(device)).cpu())
        vp=torch.cat(preds); val=float(nn.functional.mse_loss(vp,Yva)); train=total/n
        if val<best-1e-12:
            best=val; best_epoch=epoch; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if epoch==0 or (epoch+1)%25==0 or epoch==a.epochs-1:
            print(f'epoch={epoch:04d} train_scaled_mse={train:.6e} val_scaled_mse={val:.6e} best={best:.6e}@{best_epoch}',flush=True)
        if bad>=a.patience:
            print(f'early stop epoch={epoch} patience={a.patience}',flush=True); break
    if best_state is None: raise RuntimeError('no best state')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    ck={'state_dict':best_state,'rank':a.rank,'width':a.width,'depth':a.depth,'geom_mean':gm,'geom_std':gs,'feature_names':names,'n_inputs':int(X.shape[1]),'y_scale':y_scale,'seed':a.seed,'best_epoch':best_epoch,'best_val_scaled_mse':best,'batch_size':a.batch_size,'lr':a.lr,'weight_decay':a.weight_decay,'dataset':str(a.dataset),'representation':rep,'coefficient_output_space':'normalized_by_global_train_max'}
    torch.save(ck,a.output)
    a.output.with_suffix('.json').write_text(json.dumps({k:v for k,v in ck.items() if k!='state_dict' and not isinstance(v,np.ndarray)},indent=2,default=str))
    print('wrote:',a.output)

if __name__=='__main__': main()
