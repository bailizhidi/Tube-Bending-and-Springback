# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, random, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from cached_springback_dataset import CachedSpringbackDataset
from model import MeshGraphNetOneStep
from springback_dataset import collate_graphs, denormalize_y
from utils import ensure_dir, read_text_lines, read_yaml, save_json

def setup():
    if "RANK" not in os.environ:
        return 0,1,0,torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl"); rank=dist.get_rank(); world=dist.get_world_size(); local=int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); return rank,world,local,torch.device("cuda",local)
def mainrank(r): return r==0
def seed_all(seed,rank): random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank); torch.cuda.manual_seed_all(seed+rank)
def reduce_mean(x):
    if dist.is_initialized(): dist.all_reduce(x,op=dist.ReduceOp.SUM); x=x/dist.get_world_size()
    return x

def weighted_mean(v,w): return (v*w).sum()/(w.sum()+1e-12)
def angle_node_weight(data,cfg,device,dtype):
    a=float(cfg.get("angle_weight_alpha",0.0)); p=float(cfg.get("angle_weight_power",2.0))
    if a<=0:return torch.ones((data.y.shape[0],1),device=device,dtype=dtype)
    graph_w=1.0+a*torch.clamp(data.angle_deg.to(device)/180.0,0.0,1.0).pow(p)
    return graph_w[data.batch].reshape(-1,1).to(dtype)

def make_loss(pred_norm,data,stats,cfg):
    target=data.y; w=angle_node_weight(data,cfg,pred_norm.device,pred_norm.dtype); diff=pred_norm-target
    mse=(diff**2).mean(1,keepdim=True); mae=torch.abs(diff).mean(1,keepdim=True)
    node_w=w
    bc=float(cfg.get("bc_loss_weight",0.0))
    if bc>0: node_w=node_w*(1.0+bc*data.clamp_mask.view(-1,1).float())
    ldata=weighted_mean(mse,node_w)+float(cfg.get("mae_loss_weight",0.2))*weighted_mean(mae,node_w)
    pp=denormalize_y(pred_norm,stats); tt=denormalize_y(target,stats)
    ew=float(cfg.get("edge_grad_loss_weight",0.0)); tw=float(cfg.get("topk_loss_weight",0.0))
    if ew>0:
        s,d=data.edge_index; e=((pp[d]-pp[s])-(tt[d]-tt[s]))**2; eval=e.mean(1,keepdim=True); eweight=0.5*(w[s]+w[d]); ledge=weighted_mean(eval,eweight)
    else: ledge=pred_norm.new_zeros(())
    if tw>0:
        # batch_size is intentionally 1 for heterogeneous meshes; TopK is graph-local.
        em=torch.linalg.norm(pp-tt,dim=1); ratio=float(cfg.get("topk_ratio",0.10)); k=max(1,min(em.numel(),int(np.ceil(ratio*em.numel())))); vals,idx=torch.topk(em,k,largest=True,sorted=False); ww=w.view(-1)[idx]; ltop=(vals*ww).sum()/(ww.sum()+1e-12)
    else: ltop=pred_norm.new_zeros(())
    return ldata+ew*ledge+tw*ltop,{"data":ldata.detach(),"edge":ledge.detach(),"topk":ltop.detach()}

@torch.no_grad()
def evaluate(model,loader,stats,cfg,device,max_batches=0):
    model.eval(); total_loss=0.; total_nodes=0; sum_mag=0.; max_mag=0.; nb=0
    for data in loader:
        data=data.to(device); pred=model(data); loss,_=make_loss(pred,data,stats,cfg); pp=denormalize_y(pred,stats); tt=denormalize_y(data.y,stats); mag=torch.linalg.norm(pp-tt,dim=1)
        n=mag.numel(); total_loss+=float(loss.item()); total_nodes+=n; sum_mag+=float(mag.sum().item()); max_mag=max(max_mag,float(mag.max().item())); nb+=1
        if max_batches>0 and nb>=max_batches: break
    return {"loss":total_loss/max(1,nb),"mae_mag":sum_mag/max(1,total_nodes),"max_mag_err":max_mag,"batches":nb}

def checkpoint(path,model,opt,sched,scaler,epoch,global_step,stats,cfg,splits,best):
    state=model.module.state_dict() if isinstance(model,DDP) else model.state_dict()
    torch.save({"model_state":state,"optimizer_state":opt.state_dict(),"scheduler_state":sched.state_dict(),"scaler_state":scaler.state_dict(),"epoch":epoch,"global_step":global_step,"stats":{k:v.cpu() if torch.is_tensor(v) else v for k,v in stats.items()},"cfg":cfg,"splits":splits,"best_val":best},path)

def main():
    rank,world,local,device=setup(); ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--processed_dir"); ap.add_argument("--output_dir"); ap.add_argument("--epochs",type=int); ap.add_argument("--max_steps_per_epoch",type=int,default=0); ap.add_argument("--val_max_batches",type=int,default=0); ap.add_argument("--resume",default=""); args=ap.parse_args()
    cfg=read_yaml(args.config); processed=args.processed_dir or cfg["processed_dir"]; out=args.output_dir or cfg["output_dir"]; epochs=args.epochs or int(cfg.get("epochs",100)); seed_all(int(cfg.get("seed",42)),rank)
    with open(os.path.join(processed,"processed_meta.json"),"r",encoding="utf-8") as f: meta=json.load(f)
    splitdir=os.path.join(processed,"splits"); train=read_text_lines(os.path.join(splitdir,"train_files.txt")); val=read_text_lines(os.path.join(splitdir,"val_files.txt")); stats_cpu=torch.load(os.path.join(processed,"stats.pt"),map_location="cpu",weights_only=False); stats={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in stats_cpu.items()}
    if mainrank(rank): ensure_dir(out); save_json(cfg,os.path.join(out,"config_resolved.json")); print("train={} val={} world={} epochs={}".format(len(train),len(val),world,epochs),flush=True)
    trset=CachedSpringbackDataset(train,stats_cpu); sampler=DistributedSampler(trset,num_replicas=world,rank=rank,shuffle=True,seed=int(cfg.get("seed",42))) if world>1 else None
    loader=DataLoader(trset,batch_size=int(cfg.get("batch_size",1)),sampler=sampler,shuffle=(sampler is None),num_workers=int(cfg.get("num_workers",2)),pin_memory=True,collate_fn=collate_graphs,persistent_workers=int(cfg.get("num_workers",2))>0)
    vloader=DataLoader(CachedSpringbackDataset(val,stats_cpu),batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_graphs) if mainrank(rank) else None
    model=MeshGraphNetOneStep(int(stats_cpu["num_node_features"].item()),int(stats_cpu["num_edge_features"].item()),int(stats_cpu["num_output_features"].item()),int(cfg.get("hidden_dim",128)),int(cfg.get("num_message_passing_steps",12)),int(cfg.get("mlp_layers",2)),float(cfg.get("dropout",0.0))).to(device)
    if world>1: model=DDP(model,device_ids=[local],output_device=local)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg.get("lr",1e-4)),weight_decay=float(cfg.get("weight_decay",0.0))); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); scaler=torch.amp.GradScaler("cuda",enabled=bool(cfg.get("amp",True)) and device.type=="cuda")
    best=float("inf"); global_step=0; start_epoch=0; valint=int(cfg.get("val_interval",5)); saveint=int(cfg.get("save_interval",25)); bestpath=os.path.join(out,"best_model.pt"); lastpath=os.path.join(out,"last_model.pt")
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False); target_model=model.module if isinstance(model,DDP) else model; target_model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"]); sched.load_state_dict(ck["scheduler_state"]); scaler.load_state_dict(ck.get("scaler_state",{})); start_epoch=int(ck.get("epoch",0)); global_step=int(ck.get("global_step",0)); best=float(ck.get("best_val",float("inf")))
        if mainrank(rank): print("[RESUME] path={} epoch={} global_step={} best_val={:.6e}".format(args.resume,start_epoch,global_step,best),flush=True)
    for epoch in range(start_epoch+1,epochs+1):
        if sampler is not None:sampler.set_epoch(epoch)
        model.train(); t0=time.time(); sums=torch.zeros(4,device=device); steps=0; torch.cuda.reset_peak_memory_stats(device) if device.type=="cuda" else None
        for data in loader:
            data=data.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=bool(cfg.get("amp",True)) and device.type=="cuda"):
                pred=model(data); loss,parts=make_loss(pred,data,stats,cfg)
            scaler.scale(loss).backward(); scaler.unscale_(opt); gc=float(cfg.get("grad_clip_norm",1.0));
            if gc>0: torch.nn.utils.clip_grad_norm_(model.parameters(),gc)
            scaler.step(opt); scaler.update(); global_step+=1; steps+=1; sums += torch.stack([loss.detach(),parts["data"],parts["edge"],parts["topk"]])
            if args.max_steps_per_epoch>0 and steps>=args.max_steps_per_epoch: break
        sched.step(); avg=reduce_mean(sums/max(1,steps)); do_val=(epoch%valint==0 or epoch==epochs or args.max_steps_per_epoch>0)
        if mainrank(rank):
            msg="[EPOCH] epoch={}/{} step={} train_loss={:.6e} data={:.6e} edge={:.6e} topk={:.6e} lr={:.3e} time_sec={:.2f}".format(epoch,epochs,global_step,*[float(x) for x in avg],opt.param_groups[0]["lr"],time.time()-t0)
            if device.type=="cuda": msg += " peak_reserved_gib={:.3f}".format(torch.cuda.max_memory_reserved(device)/1024**3)
            print(msg,flush=True)
            if do_val:
                vm=evaluate(model.module if isinstance(model,DDP) else model,vloader,stats,cfg,device,args.val_max_batches); print("[VAL] epoch={} mean_node_mae_mm={:.6e} max_node_err_mm={:.6e} loss={:.6e} batches={}".format(epoch,vm["mae_mag"],vm["max_mag_err"],vm["loss"],vm["batches"]),flush=True)
                if vm["mae_mag"]<best: best=vm["mae_mag"]; checkpoint(bestpath,model,opt,sched,scaler,epoch,global_step,stats_cpu,cfg,meta["splits_npz"],best); print("[BEST] {} {:.6e}".format(bestpath,best),flush=True)
            checkpoint(lastpath,model,opt,sched,scaler,epoch,global_step,stats_cpu,cfg,meta["splits_npz"],best)
            if saveint>0 and epoch%saveint==0:
                snap=os.path.join(out,"epoch_{:04d}.pt".format(epoch)); checkpoint(snap,model,opt,sched,scaler,epoch,global_step,stats_cpu,cfg,meta["splits_npz"],best); print("[SNAPSHOT] {}".format(snap),flush=True)
        if dist.is_initialized(): dist.barrier()
    if mainrank(rank): print("TRAINING COMPLETE best_val={:.6e}".format(best),flush=True)
    if dist.is_initialized(): dist.destroy_process_group()
if __name__=="__main__": main()
