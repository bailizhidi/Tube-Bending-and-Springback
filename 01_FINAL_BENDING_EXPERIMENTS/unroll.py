"""Differentiable one/two-step unroll for direct and ANARESID targets."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List,Sequence,Dict
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from graph_builder import build_graph_from_world
from target_space import exact_state,state_to_world,canonical_prediction_mode

@dataclass
class UnrollResult:
    loss: torch.Tensor
    step_losses: List[torch.Tensor]
    step_free_mae_mm: List[torch.Tensor]
    step_world_edges: List[int]
    noise_std_xyz_mm: torch.Tensor


def validate_training_contract(cfg):
    k=int(cfg.multistep_rollout_steps); w=[float(x) for x in cfg.multistep_loss_weights]
    if k not in (1,2): raise ValueError("multistep_rollout_steps must be 1 or 2")
    expected=[1.0] if k==1 else [1.0,0.5]
    if w!=expected: raise ValueError(f"loss weights must be {expected}, got {w}")
    if bool(cfg.multistep_detach_between_steps): raise ValueError("detach_between_steps must be false")
    canonical_prediction_mode(cfg.prediction_mode)


def _forward(model,g,me,we,checkpoint_whole):
    if not checkpoint_whole: return model(g.x,me,we,g)
    def fn(x,a,b,gr=g): return model(x,a,b,gr)
    return activation_checkpoint(fn,g.x,me,we,use_reentrant=False,preserve_rng_state=True)


def unroll(*,model,packed,start_step,edge_mean,edge_std,target_mean,target_std,device,world_edge_radius,feature_mode,prediction_mode,theta_final_deg,loss_weights,noise_alpha,training,amp_enabled,amp_dtype=torch.bfloat16,use_whole_model_checkpoint=False):
    mode=canonical_prediction_mode(prediction_mode); K=len(loss_weights)
    T=int(packed['num_time_steps']); start=int(start_step)
    if start<0 or start+K>=T: raise IndexError(start)
    tube=packed['is_tube_node'].reshape(-1).bool(); nt=packed['node_type'].reshape(-1).long(); free=(nt==0)&tube; cover=~free
    mean=target_mean.to(device).reshape(1,3).float(); std=target_std.to(device).reshape(1,3).float()
    current=exact_state(packed,start,device,mode,theta_final_deg)
    noise_std=torch.zeros(3,dtype=current.dtype,device=device)
    if training and float(noise_alpha)>0:
        noise_std=(float(noise_alpha)*std.reshape(-1)).to(current.dtype)
        noise=torch.randn_like(current)*noise_std.reshape(1,3)
        noise=torch.where(free[:,None],noise,torch.zeros_like(noise))
        current=current+noise
    losses=[]; maes=[]; nedges=[]
    for local in range(K):
        t=start+local
        current_world=state_to_world(packed,current,t,device,mode,theta_final_deg)
        g,me,we=build_graph_from_world(packed=packed,step_index=t,current_world=current_world,edge_mean=edge_mean,edge_std=edge_std,device=device,world_edge_radius=world_edge_radius,feature_mode=feature_mode)
        with torch.amp.autocast('cuda',dtype=amp_dtype,enabled=bool(amp_enabled)):
            pred=_forward(model,g,me,we,bool(use_whole_model_checkpoint))
        pred=pred.float()
        if pred.ndim!=2 or pred.shape[-1]!=3: raise RuntimeError(f"model output shape={tuple(pred.shape)}")
        delta=pred*std+mean
        candidate=current+delta
        exact_next=exact_state(packed,t+1,device,mode,theta_final_deg)
        err_n=(candidate-exact_next)/std
        loss=err_n[tube].square().mean()
        if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
        pred_world=state_to_world(packed,candidate,t+1,device,mode,theta_final_deg)
        exact_world=packed['world_pos'][t+1].to(device,non_blocking=True).float()
        mae=torch.linalg.norm(pred_world[free]-exact_world[free],dim=-1).mean()
        losses.append(loss); maes.append(mae); nedges.append(int(g.num_world_edges.item()))
        current=torch.where(cover[:,None],exact_next,candidate)
    total=sum(float(w)*l for w,l in zip(loss_weights,losses))/sum(float(w) for w in loss_weights)
    return UnrollResult(total,losses,maes,nedges,noise_std)
