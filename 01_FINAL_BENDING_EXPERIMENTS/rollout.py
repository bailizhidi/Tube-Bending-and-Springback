from __future__ import annotations
from typing import Dict,List,Optional
import csv,os
import numpy as np,torch
from packed_dataset import read_manifest,safe_torch_load
from target_space import exact_state,state_to_world,load_target_stats
from graph_builder import build_graph_from_world

def unwrap_model(m): return m.module if hasattr(m,'module') else m
@torch.no_grad()
def rollout_one(model,packed,edge_mean,edge_std,target_stats,device,cfg,collect_positions=False):
    base=unwrap_model(model); was=base.training; base.eval(); T=int(packed['num_time_steps']); nt=packed['node_type'].to(device).reshape(-1).long(); tube=packed['is_tube_node'].to(device).reshape(-1).bool(); free=(nt==0)&tube; cover=~free
    mean=target_stats['mean'].to(device).reshape(1,3); std=target_stats['std'].to(device).reshape(1,3); state=exact_state(packed,0,device,str(cfg.prediction_mode),float(cfg.theta_final_deg)); errs=[]; preds=[]; exacts=[]
    for t in range(T-1):
        world=state_to_world(packed,state,t,device,str(cfg.prediction_mode),float(cfg.theta_final_deg)); g,me,we=build_graph_from_world(packed=packed,step_index=t,current_world=world,edge_mean=edge_mean,edge_std=edge_std,device=device,world_edge_radius=float(cfg.world_edge_radius),feature_mode=str(cfg.feature_mode))
        with torch.amp.autocast('cuda',dtype=torch.bfloat16,enabled=bool(cfg.amp)): pred=base(g.x,me,we,g)
        candidate=state+pred.float()*std+mean; exact_state_next=exact_state(packed,t+1,device,str(cfg.prediction_mode),float(cfg.theta_final_deg)); candidate=torch.where(cover[:,None],exact_state_next,candidate); pw=state_to_world(packed,candidate,t+1,device,str(cfg.prediction_mode),float(cfg.theta_final_deg)); ew=packed['world_pos'][t+1].to(device).float(); e=torch.linalg.norm(pw[free]-ew[free],dim=-1); errs.append(e.detach().cpu());
        if collect_positions: preds.append(pw.detach().cpu()); exacts.append(ew.detach().cpu())
        state=candidate.detach()
    allerr=torch.cat(errs); frame_means=[float(x.mean()) for x in errs]
    out={'sample_id':int(packed['sample_id']),'mean_free_mm':float(allerr.mean()),'final_free_mean_mm':frame_means[-1],'p95_free_mm':float(torch.quantile(allerr,0.95)),'p99_free_mm':float(torch.quantile(allerr,0.99)),'max_free_mm':float(allerr.max()),'nan_count':int((~torch.isfinite(allerr)).sum()),'num_steps':T-1}
    if collect_positions: out['pred_world_pos']=torch.stack(preds).numpy(); out['exact_world_pos']=torch.stack(exacts).numpy()
    if was: base.train()
    return out
@torch.no_grad()
def evaluate_manifest(model,cache_dir,split,edge_mean,edge_std,target_stats_path,device,cfg,output_csv=None,collect_positions_dir=None):
    ts=load_target_stats(target_stats_path,str(cfg.prediction_mode)); rows=[]; paths=read_manifest(cache_dir,split)
    if collect_positions_dir: os.makedirs(collect_positions_dir,exist_ok=True)
    for i,path in enumerate(paths,1):
        p=safe_torch_load(path)
        for k in ('mesh_pos','node_type','is_tube_node','is_tool_node','static_x3','mesh_edge_index','mesh_edge_attr_static_raw'): p[k]=p[k].to(device,non_blocking=True)
        r=rollout_one(model,p,edge_mean,edge_std,ts,device,cfg,bool(collect_positions_dir))
        if collect_positions_dir:
            np.savez_compressed(os.path.join(collect_positions_dir,f"sample_{r['sample_id']:04d}_rollout.npz"),pred_world_pos=r.pop('pred_world_pos'),exact_world_pos=r.pop('exact_world_pos'))
        rows.append(r); print(f"[{i}/{len(paths)}] sample={r['sample_id']:04d} mean={r['mean_free_mm']:.6f} final={r['final_free_mean_mm']:.6f} max={r['max_free_mm']:.6f}",flush=True)
    worst=max(rows,key=lambda x:x['mean_free_mm']); summary={'split':split,'num_samples':len(rows),'mean_free_mm':float(np.mean([x['mean_free_mm'] for x in rows])),'final_free_mean_mm':float(np.mean([x['final_free_mean_mm'] for x in rows])),'mean_p95_free_mm':float(np.mean([x['p95_free_mm'] for x in rows])),'max_free_mm':max(x['max_free_mm'] for x in rows),'worst_sample_id':worst['sample_id'],'worst_sample_mean_free_mm':worst['mean_free_mm'],'nan_count':sum(x['nan_count'] for x in rows),'rows':rows}
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or '.',exist_ok=True); f=open(output_csv,'w',newline='',encoding='utf-8'); w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows); f.close()
    return summary
