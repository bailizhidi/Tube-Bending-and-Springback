"""Graph construction shared by every final bending experiment.

The world-edge and edge-feature definitions intentionally match the validated
CLEAN V2 implementation exactly, so changing Direct/Global/Local or X3/X6/X10
does not silently change the graph contract.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch_geometric as pyg
from physicsnemo.nn.functional import radius_search
from features import build_runtime_features

def relative_edge4(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row,col=edge_index.long()
    disp=pos[row]-pos[col]
    return torch.cat([disp,torch.linalg.norm(disp,dim=-1,keepdim=True)],dim=-1)
def normalize_edge4(edge4,edge_mean,edge_std):
    mean=edge_mean.to(edge4.device,edge4.dtype).reshape(1,4); std=edge_std.to(edge4.device,edge4.dtype).reshape(1,4); std=torch.where(std.abs()<1e-8,torch.ones_like(std),std); return (edge4-mean)/std
def _exclude_mesh_edges(candidate,mesh_edge_index,num_nodes):
    if candidate.numel()==0: return candidate.reshape(2,0)
    mh=mesh_edge_index[0].long()*int(num_nodes)+mesh_edge_index[1].long(); mh=torch.sort(mh).values; ch=candidate[0].long()*int(num_nodes)+candidate[1].long(); pos=torch.searchsorted(mh,ch); valid=pos<mh.numel(); safe=torch.clamp(pos,max=max(mh.numel()-1,0)); is_mesh=valid & (mh[safe]==ch); return candidate[:,~is_mesh]
def build_world_edge_index(world_pos,mesh_edge_index,is_tube,radius:float):
    world_pos=world_pos.contiguous(); mesh_edge_index=mesh_edge_index.to(world_pos.device).long().contiguous(); tube=is_tube.to(world_pos.device).reshape(-1).bool()
    idx=radius_search(world_pos,world_pos,radius=float(radius),return_dists=False,return_points=False).long()
    if idx.numel()==0: return torch.empty((2,0),dtype=torch.long,device=world_pos.device)
    idx=idx[:,idx[0]!=idx[1]]
    if idx.numel()==0: return torch.empty((2,0),dtype=torch.long,device=world_pos.device)
    src,dst=idx; idx=idx[:,tube[src] | tube[dst]]
    if idx.numel()==0: return torch.empty((2,0),dtype=torch.long,device=world_pos.device)
    idx=_exclude_mesh_edges(idx,mesh_edge_index,world_pos.shape[0])
    if idx.numel()==0: return torch.empty((2,0),dtype=torch.long,device=world_pos.device)
    idx=pyg.utils.coalesce(idx); idx=idx[0] if isinstance(idx,tuple) else idx; return idx.contiguous()
def build_graph_from_world(*,packed:Dict,step_index:int,current_world:torch.Tensor,edge_mean,edge_std,device,world_edge_radius:float,feature_mode:str):
    mesh_pos=packed['mesh_pos'].to(device,non_blocking=True).float(); mesh_idx=packed['mesh_edge_index'].to(device,non_blocking=True).long(); static_raw=packed['mesh_edge_attr_static_raw'].to(device,non_blocking=True).float(); is_tube=packed['is_tube_node'].to(device,non_blocking=True).reshape(-1).bool(); node_type=packed['node_type'].to(device,non_blocking=True).reshape(-1).long(); x,diag=build_runtime_features(packed,step_index,device,feature_mode)
    world_idx=build_world_edge_index(current_world.detach(),mesh_idx,is_tube,float(world_edge_radius)); static4=normalize_edge4(static_raw,edge_mean,edge_std); current4=normalize_edge4(relative_edge4(current_world,mesh_idx),edge_mean,edge_std); mesh8=torch.cat([static4,current4],dim=-1).contiguous()
    if world_idx.numel():
        world4=normalize_edge4(relative_edge4(current_world,world_idx),edge_mean,edge_std); world8=world4.repeat(1,2).contiguous(); edge_index=torch.cat([mesh_idx,world_idx],dim=1).contiguous(); edge_attr=torch.cat([mesh8,world8],dim=0).contiguous()
    else:
        world8=torch.empty((0,8),dtype=current_world.dtype,device=device); edge_index=mesh_idx; edge_attr=mesh8
    g=pyg.data.Data(x=x,edge_index=edge_index,edge_attr=edge_attr,world_pos=current_world,mesh_pos=mesh_pos); g.is_tube_node=is_tube; g.is_tool_node=packed['is_tool_node'].to(device,non_blocking=True).reshape(-1).bool(); g.node_type_raw=node_type; g.num_world_edges=torch.tensor([world_idx.shape[1]],dtype=torch.long,device=device); g.sample_id=torch.tensor([int(packed['sample_id'])],dtype=torch.long,device=device); g.angle_progress=torch.tensor([diag.get('angle_progress',0.0)],dtype=torch.float32,device=device); return g,mesh8,world8
