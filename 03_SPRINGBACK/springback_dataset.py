# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import get_angle, get_cells, get_edge_index, get_sample_id, get_target, get_x_bend, get_x_spring, make_edge_features, make_node_features

class GraphData:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)
    def to(self, device):
        return GraphData(**{k:(v.to(device) if torch.is_tensor(v) else v) for k,v in self.__dict__.items()})

class RunningStats:
    def __init__(self, dim):
        self.dim=int(dim); self.n=0
        self.sum=torch.zeros(self.dim,dtype=torch.float64); self.sq=torch.zeros(self.dim,dtype=torch.float64)
    def update(self, arr):
        x=torch.as_tensor(arr,dtype=torch.float64).reshape(-1,self.dim)
        if x.numel()==0:return
        self.n += int(x.shape[0]); self.sum += x.sum(0).cpu(); self.sq += (x*x).sum(0).cpu()
    def finalize(self, eps=1e-8):
        m=self.sum/float(self.n); v=torch.clamp(self.sq/float(self.n)-m*m,min=0.0); s=torch.sqrt(v)
        s=torch.where(s<eps,torch.ones_like(s),s)
        return m.float(),s.float()

def build_raw_data(path: str, feature_set: str = "geom") -> GraphData:
    with np.load(path, allow_pickle=False) as z:
        xb=get_x_bend(z); xs=get_x_spring(z); y=get_target(z)
        x=make_node_features(z,path,feature_set); ei=get_edge_index(z,xb.shape[0]); ea=make_edge_features(xb,ei); cells=get_cells(z)
        clamp=np.asarray(z["clamp_mask"],dtype=np.bool_).reshape(-1)
        sid=get_sample_id(z,path); angle=get_angle(z,path)
    return GraphData(x=torch.tensor(x), edge_index=torch.tensor(ei,dtype=torch.long), edge_attr=torch.tensor(ea), y=torch.tensor(y),
                     pos=torch.tensor(xb), x_bend=torch.tensor(xb), x_spring=torch.tensor(xs), cells=torch.tensor(cells,dtype=torch.long),
                     clamp_mask=torch.tensor(clamp), sample_id=torch.tensor([sid],dtype=torch.long), angle_deg=torch.tensor([angle],dtype=torch.float32))

def collate_graphs(batch: List[GraphData]) -> GraphData:
    if len(batch)==1:
        g=batch[0]; g.batch=torch.zeros(g.x.shape[0],dtype=torch.long); g.ptr=torch.tensor([0,g.x.shape[0]],dtype=torch.long); g.num_graphs=1; return g
    xs=[]; eas=[]; ys=[]; poss=[]; xbs=[]; xss=[]; clamps=[]; eis=[]; sids=[]; ang=[]; bvec=[]; ptr=[0]; off=0
    for gi,g in enumerate(batch):
        n=g.x.shape[0]; xs.append(g.x); eas.append(g.edge_attr); ys.append(g.y); poss.append(g.pos); xbs.append(g.x_bend); xss.append(g.x_spring); clamps.append(g.clamp_mask)
        eis.append(g.edge_index+off); sids.append(g.sample_id); ang.append(g.angle_deg); bvec.append(torch.full((n,),gi,dtype=torch.long)); off+=n; ptr.append(off)
    return GraphData(x=torch.cat(xs), edge_attr=torch.cat(eas), y=torch.cat(ys), pos=torch.cat(poss), x_bend=torch.cat(xbs), x_spring=torch.cat(xss),
                     clamp_mask=torch.cat(clamps), edge_index=torch.cat(eis,1), sample_id=torch.cat(sids), angle_deg=torch.cat(ang),
                     batch=torch.cat(bvec), ptr=torch.tensor(ptr,dtype=torch.long), num_graphs=len(batch))

def denormalize_y(y_norm: torch.Tensor, stats: Dict[str,torch.Tensor]):
    return y_norm*stats["y_std"].to(y_norm.device)+stats["y_mean"].to(y_norm.device)
