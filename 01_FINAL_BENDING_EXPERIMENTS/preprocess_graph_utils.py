from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from graph_builder import build_world_edge_index,relative_edge4

def canonical_split(value: object) -> str:
    if isinstance(value,bytes): value=value.decode('utf-8','replace')
    text=str(value).strip().lower(); aliases={'train':'train','training':'train','tr':'train','val':'val','valid':'val','validation':'val','dev':'val','test':'test','testing':'test','te':'test'}
    if text not in aliases: raise ValueError(f'Unknown split value: {value!r}')
    return aliases[text]
def scalar_from_npz(data,keys,default=None):
    for key in keys:
        if key not in data.files: continue
        arr=data[key]
        if arr.size==0: continue
        v=arr.reshape(-1)[0]
        if isinstance(v,bytes): return v.decode('utf-8','replace')
        if hasattr(v,'item'):
            try: return v.item()
            except Exception: pass
        return v
    return default
def one_hot_node_type(node_type:torch.Tensor)->torch.Tensor:
    raw=node_type.reshape(-1).long(); mapped=torch.full_like(raw,-1); mapped[raw==0]=0; mapped[raw==1]=1; mapped[raw==3]=2
    if torch.any(mapped<0): raise ValueError(f'Unsupported node_type values: {torch.unique(raw[mapped<0]).tolist()}')
    return F.one_hot(mapped,num_classes=3).float()
def cells_to_mesh_edge_index(cells:torch.Tensor,cell_num_nodes:Optional[torch.Tensor]=None)->torch.Tensor:
    cells=torch.as_tensor(cells,dtype=torch.long)
    if cell_num_nodes is None: cell_num_nodes=torch.full((cells.shape[0],),cells.shape[1],dtype=torch.long)
    else: cell_num_nodes=torch.as_tensor(cell_num_nodes,dtype=torch.long).reshape(-1)
    src=[]; dst=[]
    for i in range(cells.shape[0]):
        n=int(cell_num_nodes[i]); conn=cells[i,:n].tolist(); pairs=((0,1),(1,2),(2,0)) if n==3 else (((0,1),(1,2),(2,3),(3,0),(0,2),(1,3)) if n==4 else ())
        for a,b in pairs: src.append(int(conn[a])); dst.append(int(conn[b]))
    if not src: raise ValueError('No valid mesh edges')
    e=torch.tensor([src,dst],dtype=torch.long); e=pyg.utils.to_undirected(e); e=pyg.utils.coalesce(e); e=e[0] if isinstance(e,tuple) else e; return e.contiguous()
