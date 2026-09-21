# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Sequence
import torch
from torch.utils.data import Dataset
from springback_dataset import GraphData

class CachedSpringbackDataset(Dataset):
    def __init__(self, pt_files: Sequence[str], stats: Dict[str,torch.Tensor]): self.pt_files=list(pt_files); self.stats=stats
    def __len__(self): return len(self.pt_files)
    def __getitem__(self, idx):
        d=torch.load(self.pt_files[idx],map_location="cpu",weights_only=False); g=GraphData(**d)
        g.x_raw=g.x.clone(); g.edge_attr_raw=g.edge_attr.clone(); g.y_raw=g.y.clone()
        g.x=(g.x-self.stats["node_mean"])/self.stats["node_std"]
        g.edge_attr=(g.edge_attr-self.stats["edge_mean"])/self.stats["edge_std"]
        g.y=(g.y-self.stats["y_mean"])/self.stats["y_std"]
        return g
