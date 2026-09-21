from __future__ import annotations
import random
from typing import Dict, Iterator, List
import torch
from packed_dataset import read_manifest,safe_torch_load,sample_id_from_cache_path,load_normalization_stats
from target_space import load_target_stats
STATIC_KEYS=("mesh_pos","node_type","is_tube_node","is_tool_node","static_x3","mesh_edge_index","mesh_edge_attr_static_raw")
class PackedTrajectoryWindowIterator:
    def __init__(self,*,cache_dir,stats_dir,target_stats_path,prediction_mode,num_time_steps,unroll_steps,window_stride,seed,shuffle_windows,expected_train_samples):
        self.cache_dir=str(cache_dir); self.stats_dir=str(stats_dir); self.T=int(num_time_steps); self.K=int(unroll_steps); self.stride=int(window_stride); self.seed=int(seed); self.shuffle=bool(shuffle_windows)
        if self.K not in (1,2): raise ValueError("formal experiments support unroll_steps 1 or 2")
        self.train_files=read_manifest(self.cache_dir,'train')
        if len(self.train_files)!=int(expected_train_samples): raise RuntimeError(f"expected {expected_train_samples} train samples, got {len(self.train_files)}")
        self.stats=load_normalization_stats(self.stats_dir); self.target=load_target_stats(target_stats_path,prediction_mode)
        ids=sorted(sample_id_from_cache_path(p) for p in self.train_files)
        if ids!=sorted(self.stats['train_sample_ids']): raise RuntimeError("edge stats IDs != active train manifest")
        if ids!=sorted(self.target['train_sample_ids']): raise RuntimeError("target stats IDs != active train manifest")
        self.starts=list(range(0,self.T-self.K,self.stride))
    @property
    def windows_per_trajectory(self): return len(self.starts)
    def steps_per_rank(self,world_size):
        if len(self.train_files)%int(world_size): raise ValueError("train count not divisible by world size")
        return len(self.train_files)//int(world_size)*len(self.starts)
    def _files(self,epoch,rank,world_size):
        order=list(range(len(self.train_files))); random.Random(self.seed+int(epoch)).shuffle(order)
        return [self.train_files[i] for i in order[int(rank)::int(world_size)]]
    def iter_epoch(self,epoch,rank,world_size,device)->Iterator[Dict]:
        for path in self._files(epoch,rank,world_size):
            p=safe_torch_load(path)
            if str(p.get('cache_signature',''))!=str(self.stats['shared_cache_signature']): raise RuntimeError(f"cache signature mismatch {path}")
            for k in STATIC_KEYS: p[k]=p[k].to(device,non_blocking=True)
            starts=list(self.starts)
            if self.shuffle: random.Random(self.seed+epoch*1000003+int(p['sample_id'])*9176).shuffle(starts)
            for s in starts: yield {'packed':p,'start_step':int(s),'sample_id':int(p['sample_id']),'source_file':path}
            del p
