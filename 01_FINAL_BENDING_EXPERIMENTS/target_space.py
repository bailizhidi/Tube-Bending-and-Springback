"""Unified prediction target/state spaces for controlled comparisons."""
from __future__ import annotations
from typing import Dict
import json, os
import torch
from analytical_kinematics import analytical_frame_from_packed, world_to_residual, residual_to_world

VALID_MODES = {"direct","global_residual","local_residual"}

def canonical_prediction_mode(value: object) -> str:
    m=str(value).strip().lower()
    aliases={"global":"global_residual","local":"local_residual","anaresid_global":"global_residual","anaresid_local":"local_residual"}
    m=aliases.get(m,m)
    if m not in VALID_MODES: raise ValueError(f"prediction_mode must be {sorted(VALID_MODES)}, got {value!r}")
    return m

def residual_frame_for_mode(mode: object):
    m=canonical_prediction_mode(mode)
    return None if m=="direct" else ("global" if m=="global_residual" else "local")

def exact_state(packed: Dict, frame_index: int, device: torch.device, mode: str, theta_final_deg: float):
    m=canonical_prediction_mode(mode)
    world=packed["world_pos"][int(frame_index)].to(device,non_blocking=True).float()
    if m=="direct": return world
    fr=analytical_frame_from_packed(packed,int(frame_index),device,theta_final_deg=float(theta_final_deg),exact_world_non_tube=True)
    return world_to_residual(world,fr,residual_frame_for_mode(m))

def state_to_world(packed: Dict, state: torch.Tensor, frame_index: int, device: torch.device, mode: str, theta_final_deg: float):
    m=canonical_prediction_mode(mode)
    if m=="direct": return state
    fr=analytical_frame_from_packed(packed,int(frame_index),device,theta_final_deg=float(theta_final_deg),exact_world_non_tube=True)
    return residual_to_world(state,fr,residual_frame_for_mode(m))

def load_target_stats(path: str, expected_mode: str):
    if not os.path.isfile(path): raise FileNotFoundError(path)
    raw=json.load(open(path,'r',encoding='utf-8'))
    mode=canonical_prediction_mode(raw.get('prediction_mode',''))
    if mode != canonical_prediction_mode(expected_mode): raise RuntimeError(f"target stats mode={mode}, expected={expected_mode}")
    mean=torch.tensor(raw['mean'],dtype=torch.float32)
    std=torch.tensor(raw['std'],dtype=torch.float32)
    if mean.shape!=(3,) or std.shape!=(3,) or torch.any(std<=0) or not torch.isfinite(std).all():
        raise RuntimeError(f"invalid target stats in {path}")
    return {"mean":mean,"std":std,"train_sample_ids":[int(x) for x in raw['train_sample_ids']],"raw":raw}
