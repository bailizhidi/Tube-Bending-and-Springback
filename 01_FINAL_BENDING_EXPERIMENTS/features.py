"""Unified X3/X6/X10 runtime node-feature contracts.

X3  = node-type one-hot (free/tool/clamped)
X6  = X3 + normalized global design scalars [D, t/D, R/D]
X10 = X6 + tube coordinates [s/L, sin(phi), cos(phi)] + angle_progress

The three contracts are strictly nested so the feature-ablation study changes
only the information content of x, not data splits, target representation, or
training code.
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch

CACHE_FEATURE_CONTRACT = "ANARESID_X10_shared_raw_packed_v2"
DESIGN_RANGES = {
    "D_outer": (6.0, 20.0),
    "t_over_D": (0.04, 0.14),
    "R_over_D": (1.25, 4.50),
}
FEATURE_NAMES = {
    "x3": (
        "node_type_free","node_type_tool","node_type_clamped",
    ),
    "x6": (
        "node_type_free","node_type_tool","node_type_clamped",
        "D_outer_design_norm","t_over_D_design_norm","R_over_D_design_norm",
    ),
    "x10": (
        "node_type_free","node_type_tool","node_type_clamped",
        "D_outer_design_norm","t_over_D_design_norm","R_over_D_design_norm",
        "s_over_L","sin_phi","cos_phi","angle_progress",
    ),
}


def canonical_feature_mode(value: object) -> str:
    mode = str(value).strip().lower()
    if mode not in FEATURE_NAMES:
        raise ValueError(f"feature_mode must be one of {sorted(FEATURE_NAMES)}, got {value!r}")
    return mode


def num_input_features(mode: object) -> int:
    return len(FEATURE_NAMES[canonical_feature_mode(mode)])


def design_range_normalize(value: float, low: float, high: float) -> float:
    if not high > low:
        raise ValueError(f"invalid design range [{low},{high}]")
    return 2.0 * (float(value)-float(low))/(float(high)-float(low)) - 1.0


def _deterministic_axis_sign(axis: torch.Tensor) -> torch.Tensor:
    dominant = int(torch.argmax(torch.abs(axis)).item())
    return -axis if float(axis[dominant].item()) < 0.0 else axis


def _reference_frame(axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    ref = eye[int(torch.argmin(torch.abs(eye @ axis)).item())]
    e1 = torch.linalg.cross(axis, ref)
    e1 = e1 / torch.clamp(torch.linalg.norm(e1), min=1e-8)
    e2 = torch.linalg.cross(axis, e1)
    e2 = e2 / torch.clamp(torch.linalg.norm(e2), min=1e-8)
    return e1, e2


def build_static_geo6(packed: Dict, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    mesh_pos = packed["mesh_pos"].to(device, non_blocking=True).float()
    tube = packed["is_tube_node"].to(device, non_blocking=True).reshape(-1).bool()
    if int(tube.sum()) < 8:
        raise RuntimeError("too few tube nodes for GEO coordinates")
    d = float(packed["D_outer"]); th = float(packed["Thickness"]); rb = float(packed["R_bending"])
    if min(d,th,rb) <= 0:
        raise ValueError(f"invalid geometry D={d}, t={th}, R={rb}")
    tube_pos = mesh_pos[tube]
    centroid = tube_pos.mean(0)
    centered = tube_pos-centroid
    cov = centered.T @ centered / max(int(tube_pos.shape[0])-1,1)
    _, eigvec = torch.linalg.eigh(cov)
    axis = _deterministic_axis_sign(eigvec[:,-1])
    axial = centered @ axis
    amin, amax = axial.min(), axial.max()
    length = amax-amin
    if float(length.item()) < 1e-6: raise RuntimeError("degenerate tube axis")
    s_over_l = (axial-amin)/length
    radial = centered - axial[:,None]*axis[None,:]
    e1,e2 = _reference_frame(axis)
    r1,r2 = radial@e1, radial@e2
    rn = torch.clamp(torch.sqrt(r1.square()+r2.square()), min=1e-8)
    cos_phi, sin_phi = r1/rn, r2/rn
    td, rd = th/d, rb/d
    vals = [
        design_range_normalize(d,*DESIGN_RANGES["D_outer"]),
        design_range_normalize(td,*DESIGN_RANGES["t_over_D"]),
        design_range_normalize(rd,*DESIGN_RANGES["R_over_D"]),
    ]
    geo = torch.zeros((mesh_pos.shape[0],6),dtype=torch.float32,device=device)
    geo[:,0],geo[:,1],geo[:,2] = vals
    geo[tube,3],geo[tube,4],geo[tube,5] = s_over_l,sin_phi,cos_phi
    if not torch.isfinite(geo).all(): raise RuntimeError("GEO6 contains NaN/Inf")
    diag = {"D_outer_design_norm":vals[0],"t_over_D_design_norm":vals[1],"R_over_D_design_norm":vals[2]}
    return geo.contiguous(), diag


def build_runtime_features(packed: Dict, step_index: int, device: torch.device, feature_mode: str):
    mode = canonical_feature_mode(feature_mode)
    if str(packed.get("feature_contract","")) != CACHE_FEATURE_CONTRACT:
        raise RuntimeError(f"unexpected packed feature contract {packed.get('feature_contract')!r}")
    x3 = packed["static_x3"].to(device,non_blocking=True).float()
    if x3.ndim != 2 or x3.shape[1] != 3: raise RuntimeError(f"static_x3 shape={tuple(x3.shape)}")
    T = int(packed["num_time_steps"])
    if not (0 <= int(step_index) < T-1): raise IndexError(step_index)
    if mode == "x3":
        return x3.contiguous(), {"angle_progress":float(step_index)/(T-1)}
    key = "_runtime_static_geo6"
    dkey = "_runtime_geo6_diag"
    geo = packed.get(key)
    if geo is None or not torch.is_tensor(geo) or geo.device != device:
        geo,diag = build_static_geo6(packed,device)
        packed[key]=geo; packed[dkey]=diag
    geo = packed[key]
    if mode == "x6":
        x = torch.cat([x3,geo[:,:3]],dim=-1)
    else:
        progress = torch.full((x3.shape[0],1),float(step_index)/(T-1),dtype=torch.float32,device=device)
        x = torch.cat([x3,geo,progress],dim=-1)
    if x.shape[1] != num_input_features(mode): raise RuntimeError(f"feature shape {tuple(x.shape)} for {mode}")
    if not torch.isfinite(x).all(): raise RuntimeError("node features contain NaN/Inf")
    diag = dict(packed.get(dkey,{})); diag["angle_progress"] = float(step_index)/(T-1)
    return x.contiguous(),diag
