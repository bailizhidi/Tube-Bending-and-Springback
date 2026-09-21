"""Analytical-residual kinematics for rotary-draw tube bending.

Contract
--------
For a tube reference point ``(x, y, s)`` and commanded bend angle ``theta``::

    psi = clip(theta - s / R, 0, theta)
    u   = s - R * (theta - psi)

    e_y(psi) = (0, cos(psi), -sin(psi))
    e_z(psi) = (0, sin(psi),  cos(psi))

    A = (x, -R, 0) + (R + y) * e_y(psi) + u * e_z(psi)

The learned state is the deviation ``e = X - A``.  In ``local`` mode the
world residual is projected onto the orthonormal basis
``[e_x, e_y(psi), e_z(psi)]``.

The implementation is intentionally exact/invertible and keeps the analytical
baseline independent of the neural-network prediction.  Tool nodes use the
known exact tool position as their analytical baseline, hence their residual is
identically zero; tool/clamped nodes are still overwritten with truth after
prediction in the multistep/rollout code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math

import torch


VALID_RESIDUAL_FRAMES = {"global", "local"}


@dataclass
class AnalyticalFrame:
    analytical_pos: torch.Tensor  # [N,3]
    psi: torch.Tensor             # [N]
    cos_psi: torch.Tensor         # [N]
    sin_psi: torch.Tensor         # [N]
    theta_rad: float


def canonical_residual_frame(value: object) -> str:
    frame = str(value).strip().lower()
    if frame not in VALID_RESIDUAL_FRAMES:
        raise ValueError(
            f"Unsupported residual_frame={value!r}; expected one of "
            f"{sorted(VALID_RESIDUAL_FRAMES)}"
        )
    return frame


def frame_angle_rad(
    packed: Dict,
    frame_index: int,
    device: torch.device,
    theta_final_deg: float = 180.0,
) -> torch.Tensor:
    """Return the prescribed bend angle for one saved FE frame.

    If preprocessing later stores an explicit ``theta_rad_per_frame`` array it
    is used directly.  Otherwise this package uses the validated TC4 dataset
    contract: 181 saved frames from 0 to the fixed terminal bend angle 180 deg.
    """
    frame_index = int(frame_index)
    num_time_steps = int(packed["num_time_steps"])
    if not (0 <= frame_index < num_time_steps):
        raise IndexError(
            f"frame_index={frame_index} outside [0,{num_time_steps - 1}]"
        )

    if "theta_rad_per_frame" in packed:
        values = torch.as_tensor(packed["theta_rad_per_frame"])
        if values.numel() != num_time_steps:
            raise ValueError(
                "theta_rad_per_frame length mismatch: "
                f"{values.numel()} != {num_time_steps}"
            )
        return values.reshape(-1)[frame_index].to(device=device, dtype=torch.float32)

    theta_final = math.radians(float(theta_final_deg))
    theta = theta_final * float(frame_index) / float(num_time_steps - 1)
    return torch.tensor(theta, device=device, dtype=torch.float32)


def analytical_frame(
    mesh_pos: torch.Tensor,
    is_tube_node: torch.Tensor,
    r_bending: float,
    theta_rad: torch.Tensor | float,
    *,
    exact_world_non_tube: Optional[torch.Tensor] = None,
) -> AnalyticalFrame:
    """Compute analytical baseline ``A`` and local section rotation ``psi``.

    ``mesh_pos`` must use the dataset's canonical world convention: x is the
    tube cross-section x coordinate and z is the reference axial/material
    coordinate ``s``.  At theta=0 the map is exactly the identity on tube nodes.
    """
    mesh_pos = torch.as_tensor(mesh_pos)
    if mesh_pos.ndim != 2 or mesh_pos.shape[1] != 3:
        raise ValueError(f"mesh_pos must be [N,3], got {tuple(mesh_pos.shape)}")
    tube = torch.as_tensor(is_tube_node, device=mesh_pos.device).reshape(-1).bool()
    if tube.numel() != mesh_pos.shape[0]:
        raise ValueError("is_tube_node length mismatch")

    r = float(r_bending)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError(f"Invalid R_bending={r_bending!r}")

    theta_t = torch.as_tensor(theta_rad, device=mesh_pos.device, dtype=mesh_pos.dtype).reshape(())
    if not torch.isfinite(theta_t):
        raise ValueError(f"Non-finite theta_rad={theta_rad!r}")
    if float(theta_t.item()) < -1.0e-10:
        raise ValueError(f"theta_rad must be non-negative, got {float(theta_t.item())}")

    x = mesh_pos[:, 0]
    y = mesh_pos[:, 1]
    s = mesh_pos[:, 2]  # fixed reference axial/material coordinate

    psi = torch.clamp(theta_t - s / r, min=0.0)
    psi = torch.minimum(psi, theta_t.expand_as(psi))
    c = torch.cos(psi)
    sn = torch.sin(psi)
    u = s - r * (theta_t - psi)

    analytical = torch.empty_like(mesh_pos)
    analytical[:, 0] = x
    analytical[:, 1] = -r + (r + y) * c + u * sn
    analytical[:, 2] = -(r + y) * sn + u * c

    # The analytical tube map is not intended to prescribe tool kinematics.
    # Tool nodes are known exactly and therefore receive A=X_exact, e=0.
    not_tube = ~tube
    if torch.any(not_tube):
        if exact_world_non_tube is not None:
            exact_world_non_tube = torch.as_tensor(
                exact_world_non_tube, device=mesh_pos.device, dtype=mesh_pos.dtype
            )
            if exact_world_non_tube.shape != mesh_pos.shape:
                raise ValueError(
                    "exact_world_non_tube shape mismatch: "
                    f"{tuple(exact_world_non_tube.shape)} != {tuple(mesh_pos.shape)}"
                )
            analytical[not_tube] = exact_world_non_tube[not_tube]
        else:
            analytical[not_tube] = mesh_pos[not_tube]
        psi = psi.clone()
        c = c.clone()
        sn = sn.clone()
        psi[not_tube] = 0.0
        c[not_tube] = 1.0
        sn[not_tube] = 0.0

    if not torch.isfinite(analytical).all():
        raise RuntimeError("Analytical baseline contains NaN/Inf")

    return AnalyticalFrame(
        analytical_pos=analytical.contiguous(),
        psi=psi.contiguous(),
        cos_psi=c.contiguous(),
        sin_psi=sn.contiguous(),
        theta_rad=float(theta_t.detach().cpu().item()),
    )


def analytical_frame_from_packed(
    packed: Dict,
    frame_index: int,
    device: torch.device,
    *,
    theta_final_deg: float = 180.0,
    exact_world_non_tube: bool = True,
) -> AnalyticalFrame:
    mesh_pos = packed["mesh_pos"].to(device, non_blocking=True).float()
    is_tube = packed["is_tube_node"].to(device, non_blocking=True).reshape(-1).bool()
    theta = frame_angle_rad(
        packed, frame_index, device=device, theta_final_deg=float(theta_final_deg)
    )
    exact = None
    if exact_world_non_tube:
        exact = packed["world_pos"][int(frame_index)].to(device, non_blocking=True).float()
    return analytical_frame(
        mesh_pos=mesh_pos,
        is_tube_node=is_tube,
        r_bending=float(packed["R_bending"]),
        theta_rad=theta,
        exact_world_non_tube=exact,
    )


def world_to_residual(
    world_pos: torch.Tensor,
    frame: AnalyticalFrame,
    residual_frame: str,
) -> torch.Tensor:
    """World position ``X`` -> analytical residual ``e``."""
    residual_frame = canonical_residual_frame(residual_frame)
    world_pos = torch.as_tensor(
        world_pos,
        device=frame.analytical_pos.device,
        dtype=frame.analytical_pos.dtype,
    )
    if world_pos.shape != frame.analytical_pos.shape:
        raise ValueError(
            f"world_pos shape {tuple(world_pos.shape)} != "
            f"analytical shape {tuple(frame.analytical_pos.shape)}"
        )
    d = world_pos - frame.analytical_pos
    if residual_frame == "global":
        return d

    # M=[ex,ey,ez] has columns in world coordinates; e_local=M^T d.
    out = torch.empty_like(d)
    out[:, 0] = d[:, 0]
    out[:, 1] = frame.cos_psi * d[:, 1] - frame.sin_psi * d[:, 2]
    out[:, 2] = frame.sin_psi * d[:, 1] + frame.cos_psi * d[:, 2]
    return out.contiguous()


def residual_to_world(
    residual: torch.Tensor,
    frame: AnalyticalFrame,
    residual_frame: str,
) -> torch.Tensor:
    """Analytical residual ``e`` -> world position ``X`` (exact inverse)."""
    residual_frame = canonical_residual_frame(residual_frame)
    residual = torch.as_tensor(
        residual,
        device=frame.analytical_pos.device,
        dtype=frame.analytical_pos.dtype,
    )
    if residual.shape != frame.analytical_pos.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} != "
            f"analytical shape {tuple(frame.analytical_pos.shape)}"
        )
    if residual_frame == "global":
        return (frame.analytical_pos + residual).contiguous()

    d = torch.empty_like(residual)
    d[:, 0] = residual[:, 0]
    d[:, 1] = frame.cos_psi * residual[:, 1] + frame.sin_psi * residual[:, 2]
    d[:, 2] = -frame.sin_psi * residual[:, 1] + frame.cos_psi * residual[:, 2]
    return (frame.analytical_pos + d).contiguous()


def assert_roundtrip(
    world_pos: torch.Tensor,
    frame: AnalyticalFrame,
    residual_frame: str,
    atol: float = 2.0e-5,
) -> float:
    residual = world_to_residual(world_pos, frame, residual_frame)
    recovered = residual_to_world(residual, frame, residual_frame)
    max_abs = float((recovered - world_pos).abs().max().item())
    if max_abs > float(atol):
        raise AssertionError(
            f"ANARESID {residual_frame} roundtrip failed: max_abs={max_abs:.3e} > {atol:.3e}"
        )
    return max_abs
