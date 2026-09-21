"""ANARESID training-noise contracts.

For normalized-componentwise noise the physical component std is calibrated
from the TRAIN-only residual-increment standard deviation, not the old velocity
standard deviation.
"""
from __future__ import annotations

from typing import Tuple
import torch

NOISE_NONE = "none"
NOISE_PHYSICAL_ISOTROPIC = "physical_isotropic"
NOISE_NORMALIZED_COMPONENTWISE = "normalized_componentwise"
VALID_NOISE_MODES = {
    NOISE_NONE,
    NOISE_PHYSICAL_ISOTROPIC,
    NOISE_NORMALIZED_COMPONENTWISE,
}


def canonical_noise_mode(value: object) -> str:
    mode = str(value).strip().lower()
    aliases = {
        "off": NOISE_NONE,
        "disabled": NOISE_NONE,
        "physical": NOISE_PHYSICAL_ISOTROPIC,
        "isotropic": NOISE_PHYSICAL_ISOTROPIC,
        "normalized": NOISE_NORMALIZED_COMPONENTWISE,
        "componentwise": NOISE_NORMALIZED_COMPONENTWISE,
        "norm_componentwise": NOISE_NORMALIZED_COMPONENTWISE,
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_NOISE_MODES:
        raise ValueError(
            f"Unsupported noise_mode={value!r}; expected one of {sorted(VALID_NOISE_MODES)}"
        )
    return mode


def validate_noise_config(noise_mode: object, noise_std: float, noise_alpha: float) -> str:
    mode = canonical_noise_mode(noise_mode)
    noise_std = float(noise_std)
    noise_alpha = float(noise_alpha)
    if noise_std < 0.0 or noise_alpha < 0.0:
        raise ValueError("noise_std/noise_alpha must be non-negative")
    if mode == NOISE_NONE:
        if noise_std != 0.0 or noise_alpha != 0.0:
            raise ValueError("noise_mode='none' requires noise_std=0 and noise_alpha=0")
    elif mode == NOISE_PHYSICAL_ISOTROPIC:
        if noise_alpha != 0.0:
            raise ValueError("physical_isotropic requires noise_alpha=0")
    elif mode == NOISE_NORMALIZED_COMPONENTWISE:
        if noise_std != 0.0:
            raise ValueError(
                "normalized_componentwise requires noise_std=0; use noise_alpha * residual_increment_std"
            )
    return mode


def component_noise_std_mm(
    residual_increment_std: torch.Tensor,
    noise_mode: object,
    noise_std: float,
    noise_alpha: float,
) -> torch.Tensor:
    """Return component noise std [3] in residual coordinates, physical mm."""
    mode = validate_noise_config(noise_mode, noise_std, noise_alpha)
    std = torch.as_tensor(residual_increment_std)
    if std.numel() != 3:
        raise ValueError(
            f"residual_increment_std must contain 3 components, got {tuple(std.shape)}"
        )
    std = std.reshape(3)
    if not torch.isfinite(std).all() or torch.any(std <= 0.0):
        raise ValueError(
            f"residual_increment_std must be finite positive, got {std.tolist()}"
        )
    if mode == NOISE_NONE:
        return torch.zeros_like(std)
    if mode == NOISE_PHYSICAL_ISOTROPIC:
        return torch.full_like(std, float(noise_std))
    return std * float(noise_alpha)


def sample_residual_noise(
    current_residual: torch.Tensor,
    free_tube_mask: torch.Tensor,
    residual_increment_std: torch.Tensor,
    noise_mode: object,
    noise_std: float,
    noise_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample online noise in the chosen residual coordinate system."""
    if current_residual.ndim != 2 or current_residual.shape[1] != 3:
        raise ValueError(
            f"current_residual must be [N,3], got {tuple(current_residual.shape)}"
        )
    mask = torch.as_tensor(
        free_tube_mask, device=current_residual.device
    ).reshape(-1).bool()
    if mask.numel() != current_residual.shape[0]:
        raise ValueError("free_tube_mask length mismatch")
    std_xyz = component_noise_std_mm(
        residual_increment_std=residual_increment_std,
        noise_mode=noise_mode,
        noise_std=noise_std,
        noise_alpha=noise_alpha,
    ).to(device=current_residual.device, dtype=current_residual.dtype)
    noise = torch.zeros_like(current_residual)
    if torch.any(mask) and torch.any(std_xyz > 0.0):
        noise[mask] = torch.randn_like(current_residual[mask]) * std_xyz.reshape(1, 3)
    return noise, std_xyz


# Compatibility alias for scripts that still import sample_position_noise.
def sample_position_noise(
    current: torch.Tensor,
    free_tube_mask: torch.Tensor,
    residual_increment_std: torch.Tensor,
    noise_mode: object,
    noise_std: float,
    noise_alpha: float,
):
    return sample_residual_noise(
        current_residual=current,
        free_tube_mask=free_tube_mask,
        residual_increment_std=residual_increment_std,
        noise_mode=noise_mode,
        noise_std=noise_std,
        noise_alpha=noise_alpha,
    )
