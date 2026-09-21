"""Reference-configuration stationary-wave positional encoding for MGN-T.

The MGN-T paper states that positional encoding is a stationary-wave
representation over the undeformed spatial domain, with sinusoidal functions
along length, width, and height. The paper does not publish the exact frequency
schedule/dimensionality. This module therefore makes those details explicit and
configurable while preserving the stated design:

* source coordinates are graph.mesh_pos (undeformed/reference coordinates),
* coordinates are normalized per physical graph, never using current rollout x,
* sine/cosine stationary harmonics are evaluated independently on x/y/z,
* the PE is concatenated to the pre-MPNN node latent and learned-projects back
  to the Transformer latent dimension before physics-attention.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class ReferenceStationaryWavePE(nn.Module):
    """Sin/cos stationary-wave PE on a graph's undeformed bounding box.

    For each axis a in {x,y,z}, reference coordinate X_a is mapped to xi_a in
    [0, 1] using that physical graph's undeformed bounding box. For harmonic
    k=1..K we emit

        sin(k*pi*xi_a), cos(k*pi*xi_a).

    This is translation invariant and normalized with respect to the graph's
    reference spatial extent. A degenerate axis safely becomes a constant
    encoding on that axis.
    """

    def __init__(self, num_frequencies: int = 8, eps: float = 1.0e-8) -> None:
        super().__init__()
        if int(num_frequencies) <= 0:
            raise ValueError("num_frequencies must be positive")
        if float(eps) <= 0:
            raise ValueError("eps must be positive")
        self.num_frequencies = int(num_frequencies)
        self.eps = float(eps)
        harmonics = torch.arange(1, self.num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics * math.pi, persistent=False)
        self.output_dim = 3 * 2 * self.num_frequencies
        self.last_debug: Dict[str, object] = {}

    def forward(self, reference_pos: Tensor) -> Tensor:
        if reference_pos.ndim != 2 or reference_pos.shape[-1] != 3:
            raise ValueError(
                "reference_pos must have shape [N,3], got "
                f"{tuple(reference_pos.shape)}"
            )
        if reference_pos.shape[0] == 0:
            raise ValueError("reference_pos cannot be empty")
        if not torch.isfinite(reference_pos).all():
            raise ValueError("reference_pos contains NaN/Inf")

        # Normalize in fp32 for numerical robustness under BF16 autocast.
        ref = reference_pos.float()
        ref_min = ref.amin(dim=0, keepdim=True)
        ref_max = ref.amax(dim=0, keepdim=True)
        span = ref_max - ref_min
        safe_span = span.clamp_min(self.eps)
        xi = (ref - ref_min) / safe_span

        # [N,3,K]
        phase = xi.unsqueeze(-1) * self.harmonics.reshape(1, 1, -1)
        pe = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        pe = pe.reshape(reference_pos.shape[0], self.output_dim)

        # Constant coordinates on degenerate axes are intentional and stable.
        with torch.no_grad():
            self.last_debug = {
                "reference_nodes": int(reference_pos.shape[0]),
                "pe_dim": int(self.output_dim),
                "num_frequencies": int(self.num_frequencies),
                "normalization": "per_graph_reference_bbox_0_1",
                "reference_min": [float(v) for v in ref_min.reshape(-1).tolist()],
                "reference_max": [float(v) for v in ref_max.reshape(-1).tolist()],
                "degenerate_axes": [bool(v) for v in (span.reshape(-1) < self.eps).tolist()],
                "pe_min": float(pe.min().item()),
                "pe_max": float(pe.max().item()),
                "has_nan": int(torch.isnan(pe).any().item()),
                "has_inf": int(torch.isinf(pe).any().item()),
            }

        return pe

    def get_debug(self) -> Dict[str, object]:
        return dict(self.last_debug)
