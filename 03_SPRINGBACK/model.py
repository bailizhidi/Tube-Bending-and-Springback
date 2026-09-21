# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    last = in_dim
    for _ in range(max(1, num_layers - 1)):
        layers.append(nn.Linear(last, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class EdgeBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_layers: int, dropout: float):
        super().__init__()
        self.mlp = make_mlp(hidden_dim * 3, hidden_dim, hidden_dim, mlp_layers, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_h: torch.Tensor, edge_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        inp = torch.cat([edge_h, node_h[src], node_h[dst]], dim=-1)
        update = self.mlp(inp)
        return self.norm(edge_h + update)


class NodeBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_layers: int, dropout: float):
        super().__init__()
        self.mlp = make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, mlp_layers, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_h: torch.Tensor, edge_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        _, dst = edge_index
        agg = torch.zeros_like(node_h)
        agg.index_add_(0, dst, edge_h)
        deg = torch.zeros((node_h.shape[0], 1), dtype=node_h.dtype, device=node_h.device)
        ones = torch.ones((edge_h.shape[0], 1), dtype=node_h.dtype, device=node_h.device)
        deg.index_add_(0, dst, ones)
        agg = agg / deg.clamp_min(1.0)
        update = self.mlp(torch.cat([node_h, agg], dim=-1))
        return self.norm(node_h + update)


class ProcessorBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_layers: int, dropout: float):
        super().__init__()
        self.edge_block = EdgeBlock(hidden_dim, mlp_layers, dropout)
        self.node_block = NodeBlock(hidden_dim, mlp_layers, dropout)

    def forward(self, node_h: torch.Tensor, edge_h: torch.Tensor, edge_index: torch.Tensor):
        edge_h = self.edge_block(node_h, edge_h, edge_index)
        node_h = self.node_block(node_h, edge_h, edge_index)
        return node_h, edge_h


class MeshGraphNetOneStep(nn.Module):
    """MGN-style Encoder-Processor-Decoder for one-step dU_springback prediction."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int = 3,
        hidden_dim: int = 128,
        num_message_passing_steps: int = 12,
        mlp_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            make_mlp(node_in_dim, hidden_dim, hidden_dim, mlp_layers, dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            make_mlp(edge_in_dim, hidden_dim, hidden_dim, mlp_layers, dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.processor = nn.ModuleList([
            ProcessorBlock(hidden_dim, mlp_layers, dropout) for _ in range(num_message_passing_steps)
        ])
        self.decoder = make_mlp(hidden_dim, out_dim, hidden_dim, mlp_layers, dropout)

    def forward(self, data):
        node_h = self.node_encoder(data.x)
        edge_h = self.edge_encoder(data.edge_attr)
        for block in self.processor:
            node_h, edge_h = block(node_h, edge_h, data.edge_index)
        return self.decoder(node_h)
