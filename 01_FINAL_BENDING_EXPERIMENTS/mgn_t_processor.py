"""Local-Global-Local processor for the controlled MGN-T reproduction.

Design constraints
------------------
* Local updates reuse PhysicsNeMo's exact HybridMeshGraphNet processor layers.
* Global updates use PhysicsNeMo Transolver physics-attention on learned tokens.
* Transformer attention is P x P, never N x N.
* Different graphs in a PyG batch are processed independently in the global stage.
* Mesh/world edge latents updated by the pre-MPNN are preserved and handed to
  the post-MPNN refinement stage.
* Optional paper-like reference positional encoding is injected before the global stage.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from physicsnemo.models.meshgraphnet.hybrid_meshgraphnet import (
    HybridMeshGraphNetProcessor,
)
from physicsnemo.models.transolver.transolver import TransolverBlock
from physicsnemo.nn.module.gumbel_softmax import gumbel_softmax
from physicsnemo.nn.module.physics_attention import PhysicsAttentionIrregularMesh
from physicsnemo.nn.module.gnn_layers.utils import GraphType

from reference_positional_encoding import ReferenceStationaryWavePE


_TOKENIZER_MODES = {"softmax", "adaptive_softmax", "adaptive_gumbel"}


class DiagnosticPhysicsAttentionIrregularMesh(PhysicsAttentionIrregularMesh):
    """Physics-attention with configurable slicing and scalar diagnostics.

    ``softmax`` follows PhysicsNeMo Transolver slicing (global learned
    temperature). ``adaptive_gumbel`` follows the supplied Transolver++ path
    (adaptive local temperature + Gumbel-Softmax). ``adaptive_softmax`` is an
    explicit diagnostic implementation choice that retains adaptive temperature
    while removing Gumbel noise.

    The module never retains the full assignment matrix for diagnostics.  Only
    detached scalar summaries are stored in ``last_debug``.
    """

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        slice_num: int,
        use_te: bool,
        tokenizer_mode: str,
        gumbel_eval_stochastic: bool,
        debug_tokenization: bool,
        dead_token_mass_threshold: float,
    ) -> None:
        mode = str(tokenizer_mode).strip().lower()
        if mode not in _TOKENIZER_MODES:
            raise ValueError(
                f"Unknown tokenizer_mode={tokenizer_mode!r}; "
                f"expected one of {sorted(_TOKENIZER_MODES)}"
            )
        plus = mode in {"adaptive_softmax", "adaptive_gumbel"}
        super().__init__(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            slice_num=slice_num,
            use_te=use_te,
            plus=plus,
        )
        self.tokenizer_mode = mode
        self.gumbel_eval_stochastic = bool(gumbel_eval_stochastic)
        self.debug_tokenization = bool(debug_tokenization)
        self.dead_token_mass_threshold = float(dead_token_mass_threshold)
        self.slice_num = int(slice_num)
        self.last_debug: Dict[str, float | int | str | list] = {}

    def _record_debug(
        self,
        slice_weights: Tensor,
        effective_temperature: Tensor,
    ) -> None:
        if not self.debug_tokenization:
            self.last_debug = {}
            return

        with torch.no_grad():
            weights = slice_weights.detach()
            # [B, H, P].  dtype=fp32 improves the mass reduction without
            # creating an fp32 copy of the full [B,N,H,P] tensor.
            token_mass = weights.sum(dim=1, dtype=torch.float32)
            dead = token_mass < self.dead_token_mass_threshold

            # Entropy is diagnostic only. Keep it in the attention dtype to
            # avoid a second full-size fp32 assignment tensor.
            tiny = 1.0e-7
            probs = weights.clamp_min(tiny)
            entropy = -(weights * probs.log()).sum(dim=-1)
            entropy_mean = entropy.float().mean()
            entropy_norm = entropy_mean / max(math.log(float(self.slice_num)), 1.0e-12)
            max_prob_mean = weights.max(dim=-1).values.float().mean()

            temp = effective_temperature.detach().float()
            self.last_debug = {
                "tokenizer_mode": self.tokenizer_mode,
                "assignment_shape": list(slice_weights.shape),
                "token_mass_min": float(token_mass.min().item()),
                "token_mass_mean": float(token_mass.mean().item()),
                "token_mass_max": float(token_mass.max().item()),
                "dead_token_count": int(dead.sum().item()),
                "dead_token_fraction": float(dead.float().mean().item()),
                "assignment_entropy_mean": float(entropy_mean.item()),
                "assignment_entropy_normalized": float(entropy_norm.item()),
                "assignment_max_probability_mean": float(max_prob_mean.item()),
                "temperature_min": float(temp.min().item()),
                "temperature_mean": float(temp.mean().item()),
                "temperature_max": float(temp.max().item()),
                "has_nan": int(torch.isnan(weights).any().item()),
                "has_inf": int(torch.isinf(weights).any().item()),
            }

    def _compute_slices_from_projections(
        self,
        slice_projections: Tensor,
        fx: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the actual assignments used by the forward pass."""
        if self.tokenizer_mode == "softmax":
            effective_temperature = torch.clamp(
                self.temperature, min=0.5, max=5.0
            ).to(slice_projections.dtype)
            slice_weights = torch.nn.functional.softmax(
                slice_projections / effective_temperature,
                dim=-1,
            )
        else:
            # plus=True guarantees proj_temperature exists in the parent.
            effective_temperature = self.temperature + self.proj_temperature(fx)
            effective_temperature = torch.clamp(
                effective_temperature, min=0.01
            ).to(slice_projections.dtype)

            use_gumbel = (
                self.tokenizer_mode == "adaptive_gumbel"
                and (self.training or self.gumbel_eval_stochastic)
            )
            if use_gumbel:
                slice_weights = gumbel_softmax(
                    slice_projections,
                    effective_temperature,
                )
            else:
                slice_weights = torch.nn.functional.softmax(
                    slice_projections / effective_temperature,
                    dim=-1,
                )

        slice_weights = slice_weights.to(slice_projections.dtype)
        self._record_debug(slice_weights, effective_temperature)

        # Match PhysicsNeMo's stable normalization: normalize assignments first
        # and only then perform the weighted aggregation.
        slice_norm = slice_weights.sum(1) + 1.0e-2  # [B,H,P]
        normed_weights = slice_weights / slice_norm[:, None, :, :]
        slice_token = torch.matmul(
            normed_weights.permute(0, 2, 3, 1),
            fx.permute(0, 2, 1, 3),
        )
        return slice_weights, slice_token


class MGNTTransformerBlock(TransolverBlock):
    """PhysicsNeMo Transolver block with inspectable/configurable tokenization."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_tokens: int,
        dropout: float,
        activation: str,
        mlp_ratio: int,
        use_te: bool,
        tokenizer_mode: str,
        gumbel_eval_stochastic: bool,
        debug_tokenization: bool,
        dead_token_mass_threshold: float,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")

        plus = str(tokenizer_mode).lower() != "softmax"
        super().__init__(
            num_heads=int(num_heads),
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            act=str(activation),
            mlp_ratio=int(mlp_ratio),
            last_layer=False,
            slice_num=int(num_tokens),
            spatial_shape=None,
            use_te=bool(use_te),
            plus=plus,
        )

        # Replace only the attention object.  LayerNorm, FFN, residual layout
        # and the rest of TransolverBlock remain exactly PhysicsNeMo's.
        self.Attn = DiagnosticPhysicsAttentionIrregularMesh(
            dim=int(hidden_dim),
            heads=int(num_heads),
            dim_head=int(hidden_dim) // int(num_heads),
            dropout=float(dropout),
            slice_num=int(num_tokens),
            use_te=bool(use_te),
            tokenizer_mode=str(tokenizer_mode),
            gumbel_eval_stochastic=bool(gumbel_eval_stochastic),
            debug_tokenization=bool(debug_tokenization),
            dead_token_mass_threshold=float(dead_token_mass_threshold),
        )

    def get_token_debug(self) -> Dict:
        return dict(self.Attn.last_debug)


class HybridMGNTProcessor(HybridMeshGraphNetProcessor):
    """2-local -> token Transformer -> 2-local hybrid processor.

    The parent constructor is intentionally used to create the local hybrid
    MeshGraphNet blocks.  This guarantees the same HybridMeshEdgeBlock and
    HybridMeshNodeBlock implementation, aggregation, residual paths and MLP
    structure as the original B2 baseline.
    """

    def __init__(
        self,
        *,
        num_pre_mp: int,
        num_post_mp: int,
        input_dim_node: int,
        input_dim_edge: int,
        num_layers_node: int,
        num_layers_edge: int,
        aggregation: str,
        norm_type: str,
        activation_fn: nn.Module,
        do_concat_trick: bool,
        num_processor_checkpoint_segments: int,
        checkpoint_offloading: bool,
        num_transformer_blocks: int,
        num_heads: int,
        num_tokens: int,
        transformer_dim: int,
        transformer_mlp_ratio: int,
        transformer_dropout: float,
        transformer_activation: str,
        use_transformer_engine: bool,
        tokenizer_mode: str,
        checkpoint_transformer: bool,
        gumbel_eval_stochastic: bool,
        debug_tokenization: bool,
        dead_token_mass_threshold: float,
        use_positional_encoding: bool,
        pe_num_frequencies: int,
    ) -> None:
        self.num_pre_mp = int(num_pre_mp)
        self.num_post_mp = int(num_post_mp)
        if self.num_pre_mp <= 0 or self.num_post_mp <= 0:
            raise ValueError(
                "MGN-T requires positive pre/post local MP counts, got "
                f"{self.num_pre_mp}/{self.num_post_mp}"
            )
        if int(transformer_dim) != int(input_dim_node):
            raise ValueError(
                "MGN-T requires transformer_dim == "
                f"hidden_dim_processor, got {transformer_dim} != {input_dim_node}"
            )
        if int(num_transformer_blocks) <= 0:
            raise ValueError("num_transformer_blocks must be positive")

        total_local_mp = self.num_pre_mp + self.num_post_mp

        # Build the exact local blocks through the PhysicsNeMo parent.  Parent
        # checkpointing is disabled here because a global stage splits the local
        # stack; we reproduce checkpointing separately on each local stage.
        super().__init__(
            processor_size=total_local_mp,
            input_dim_node=int(input_dim_node),
            input_dim_edge=int(input_dim_edge),
            num_layers_node=int(num_layers_node),
            num_layers_edge=int(num_layers_edge),
            aggregation=str(aggregation),
            norm_type=str(norm_type),
            activation_fn=activation_fn,
            do_concat_trick=bool(do_concat_trick),
            num_processor_checkpoint_segments=0,
            checkpoint_offloading=False,
        )

        self.pre_layer_end = 2 * self.num_pre_mp
        self.post_layer_start = self.pre_layer_end
        self.local_checkpoint_segments = int(num_processor_checkpoint_segments)
        if self.local_checkpoint_segments < 0:
            raise ValueError("num_processor_checkpoint_segments cannot be negative")

        # Interpret the same value per local stage. With the formal value 1,
        # each complete pre/post local stage is one checkpoint segment.
        for name, length in (
            ("pre", self.pre_layer_end),
            ("post", self.num_processor_layers - self.post_layer_start),
        ):
            if (
                self.local_checkpoint_segments > 0
                and length % self.local_checkpoint_segments != 0
            ):
                raise ValueError(
                    f"{name} local layer count {length} must be divisible by "
                    f"num_processor_checkpoint_segments={self.local_checkpoint_segments}"
                )

        self.local_checkpointing = self.local_checkpoint_segments > 0
        self.checkpoint_transformer = bool(checkpoint_transformer)
        self.checkpoint_offloading = bool(checkpoint_offloading) and self.local_checkpointing
        self.checkpoint_offload_ctx = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if self.checkpoint_offloading
            else nullcontext()
        )

        self.global_blocks = nn.ModuleList(
            [
                MGNTTransformerBlock(
                    hidden_dim=int(transformer_dim),
                    num_heads=int(num_heads),
                    num_tokens=int(num_tokens),
                    dropout=float(transformer_dropout),
                    activation=str(transformer_activation),
                    mlp_ratio=int(transformer_mlp_ratio),
                    use_te=bool(use_transformer_engine),
                    tokenizer_mode=str(tokenizer_mode),
                    gumbel_eval_stochastic=bool(gumbel_eval_stochastic),
                    debug_tokenization=bool(debug_tokenization),
                    dead_token_mass_threshold=float(dead_token_mass_threshold),
                )
                for _ in range(int(num_transformer_blocks))
            ]
        )

        self.use_positional_encoding = bool(use_positional_encoding)
        self.pe_num_frequencies = int(pe_num_frequencies)
        if self.use_positional_encoding:
            self.reference_pe = ReferenceStationaryWavePE(
                num_frequencies=self.pe_num_frequencies
            )
            self.global_input_projection = nn.Linear(
                int(input_dim_node) + int(self.reference_pe.output_dim),
                int(transformer_dim),
                bias=True,
            )
            nn.init.xavier_uniform_(self.global_input_projection.weight)
            nn.init.zeros_(self.global_input_projection.bias)
        else:
            self.reference_pe = None
            self.global_input_projection = None

        self.hidden_dim = int(transformer_dim)
        self.num_heads = int(num_heads)
        self.num_tokens = int(num_tokens)
        self.tokenizer_mode = str(tokenizer_mode).lower()
        self.last_global_graph_sizes: List[int] = []

    @staticmethod
    def _split_range(start: int, end: int, segments: int) -> List[Tuple[int, int]]:
        if start >= end:
            return []
        if segments <= 0:
            return [(start, end)]
        length = end - start
        if length % segments != 0:
            raise ValueError(f"Layer range length {length} not divisible by {segments}")
        width = length // segments
        return [(i, i + width) for i in range(start, end, width)]

    def _run_local_range(
        self,
        start: int,
        end: int,
        node_features: Tensor,
        mesh_edge_features: Tensor,
        world_edge_features: Tensor,
        graph: GraphType,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ranges = self._split_range(start, end, self.local_checkpoint_segments)
        for seg_start, seg_end in ranges:
            fn = self._run_function(seg_start, seg_end)
            if self.local_checkpointing and self.training:
                mesh_edge_features, world_edge_features, node_features = checkpoint(
                    fn,
                    node_features,
                    mesh_edge_features,
                    world_edge_features,
                    graph,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                mesh_edge_features, world_edge_features, node_features = fn(
                    node_features,
                    mesh_edge_features,
                    world_edge_features,
                    graph,
                )
        return mesh_edge_features, world_edge_features, node_features

    @staticmethod
    def _node_ranges(graph: GraphType, num_nodes: int) -> List[Tuple[int, int]]:
        """Return contiguous node ranges, one per physical graph.

        Current formal training has one graph per GPU, so this is a single
        range.  For a correctly constructed PyG batch, ``ptr`` is preferred.
        A ``batch`` vector is accepted as a fallback.  No padding is used and
        separate graphs never enter the same attention operation.
        """
        ptr = getattr(graph, "ptr", None)
        if ptr is not None and torch.is_tensor(ptr) and ptr.numel() >= 2:
            values = [int(v) for v in ptr.detach().cpu().tolist()]
            if values[0] != 0 or values[-1] != int(num_nodes):
                raise ValueError(
                    f"Invalid graph.ptr={values[:4]}... for num_nodes={num_nodes}"
                )
            return list(zip(values[:-1], values[1:]))

        batch = getattr(graph, "batch", None)
        if batch is None or not torch.is_tensor(batch) or batch.numel() == 0:
            return [(0, int(num_nodes))]
        if batch.numel() != int(num_nodes):
            raise ValueError(
                f"graph.batch has {batch.numel()} entries for {num_nodes} nodes"
            )

        b = batch.detach()
        if b.numel() > 1 and torch.any(b[1:] < b[:-1]):
            raise ValueError("graph.batch must group graph nodes contiguously")
        changes = torch.nonzero(b[1:] != b[:-1], as_tuple=False).reshape(-1) + 1
        boundaries = [0] + [int(v) for v in changes.cpu().tolist()] + [int(num_nodes)]
        return list(zip(boundaries[:-1], boundaries[1:]))

    @staticmethod
    def _reference_positions(graph: GraphType, num_nodes: int) -> Tensor:
        reference_pos = getattr(graph, "mesh_pos", None)
        if reference_pos is None or not torch.is_tensor(reference_pos):
            raise ValueError(
                "MGN-T positional encoding requires graph.mesh_pos containing "
                "undeformed/reference coordinates [N,3]."
            )
        if reference_pos.ndim != 2 or tuple(reference_pos.shape) != (int(num_nodes), 3):
            raise ValueError(
                "graph.mesh_pos must have shape "
                f"[{num_nodes},3], got {tuple(reference_pos.shape)}"
            )
        return reference_pos

    def _prepare_global_input(
        self,
        node_features: Tensor,
        reference_pos: Tensor,
    ) -> Tensor:
        if not self.use_positional_encoding:
            return node_features
        assert self.reference_pe is not None
        assert self.global_input_projection is not None
        pe = self.reference_pe(reference_pos).to(
            device=node_features.device, dtype=node_features.dtype
        )
        x = torch.cat((node_features, pe), dim=-1)
        return self.global_input_projection(x)

    def _apply_global_block(self, block: nn.Module, x: Tensor) -> Tensor:
        if self.checkpoint_transformer and self.training:
            # preserve_rng_state=True is mandatory for adaptive_gumbel:
            # checkpoint backward recomputation must reuse forward Gumbel noise.
            return checkpoint(
                block,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return block(x)

    def _global_update(self, node_features: Tensor, graph: GraphType) -> Tensor:
        ranges = self._node_ranges(graph, node_features.shape[0])
        self.last_global_graph_sizes = [end - start for start, end in ranges]
        reference_pos_all = (
            self._reference_positions(graph, node_features.shape[0])
            if self.use_positional_encoding
            else None
        )

        outputs: List[Tensor] = []
        for start, end in ranges:
            if end <= start:
                raise ValueError("Empty graph found in MGN-T batch")
            x_nodes = node_features[start:end]
            if reference_pos_all is not None:
                x_nodes = self._prepare_global_input(
                    x_nodes, reference_pos_all[start:end]
                )
            # PhysicsAttention expects [B,N,C]. Here B=1 per physical graph,
            # which naturally supports different node counts without padding.
            x = x_nodes.unsqueeze(0)
            for block in self.global_blocks:
                x = self._apply_global_block(block, x)
            outputs.append(x.squeeze(0))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        node_features: Tensor,
        mesh_edge_features: Tensor,
        world_edge_features: Tensor,
        graph: GraphType,
    ) -> Tensor:
        if node_features.ndim != 2 or node_features.shape[1] != self.input_dim_node:
            raise ValueError(
                f"Expected node latent [N,{self.input_dim_node}], got "
                f"{tuple(node_features.shape)}"
            )
        if mesh_edge_features.ndim != 2 or mesh_edge_features.shape[1] != self.input_dim_edge:
            raise ValueError(
                f"Expected mesh edge latent [E,{self.input_dim_edge}], got "
                f"{tuple(mesh_edge_features.shape)}"
            )
        if world_edge_features.ndim != 2 or world_edge_features.shape[1] != self.input_dim_edge:
            raise ValueError(
                f"Expected world edge latent [E,{self.input_dim_edge}], got "
                f"{tuple(world_edge_features.shape)}"
            )

        with self.checkpoint_offload_ctx:
            mesh_edge_features, world_edge_features, node_features = self._run_local_range(
                0,
                self.pre_layer_end,
                node_features,
                mesh_edge_features,
                world_edge_features,
                graph,
            )

        # Global update touches only node latents.  The locally updated edge
        # latents remain alive and continue into refinement unchanged.
        node_features = self._global_update(node_features, graph)

        with self.checkpoint_offload_ctx:
            mesh_edge_features, world_edge_features, node_features = self._run_local_range(
                self.post_layer_start,
                self.num_processor_layers,
                node_features,
                mesh_edge_features,
                world_edge_features,
                graph,
            )

        return node_features

    def get_token_debug(self) -> List[Dict]:
        result: List[Dict] = []
        for index, block in enumerate(self.global_blocks):
            stats = block.get_token_debug()
            stats["block"] = int(index)
            result.append(stats)
        return result

    def get_pe_debug(self) -> Dict:
        if not self.use_positional_encoding or self.reference_pe is None:
            return {"enabled": False}
        result = self.reference_pe.get_debug()
        result["enabled"] = True
        result["projection_in_dim"] = int(self.global_input_projection.in_features)
        result["projection_out_dim"] = int(self.global_input_projection.out_features)
        return result
