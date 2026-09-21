"""Hybrid MeshGraphNet-Transformer model for GE-BendMGN B2.

The encoder and decoder are inherited unchanged from PhysicsNeMo's
HybridMeshGraphNet.  Only ``self.processor`` is replaced by the
Local-Global-Local MGN-T processor.
"""

from __future__ import annotations

from typing import Literal

from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.nn import get_activation

from mgn_t_processor import HybridMGNTProcessor


class HybridMeshGraphNetTransformer(HybridMeshGraphNet):
    """Minimal-intrusion MGN-T extension of ``HybridMeshGraphNet``."""

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        *,
        mlp_activation_fn: str,
        num_layers_node_processor: int,
        num_layers_edge_processor: int,
        hidden_dim_processor: int,
        hidden_dim_node_encoder: int,
        num_layers_node_encoder: int,
        hidden_dim_edge_encoder: int,
        num_layers_edge_encoder: int,
        hidden_dim_node_decoder: int,
        num_layers_node_decoder: int,
        aggregation: Literal["sum", "mean"],
        do_concat_trick: bool,
        num_processor_checkpoint_segments: int,
        checkpoint_offloading: bool,
        recompute_activation: bool,
        norm_type: Literal["LayerNorm", "TELayerNorm"],
        mgnt_num_pre_mp: int,
        mgnt_num_post_mp: int,
        mgnt_num_transformer_blocks: int,
        mgnt_num_heads: int,
        mgnt_num_tokens: int,
        mgnt_transformer_dim: int,
        mgnt_transformer_mlp_ratio: int,
        mgnt_transformer_dropout: float,
        mgnt_transformer_activation: str,
        mgnt_use_transformer_engine: bool,
        mgnt_tokenizer_mode: str,
        mgnt_checkpoint_transformer: bool,
        mgnt_gumbel_eval_stochastic: bool,
        mgnt_debug_tokenization: bool,
        mgnt_dead_token_mass_threshold: float,
        mgnt_use_positional_encoding: bool,
        mgnt_pe_num_frequencies: int,
    ) -> None:
        total_local_mp = int(mgnt_num_pre_mp) + int(mgnt_num_post_mp)

        # This creates exactly the same node/mesh-edge/world-edge encoders and
        # decoder as the original HybridMeshGraphNet.  The temporary local
        # processor is immediately replaced below.
        super().__init__(
            input_dim_nodes=int(input_dim_nodes),
            input_dim_edges=int(input_dim_edges),
            output_dim=int(output_dim),
            processor_size=total_local_mp,
            mlp_activation_fn=str(mlp_activation_fn),
            num_layers_node_processor=int(num_layers_node_processor),
            num_layers_edge_processor=int(num_layers_edge_processor),
            hidden_dim_processor=int(hidden_dim_processor),
            hidden_dim_node_encoder=int(hidden_dim_node_encoder),
            num_layers_node_encoder=int(num_layers_node_encoder),
            hidden_dim_edge_encoder=int(hidden_dim_edge_encoder),
            num_layers_edge_encoder=int(num_layers_edge_encoder),
            hidden_dim_node_decoder=int(hidden_dim_node_decoder),
            num_layers_node_decoder=int(num_layers_node_decoder),
            aggregation=aggregation,
            do_concat_trick=bool(do_concat_trick),
            # The MGN-T processor owns checkpoint placement because the global
            # stage interrupts the local stack.
            num_processor_checkpoint_segments=0,
            checkpoint_offloading=False,
            recompute_activation=bool(recompute_activation),
            norm_type=norm_type,
        )

        activation_fn = get_activation(str(mlp_activation_fn))
        self.processor = HybridMGNTProcessor(
            num_pre_mp=int(mgnt_num_pre_mp),
            num_post_mp=int(mgnt_num_post_mp),
            input_dim_node=int(hidden_dim_processor),
            input_dim_edge=int(hidden_dim_processor),
            num_layers_node=int(num_layers_node_processor),
            num_layers_edge=int(num_layers_edge_processor),
            aggregation=str(aggregation),
            norm_type=str(norm_type),
            activation_fn=activation_fn,
            do_concat_trick=bool(do_concat_trick),
            num_processor_checkpoint_segments=int(num_processor_checkpoint_segments),
            checkpoint_offloading=bool(checkpoint_offloading),
            num_transformer_blocks=int(mgnt_num_transformer_blocks),
            num_heads=int(mgnt_num_heads),
            num_tokens=int(mgnt_num_tokens),
            transformer_dim=int(mgnt_transformer_dim),
            transformer_mlp_ratio=int(mgnt_transformer_mlp_ratio),
            transformer_dropout=float(mgnt_transformer_dropout),
            transformer_activation=str(mgnt_transformer_activation),
            use_transformer_engine=bool(mgnt_use_transformer_engine),
            tokenizer_mode=str(mgnt_tokenizer_mode),
            checkpoint_transformer=bool(mgnt_checkpoint_transformer),
            gumbel_eval_stochastic=bool(mgnt_gumbel_eval_stochastic),
            debug_tokenization=bool(mgnt_debug_tokenization),
            dead_token_mass_threshold=float(mgnt_dead_token_mass_threshold),
            use_positional_encoding=bool(mgnt_use_positional_encoding),
            pe_num_frequencies=int(mgnt_pe_num_frequencies),
        )
        self.processor_type = "mgn_t"

    def get_token_debug(self):
        return self.processor.get_token_debug()

    def get_pe_debug(self):
        return self.processor.get_pe_debug()
