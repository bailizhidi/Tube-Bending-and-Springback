from __future__ import annotations
import inspect
from typing import Dict,Any
import torch
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from mgn_t_model import HybridMeshGraphNetTransformer
from features import FEATURE_NAMES,canonical_feature_mode,num_input_features

def _mgn_kwargs(cfg):
    return dict(processor_size=int(cfg.processor_size),num_layers_node_processor=int(cfg.num_layers_node_processor),num_layers_edge_processor=int(cfg.num_layers_edge_processor),hidden_dim_processor=int(cfg.hidden_dim_processor),hidden_dim_node_encoder=int(cfg.hidden_dim_node_encoder),num_layers_node_encoder=int(cfg.num_layers_node_encoder),hidden_dim_edge_encoder=int(cfg.hidden_dim_edge_encoder),num_layers_edge_encoder=int(cfg.num_layers_edge_encoder),hidden_dim_node_decoder=int(cfg.hidden_dim_node_decoder),num_layers_node_decoder=int(cfg.num_layers_node_decoder),aggregation=str(cfg.aggregation),mlp_activation_fn=str(cfg.mlp_activation_fn),do_concat_trick=bool(cfg.do_concat_trick),num_processor_checkpoint_segments=int(cfg.num_processor_checkpoint_segments),checkpoint_offloading=bool(cfg.checkpoint_offloading),recompute_activation=bool(cfg.recompute_activation),norm_type=str(cfg.norm_type))
def _mgnt_kwargs(cfg):
    return dict(mgnt_num_pre_mp=int(cfg.mgnt_num_pre_mp),mgnt_num_post_mp=int(cfg.mgnt_num_post_mp),mgnt_num_transformer_blocks=int(cfg.mgnt_num_transformer_blocks),mgnt_num_heads=int(cfg.mgnt_num_heads),mgnt_num_tokens=int(cfg.mgnt_num_tokens),mgnt_transformer_dim=int(cfg.mgnt_transformer_dim),mgnt_transformer_mlp_ratio=int(cfg.mgnt_transformer_mlp_ratio),mgnt_transformer_dropout=float(cfg.mgnt_transformer_dropout),mgnt_transformer_activation=str(cfg.mgnt_transformer_activation),mgnt_use_transformer_engine=bool(cfg.mgnt_use_transformer_engine),mgnt_tokenizer_mode=str(cfg.mgnt_tokenizer_mode),mgnt_checkpoint_transformer=bool(cfg.mgnt_checkpoint_transformer),mgnt_gumbel_eval_stochastic=bool(cfg.mgnt_gumbel_eval_stochastic),mgnt_debug_tokenization=bool(cfg.mgnt_debug_tokenization),mgnt_dead_token_mass_threshold=float(cfg.mgnt_dead_token_mass_threshold),mgnt_use_positional_encoding=bool(cfg.mgnt_use_positional_encoding),mgnt_pe_num_frequencies=int(cfg.mgnt_pe_num_frequencies))
def validate_config(cfg):
    fm=canonical_feature_mode(cfg.feature_mode)
    if int(cfg.num_input_features)!=num_input_features(fm): raise ValueError(f"num_input_features={cfg.num_input_features}, expected {num_input_features(fm)} for {fm}")
    if tuple(str(x) for x in cfg.node_feature_names)!=FEATURE_NAMES[fm]: raise ValueError("node_feature_names mismatch")
    if int(cfg.num_edge_features)!=8 or int(cfg.num_output_features)!=3: raise ValueError("expected edge=8 output=3")
    p=str(cfg.processor_type).lower()
    if p not in ('mgn','mgn_t'): raise ValueError(p)
    if p=='mgn':
        if int(cfg.hidden_dim_processor)!=80: raise ValueError("formal MGN baseline must use hidden_dim=80 due to memory contract")
    else:
        if int(cfg.mgnt_transformer_dim)!=int(cfg.hidden_dim_processor): raise ValueError("MGN-T transformer_dim must equal hidden_dim_processor")
        if int(cfg.mgnt_transformer_dim)%int(cfg.mgnt_num_heads): raise ValueError("transformer dim not divisible by heads")
def create_model(cfg,device):
    validate_config(cfg); common=dict(input_dim_nodes=int(cfg.num_input_features),input_dim_edges=int(cfg.num_edge_features),output_dim=int(cfg.num_output_features))
    if str(cfg.processor_type).lower()=='mgn':
        kw=_mgn_kwargs(cfg); model=HybridMeshGraphNet(common['input_dim_nodes'],common['input_dim_edges'],common['output_dim'],**kw)
    else:
        model=HybridMeshGraphNetTransformer(input_dim_nodes=common['input_dim_nodes'],input_dim_edges=common['input_dim_edges'],output_dim=common['output_dim'],num_layers_node_processor=int(cfg.num_layers_node_processor),num_layers_edge_processor=int(cfg.num_layers_edge_processor),hidden_dim_processor=int(cfg.hidden_dim_processor),hidden_dim_node_encoder=int(cfg.hidden_dim_node_encoder),num_layers_node_encoder=int(cfg.num_layers_node_encoder),hidden_dim_edge_encoder=int(cfg.hidden_dim_edge_encoder),num_layers_edge_encoder=int(cfg.num_layers_edge_encoder),hidden_dim_node_decoder=int(cfg.hidden_dim_node_decoder),num_layers_node_decoder=int(cfg.num_layers_node_decoder),aggregation=str(cfg.aggregation),mlp_activation_fn=str(cfg.mlp_activation_fn),do_concat_trick=bool(cfg.do_concat_trick),num_processor_checkpoint_segments=int(cfg.num_processor_checkpoint_segments),checkpoint_offloading=bool(cfg.checkpoint_offloading),recompute_activation=bool(cfg.recompute_activation),norm_type=str(cfg.norm_type),**_mgnt_kwargs(cfg))
    return model.to(device)
def architecture_summary(cfg)->Dict[str,Any]:
    return {'processor_type':str(cfg.processor_type),'hidden_dim':int(cfg.hidden_dim_processor),'processor_size':int(cfg.processor_size),'feature_mode':str(cfg.feature_mode),'num_input_features':int(cfg.num_input_features),'prediction_mode':str(cfg.prediction_mode),'unroll_steps':int(cfg.multistep_rollout_steps)}
def count_trainable_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
