from __future__ import annotations

import argparse
import os
import torch
from omegaconf import OmegaConf

from model_factory import create_model, architecture_summary
from packed_dataset import load_normalization_stats
from target_space import load_target_stats
from trajectory_iterator import PackedTrajectoryWindowIterator
from unroll import unroll, validate_training_contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--steps', type=int, default=2)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    validate_training_contract(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    device = torch.device('cuda:0')

    iterator = PackedTrajectoryWindowIterator(
        cache_dir=str(cfg.preprocess_output_dir),
        stats_dir=str(cfg.stats_dir),
        target_stats_path=str(cfg.target_stats_path),
        prediction_mode=str(cfg.prediction_mode),
        num_time_steps=int(cfg.num_time_steps),
        unroll_steps=int(cfg.multistep_rollout_steps),
        window_stride=int(cfg.multistep_window_stride),
        seed=int(cfg.seed),
        shuffle_windows=False,
        expected_train_samples=int(cfg.expected_train_samples),
    )
    edge_stats = load_normalization_stats(str(cfg.stats_dir))
    target_stats = load_target_stats(str(cfg.target_stats_path), str(cfg.prediction_mode))
    model = create_model(cfg, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate))

    items = iterator.iter_epoch(0, 0, 1, device)
    for i in range(int(args.steps)):
        item = next(items)
        opt.zero_grad(set_to_none=True)
        result = unroll(
            model=model,
            packed=item['packed'],
            start_step=item['start_step'],
            edge_mean=edge_stats['edge_mean'],
            edge_std=edge_stats['edge_std'],
            target_mean=target_stats['mean'],
            target_std=target_stats['std'],
            device=device,
            world_edge_radius=float(cfg.world_edge_radius),
            feature_mode=str(cfg.feature_mode),
            prediction_mode=str(cfg.prediction_mode),
            theta_final_deg=float(cfg.theta_final_deg),
            loss_weights=[float(x) for x in cfg.multistep_loss_weights],
            noise_alpha=float(cfg.noise_alpha),
            training=True,
            amp_enabled=bool(cfg.amp),
            use_whole_model_checkpoint=bool(cfg.multistep_whole_model_checkpoint),
        )
        result.loss.backward()
        opt.step()
        print(
            f"step={i+1} sample={item['sample_id']:04d} start={item['start_step']} "
            f"loss={float(result.loss):.6e} free_mae={[float(x) for x in result.step_free_mae_mm]} "
            f"world_edges={result.step_world_edges}",
            flush=True,
        )

    print('architecture:', architecture_summary(cfg))
    print('peak_reserved_gib:', torch.cuda.max_memory_reserved(device) / 1024**3)
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
