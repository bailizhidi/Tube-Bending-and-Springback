from __future__ import annotations

import json
import math
import os
import time

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from model_factory import architecture_summary, count_trainable_parameters, create_model
from packed_dataset import load_normalization_stats
from rollout import evaluate_manifest, unwrap_model
from target_space import load_target_stats
from training_utils import (
    SummaryWriter,
    append_csv,
    atomic_save,
    barrier,
    cleanup,
    make_optimizer,
    make_scheduler,
    reduce_max,
    reduce_sum,
    set_seed,
    setup_distributed,
)
from trajectory_iterator import PackedTrajectoryWindowIterator
from unroll import unroll, validate_training_contract

CHECKPOINT_VERSION = 1


def checkpoint_state(cfg, model, optimizer, scheduler, epoch, global_step,
                     best_val_metric, world_size, data_fingerprint):
    return {
        "format_version": CHECKPOINT_VERSION,
        "data_fingerprint": str(data_fingerprint),
        "training_contract": str(cfg.training_contract),
        "experiment_id": str(cfg.experiment_id),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_metric": float(best_val_metric),
        "world_size": int(world_size),
        "architecture": architecture_summary(cfg),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="exp04_local_anaresid_x10_2step",
)
def main(cfg: DictConfig) -> None:
    rank, world_size, local_rank, device = setup_distributed()
    try:
        validate_training_contract(cfg)
        set_seed(int(cfg.seed), rank)
        if world_size != 4:
            raise ValueError(f"formal training requires 4 GPUs, got {world_size}")
        if int(cfg.batch_size_per_gpu) != 1:
            raise ValueError("batch_size_per_gpu must be 1")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        cache_dir = to_absolute_path(cfg.preprocess_output_dir)
        stats_dir = to_absolute_path(cfg.stats_dir)
        target_stats_path = to_absolute_path(cfg.target_stats_path)

        iterator = PackedTrajectoryWindowIterator(
            cache_dir=cache_dir,
            stats_dir=stats_dir,
            target_stats_path=target_stats_path,
            prediction_mode=str(cfg.prediction_mode),
            num_time_steps=int(cfg.num_time_steps),
            unroll_steps=int(cfg.multistep_rollout_steps),
            window_stride=int(cfg.multistep_window_stride),
            seed=int(cfg.seed),
            shuffle_windows=bool(cfg.shuffle_timesteps_within_sample),
            expected_train_samples=int(cfg.expected_train_samples),
        )
        edge_stats = load_normalization_stats(stats_dir)
        target_stats = load_target_stats(target_stats_path, str(cfg.prediction_mode))
        data_fingerprint = str(target_stats["raw"].get("data_fingerprint", ""))
        if not data_fingerprint:
            raise RuntimeError(
                "target stats missing data_fingerprint; recompute with compute_final_stats.py"
            )

        steps_per_full_epoch = iterator.steps_per_rank(world_size)
        target_steps = int(cfg.target_optimizer_steps_per_rank)
        if target_steps <= 0:
            raise ValueError("target_optimizer_steps_per_rank must be positive")

        base_model = create_model(cfg, device)
        trainable_parameters = count_trainable_parameters(base_model)
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        optimizer = make_optimizer(cfg, model)
        scheduler, _ = make_scheduler(cfg, optimizer, target_steps)

        last_checkpoint = to_absolute_path(cfg.last_checkpoint)
        best_checkpoint = to_absolute_path(cfg.best_checkpoint)
        metrics_csv = to_absolute_path(cfg.training_metrics_csv)
        os.makedirs(os.path.dirname(last_checkpoint), exist_ok=True)

        start_epoch = 0
        global_step = 0
        best_val_metric = float("inf")
        if os.path.isfile(last_checkpoint) and bool(cfg.resume_if_exists):
            checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
            if checkpoint.get("experiment_id") != str(cfg.experiment_id):
                raise RuntimeError("resume experiment mismatch")
            if str(checkpoint.get("data_fingerprint", "")) != data_fingerprint:
                raise RuntimeError(
                    "resume checkpoint belongs to a different frozen-data fingerprint; "
                    "delete the old checkpoint before formal training"
                )
            unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"])
            global_step = int(checkpoint["global_step"])
            best_val_metric = float(checkpoint["best_val_metric"])

        if rank == 0:
            print("=" * 100)
            print("FINAL BENDING FORMAL EXPERIMENT")
            print(json.dumps(architecture_summary(cfg), indent=2))
            print("parameters              :", f"{trainable_parameters:,}")
            print("train samples           :", int(cfg.expected_train_samples))
            print("max epochs              :", int(cfg.max_epochs))
            print("steps/full epoch/rank   :", steps_per_full_epoch)
            print("exact target steps/rank :", target_steps)
            print("data fingerprint        :", data_fingerprint)
            print("target stats            :", target_stats_path)
            print("=" * 100)

        writer = (
            SummaryWriter(to_absolute_path(cfg.tensorboard_log_dir)) if rank == 0 else None
        )

        for epoch in range(start_epoch, int(cfg.max_epochs)):
            if global_step >= target_steps:
                break

            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            local_loss_sum = 0.0
            local_step1_sum = 0.0
            local_last_step_sum = 0.0
            local_count = 0
            epoch_start = time.perf_counter()

            remaining = target_steps - global_step
            this_epoch_target = min(steps_per_full_epoch, remaining)
            progress = tqdm(
                total=this_epoch_target,
                disable=(rank != 0),
                desc=f"epoch {epoch + 1}",
            )

            for item in iterator.iter_epoch(epoch, rank, world_size, device):
                if global_step >= target_steps:
                    break

                optimizer.zero_grad(set_to_none=True)
                result = unroll(
                    model=model,
                    packed=item["packed"],
                    start_step=item["start_step"],
                    edge_mean=edge_stats["edge_mean"],
                    edge_std=edge_stats["edge_std"],
                    target_mean=target_stats["mean"],
                    target_std=target_stats["std"],
                    device=device,
                    world_edge_radius=float(cfg.world_edge_radius),
                    feature_mode=str(cfg.feature_mode),
                    prediction_mode=str(cfg.prediction_mode),
                    theta_final_deg=float(cfg.theta_final_deg),
                    loss_weights=[float(x) for x in cfg.multistep_loss_weights],
                    noise_alpha=float(cfg.noise_alpha),
                    training=True,
                    amp_enabled=bool(cfg.amp),
                    use_whole_model_checkpoint=bool(
                        cfg.multistep_whole_model_checkpoint
                    ),
                )
                result.loss.backward()
                if float(cfg.grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.grad_clip_norm)
                    )
                optimizer.step()
                scheduler.step()
                global_step += 1

                local_loss_sum += float(result.loss.detach().item())
                local_step1_sum += float(result.step_losses[0].detach().item())
                local_last_step_sum += float(result.step_losses[-1].detach().item())
                local_count += 1
                if rank == 0:
                    progress.update(1)
                    progress.set_postfix(
                        loss=f"{float(result.loss.detach().item()):.3e}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

            if rank == 0:
                progress.close()

            torch.cuda.synchronize(device)
            epoch_elapsed = reduce_max(time.perf_counter() - epoch_start, device)
            total_count = reduce_sum(local_count, device)
            avg_loss = reduce_sum(local_loss_sum, device) / max(total_count, 1.0)
            avg_step1 = reduce_sum(local_step1_sum, device) / max(total_count, 1.0)
            avg_last = reduce_sum(local_last_step_sum, device) / max(total_count, 1.0)
            peak_reserved_gib = reduce_max(
                torch.cuda.max_memory_reserved(device) / 1024**3, device
            )

            # Always validate at regular intervals and also at the exact final step.
            val_result = None
            should_validate = (
                (epoch + 1) % int(cfg.rollout_val_interval) == 0
                or global_step >= target_steps
            )
            if should_validate:
                barrier()
                if rank == 0:
                    val_result = evaluate_manifest(
                        unwrap_model(model),
                        cache_dir,
                        "val",
                        edge_stats["edge_mean"],
                        edge_stats["edge_std"],
                        target_stats_path,
                        device,
                        cfg,
                        output_csv=os.path.join(
                            to_absolute_path(cfg.rollout_output_dir),
                            f"val_epoch_{epoch + 1:04d}_step_{global_step:07d}.csv",
                        ),
                    )
                    current_val = float(val_result[str(cfg.validation_primary_metric)])
                    if math.isfinite(current_val) and current_val < best_val_metric:
                        best_val_metric = current_val
                        atomic_save(
                            checkpoint_state(
                                cfg,
                                model,
                                optimizer,
                                scheduler,
                                epoch + 1,
                                global_step,
                                best_val_metric,
                                world_size,
                                data_fingerprint,
                            ),
                            best_checkpoint,
                        )
                        print(
                            f"[BEST] epoch={epoch + 1} step={global_step} "
                            f"{cfg.validation_primary_metric}={best_val_metric:.6e}",
                            flush=True,
                        )
                barrier()

            if rank == 0:
                row = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": avg_loss,
                    "step1_loss": avg_step1,
                    "last_step_loss": avg_last,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_time_sec": epoch_elapsed,
                    "peak_reserved_gib": peak_reserved_gib,
                    "val_mean_free_mm": ""
                    if val_result is None
                    else val_result["mean_free_mm"],
                    "val_final_free_mean_mm": ""
                    if val_result is None
                    else val_result["final_free_mean_mm"],
                    "val_max_free_mm": ""
                    if val_result is None
                    else val_result["max_free_mm"],
                    "best_val_metric": best_val_metric,
                    "trainable_parameters": trainable_parameters,
                    "data_fingerprint": data_fingerprint,
                }
                append_csv(metrics_csv, row)

                # Human-readable epoch summary for Slurm stdout.
                print(
                    f"[EPOCH] "
                    f"epoch={epoch + 1}/{int(cfg.max_epochs)} "
                    f"step={global_step}/{target_steps} "
                    f"train_loss={avg_loss:.6e} "
                    f"step1_loss={avg_step1:.6e} "
                    f"last_step_loss={avg_last:.6e} "
                    f"lr={float(optimizer.param_groups[0]['lr']):.6e} "
                    f"epoch_time_sec={epoch_elapsed:.2f} "
                    f"peak_reserved_gib={peak_reserved_gib:.3f}",
                    flush=True,
                )

                if val_result is not None:
                    print(
                        f"[VAL-SUMMARY] "
                        f"epoch={epoch + 1} "
                        f"step={global_step} "
                        f"mean_free_mm={float(val_result['mean_free_mm']):.6e} "
                        f"final_free_mm={float(val_result['final_free_mean_mm']):.6e} "
                        f"max_free_mm={float(val_result['max_free_mm']):.6e} "
                        f"best_val_metric={best_val_metric:.6e}",
                        flush=True,
                    )

                atomic_save(
                    checkpoint_state(
                        cfg,
                        model,
                        optimizer,
                        scheduler,
                        epoch + 1,
                        global_step,
                        best_val_metric,
                        world_size,
                        data_fingerprint,
                    ),
                    last_checkpoint,
                )
                if writer is not None:
                    writer.add_scalar("loss/train", avg_loss, epoch + 1)
                    writer.add_scalar("optimizer/global_step", global_step, epoch + 1)
                    writer.flush()
                if global_step >= target_steps:
                    print(
                        f"[STEP BUDGET COMPLETE] {global_step}/{target_steps}",
                        flush=True,
                    )

        if rank == 0 and writer is not None:
            writer.close()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
