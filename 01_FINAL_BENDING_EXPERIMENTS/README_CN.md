# FINAL_BENDING_EXPERIMENTS

这是由旧 MGN、MGN-T 与 `ANARESID_DATA_EFFICIENCY_X10_CLEAN_V2` 合并得到的最终受控实验框架。

## 已冻结的实验变量

主实验全部使用相同 120/15/15 split、同一 shared-raw packed cache、相同 world-edge 构造、相同 TrainN-only edge/target normalization、scratch initialization 和相同 closed-loop evaluator。

- MGN baseline: **H80**, 15 message-passing blocks（H128 在现有 32 GiB GPU 上不可行，因此论文必须如实报告 H80）。
- MGN-T/ANARESID: **H128**, 2 local MP + 2 Transformer + 2 local MP。
- X3 = NodeType3。
- X6 = X3 + `[D, t/D, R/D]`。
- X10 = X6 + `[s/L, sin(phi), cos(phi), angle_progress]`。
- Direct 学习世界坐标增量 `ΔX`。
- Global-ANARESID 学习全局坐标中的解析残差增量 `Δe_global`。
- Local-ANARESID 学习随局部解析弯曲基旋转的 `Δe_local`。
- 正式主实验均 2-step；额外 `exp10` 为 1-step training-strategy ablation。所有正式配置都固定 `target_optimizer_steps_per_rank=322200`，因此 1-step 也不会因为每条轨迹多一个有效窗口而获得额外 optimizer updates。

## 10 个配置

1. `exp01_mgn_x10_direct_2step.yaml`
2. `exp02_mgnt_x10_direct_2step.yaml`
3. `exp03_global_anaresid_x10_2step.yaml`
4. `exp04_local_anaresid_x10_2step.yaml`  ← 核心 N120
5. `exp05_local_anaresid_x3_2step.yaml`
6. `exp06_local_anaresid_x6_2step.yaml`
7. `exp07_local_x10_n040_2step.yaml`
8. `exp08_local_x10_n060_2step.yaml`
9. `exp09_local_x10_n080_2step.yaml`
10. `exp10_local_x10_n120_1step.yaml` ← 2-step 消融对照

另有 `supp_mgnt_h80_x10_direct_2step.yaml`，用于可选的 H80 capacity-controlled supplementary comparison。

## Slurm 任务命名与提交约定

所有可直接 `sbatch` 提交的脚本都使用 `ysy_` 作为 Slurm job name 前缀，便于后续按账号/任务名前缀统计 GPU 花费：

- `run_compute_stats_1gpu_5090.sh` → `ysy_final_stats`
- `run_smoke_1gpu_5090.sh` → `ysy_final_smoke`
- `run_train_4gpu_5090.sh` → `ysy_final_bend`
- `run_infer_1gpu_5090.sh` → `ysy_final_inf`

GPU 数量与分区不写死在 `.sh` 中，按集群习惯在提交命令里显式指定，例如：

```bash
sbatch --gpus=1 -p gpu_5090 \
  ./slurm/run_infer_1gpu_5090.sh conf/exp07_local_x10_n040_2step.yaml test
```

4 GPU 训练则使用 `--gpus=4`。

## sample007 / sample138 修复后必须做的顺序

1. 把最终 clean NPZ 放到数据目录，并先跑 `check_clean_npz150_against_csv.py`。
2. 重新执行 `preprocess_shared_raw_x10_v2.py`。它会通过 source size/mtime 检查修复后的样本并更新 packed cache。
3. 用 `prepare_dataeff_workspace.py` 建立 N40/N60/N80/N120 嵌套视图。
4. **正式论文数据必须执行 `compute_final_stats.py --resume 0`**。新统计脚本同时生成 Direct/Global/Local 三套 target-increment statistics，并给 per-sample stats 增加 packed 文件指纹，避免 007/138 旧统计缓存被误复用。
5. `python check_experiment_contracts.py`。
6. 先 smoke test，再正式 4 GPU 训练。checkpoint 会记录冻结数据指纹；如果 007/138 或任何训练样本之后再次变化，旧 checkpoint 会被拒绝续训/评估。

## Smoke test

```bash
sbatch --gpus=1 -p gpu_5090 \
  ./slurm/run_smoke_1gpu_5090.sh conf/exp04_local_anaresid_x10_2step.yaml
```

## 训练

例如核心 Local-ANARESID-X10：

```bash
sbatch --gpus=4 -p gpu_5090 \
  ./slurm/run_train_4gpu_5090.sh conf/exp04_local_anaresid_x10_2step.yaml
```

MGN baseline：

```bash
sbatch --gpus=4 -p gpu_5090 \
  ./slurm/run_train_4gpu_5090.sh conf/exp01_mgn_x10_direct_2step.yaml
```

## 推理

```bash
sbatch --gpus=1 -p gpu_5090 \
  ./slurm/run_infer_1gpu_5090.sh conf/exp04_local_anaresid_x10_2step.yaml test
```

## 重要说明

本包已进行 Python 语法检查和配置合同静态检查，但当前 ChatGPT 容器没有你的 PhysicsNeMo/数据/GPU 环境，因此第一次上机仍应先运行一个极短 smoke test。正式实验不要混用旧 checkpoint；所有实验应从 scratch 开始。
