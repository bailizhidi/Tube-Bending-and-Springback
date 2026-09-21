# Base_TC4-150 单材料回弹 MGN：超算正式版

## 固定数据合同

- 单材料：Base_TC4
- geometry：150
- 每个 geometry：5°–180°，每 5° 一个回弹 case，共 36 个
- 总 pair NPZ：5400
- Train/Val/Test：120/15/15 geometry，即 4320/540/540 graph
- 每个 springback ODB 原始定义：2 帧
- 网络监督目标：`dU_springback = U_last - U_frame0`
- frame0 是更新参考构型后的弯曲终态，理论上 `U_frame0 ≈ 0`，但训练目标始终使用显式差值而不是直接把 `U_last` 当目标。

## 超算路径

Windows 已提取的回弹数据压缩包上传到：

```text
/data/home/scxk573/run/lsz/ysy/physicsnemo/data/zip
```

解压后的正式 NPZ 数据目录：

```text
/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_springback_pair_Base_TC4_150
```

工程目录：

```text
/data/home/scxk573/run/lsz/ysy/physicsnemo/examples/structural_mechanics/MGN_springback_onestep_150_singlemat
```

脚本会自动对存在的路径执行 `readlink -f`，因此计算节点看到 `/data/run01/...` 时也可正常工作。

## 1. 解压回弹数据

建议 Windows 压缩包命名：

```text
dataset_springback_pair_Base_TC4_150.tar.gz
```

登录节点执行：

```bash
cd /data/home/scxk573/run/lsz/ysy/physicsnemo/examples/structural_mechanics/MGN_springback_onestep_150_singlemat
./scripts/00_show_paths.sh
./scripts/01_unpack_uploaded_dataset.sh
```

如果 `data/zip` 里有多个匹配压缩包，可显式指定：

```bash
ARCHIVE=/data/home/scxk573/run/lsz/ysy/physicsnemo/data/zip/dataset_springback_pair_Base_TC4_150.tar.gz \
./scripts/01_unpack_uploaded_dataset.sh
```

## 2. 创建日志目录

Slurm 在启动任务前需要日志目录已经存在：

```bash
mkdir -p logs
```

## 3. 正式检查 5400 NPZ

```bash
sbatch --job-name=ysy_sb_verify --gpus=1 -p gpu_4090 ./slurm/run_verify_dataset_1gpu_4090.sh
```

必须看到：

```text
PAIR DATASET CONTRACT PASSED: 150 x 36 = 5400
CONTRACT PASSED
[VERIFY] PASSED
```

## 4. A-geom cache

```bash
sbatch --job-name=ysy_sb_cache_geom --gpus=1 -p gpu_5090 ./slurm/run_prepare_geom_1gpu_5090.sh
```

应得到 node_dim=8, edge_dim=4, out_dim=3。

如果后续要做 `B_stress+PEEQ` 消融，再单独准备：

```bash
sbatch --job-name=ysy_sb_cache_stress --gpus=1 -p gpu_5090 ./slurm/run_prepare_stress_1gpu_5090.sh
```

## 5. 1 GPU smoke test

```bash
sbatch --job-name=ysy_sb_smoke --gpus=1 -p gpu_4090 ./slurm/run_smoke_AW_1gpu_4090.sh
```

## 6. 正式 AW-SpringMGN

优先 4×5090：

```bash
sbatch --job-name=ysy_sb_AW --gpus=4 -p gpu_5090 ./slurm/run_train_4gpu_5090.sh conf/config_AW_150.yaml AW_SpringMGN_150
```

如果 5090 紧张，可以使用 4×4090：

```bash
sbatch --job-name=ysy_sb_AW --gpus=4 -p gpu_4090 ./slurm/run_train_4gpu_4090.sh conf/config_AW_150.yaml AW_SpringMGN_150
```

## 7. 三个正式回弹消融

A-geom data-only：

```bash
sbatch --job-name=ysy_sb_A_data --gpus=4 -p gpu_5090 ./slurm/run_train_4gpu_5090.sh conf/config_A_geom_dataonly_150.yaml A_geom_dataonly_150
```

A-geom + EdgeGrad + TopK：

```bash
sbatch --job-name=ysy_sb_EGTopK --gpus=4 -p gpu_5090 ./slurm/run_train_4gpu_5090.sh conf/config_A_geom_EGTopK_150.yaml A_geom_EGTopK_150
```

B-stress + PEEQ data-only（必须先准备 stress cache）：

```bash
sbatch --job-name=ysy_sb_Bstress --gpus=4 -p gpu_5090 ./slurm/run_train_4gpu_5090.sh conf/config_B_stress_peeq_dataonly_150.yaml B_stress_peeq_dataonly_150
```

## 8. Test15 × 36 = 540 case 正式评估

```bash
sbatch --job-name=ysy_sb_eval --gpus=1 -p gpu_4090 ./slurm/run_eval_AW_1gpu_4090.sh
```

输出：

```text
predictions/AW_SpringMGN_150_test/metrics_test.csv
predictions/AW_SpringMGN_150_test/angle_metrics_centerline.csv
```

## 资源合同

- 1×4090 = 6 CPU，因此 4090 Slurm 脚本使用 `#SBATCH --cpus-per-gpu=6`
- 1×5090 = 8 CPU，因此 5090 Slurm 脚本使用 `#SBATCH --cpus-per-gpu=8`
- 4×4090 = 24 CPU
- 4×5090 = 32 CPU
- 不设置 time limit
- Slurm stdout/stderr 写入 `./logs`
- 所有正式提交命令显式写 `--gpus=N -p <partition>`
