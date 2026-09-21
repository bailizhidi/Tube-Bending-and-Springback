# POD_ROM_DISPLACEMENT_FINAL

只包含弯管 **位移场** POD-ROM，不包含应力、应变、PEEQ。

## 1. 三种严格可控的位移表示

三种方法使用完全相同的：

- Final Clean150：Train120 / Val15 / Test15
- 固定网格：64 × 28 × 3
- POD rank：正式 ROM 固定为 16
- POD q：256（用于谱/Oracle 分析）
- MLP：10 → 128 → 128 → 128 → 16
- Process-enhanced 10D 输入
- AdamW, lr=1e-3, batch=512, patience=120, seed=0

只改变 POD 分解对象：

### Direct

`field = U_FE`

### Global residual

`U_ana = A - X_ref`

`e_global = U_FE - U_ana`

`field = e_global`

### Local residual

先在原生 FE tube mesh 上：

`e_global = U_FE - U_ana`

再使用与 FINAL_BENDING_EXPERIMENTS 完全相同的共旋坐标：

`e_local = M(psi)^T e_global`

最后才将 `e_local` 插值到 64×28 固定网格。

重构时严格逆变换：fixed local residual → native local residual → global residual → `U_pred = U_ana + e_global`。

## 2. 正式评价合同（与 GNN 对齐）

所有最终 ROM 指标：

- 仅评价 free tube nodes（`node_type == 0`）
- 不评价 frame 0，仅评价 frame 1...180
- 先逐 case 计算，再对 15 个 held-out geometry 等权平均
- 报告 `mean_free_mm`, `final_free_mm`, `p95/p99`, `worst_max_free_mm`, case-balanced RelL2

## 3. 数据目录

默认：

`/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_multifield_150_npz_compact`

可以通过环境变量 `DATA_DIR=/other/path` 覆盖。

## 4. 推荐执行顺序

### A. 数据合同检查（登录节点即可）

```bash
python check_dataset_contract.py \
  --data-dir /data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_bending_multifield_150_npz_compact
```

### B. 分别拟合三套 POD basis

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_fit_basis_1gpu_5090.sh direct
sbatch --gpus=1 -p gpu_5090 ./slurm/run_fit_basis_1gpu_5090.sh global_residual
sbatch --gpus=1 -p gpu_5090 ./slurm/run_fit_basis_1gpu_5090.sh local_residual
```

### C. Validation POD Oracle / rank curve

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_oracle_1gpu_5090.sh direct val
sbatch --gpus=1 -p gpu_5090 ./slurm/run_oracle_1gpu_5090.sh global_residual val
sbatch --gpus=1 -p gpu_5090 ./slurm/run_oracle_1gpu_5090.sh local_residual val
```

三套 basis 都完成后比较奇异值谱：

```bash
python compare_pod_spectra.py \
  --direct artifacts/pod_basis_direct_train120_grid64x28_q256.npz \
  --global-residual artifacts/pod_basis_global_residual_train120_grid64x28_q256.npz \
  --local-residual artifacts/pod_basis_local_residual_train120_grid64x28_q256.npz
```

### D. 提取 Train120 + Val15 POD 系数

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_extract_coeffs_1gpu_5090.sh direct
sbatch --gpus=1 -p gpu_5090 ./slurm/run_extract_coeffs_1gpu_5090.sh global_residual
sbatch --gpus=1 -p gpu_5090 ./slurm/run_extract_coeffs_1gpu_5090.sh local_residual
```

### E. 训练三套 rank-16 MLP

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_train_rom_1gpu_5090.sh direct
sbatch --gpus=1 -p gpu_5090 ./slurm/run_train_rom_1gpu_5090.sh global_residual
sbatch --gpus=1 -p gpu_5090 ./slurm/run_train_rom_1gpu_5090.sh local_residual
```

### F. 先评价 Validation

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_eval_rom_1gpu_5090.sh direct val
sbatch --gpus=1 -p gpu_5090 ./slurm/run_eval_rom_1gpu_5090.sh global_residual val
sbatch --gpus=1 -p gpu_5090 ./slurm/run_eval_rom_1gpu_5090.sh local_residual val
```

汇总：

```bash
python summarize_final_results.py --split val
```

### G. Validation 接受后再打开 Test15

```bash
sbatch --gpus=1 -p gpu_5090 ./slurm/run_eval_rom_1gpu_5090.sh direct test
sbatch --gpus=1 -p gpu_5090 ./slurm/run_eval_rom_1gpu_5090.sh global_residual test
sbatch --gpus=1 -p gpu_5090 ./slurm/run_eval_rom_1gpu_5090.sh local_residual test
```

最终汇总：

```bash
python summarize_final_results.py --split test
```

## 5. 关键输出

- `artifacts/pod_basis_<rep>_train120_grid64x28_q256.npz`
- `artifacts/pod_coeff_<rep>_rank32.npz`
- `artifacts/pod_rom_<rep>_rank16_process.pt`
- `results/<rep>/val/oracle_rank_curve.csv`
- `results/<rep>/val/summary.json`
- `results/<rep>/test/summary.json`
- `results/pod_spectrum_comparison.csv`
- `results/final_val_comparison.csv`
- `results/final_test_comparison.csv`

## 6. Slurm 计费标识

所有 `.sh` 任务名均以 `ysy_` 开头：

- `ysy_pod_fit`
- `ysy_pod_oracle`
- `ysy_pod_coeff`
- `ysy_pod_train`
- `ysy_pod_eval`

所有脚本使用 `--cpus-per-gpu=8`；GPU 数和队列由提交命令显式给出。
