# scripts/ — 使用手册

## 文件说明

| 文件 | 用途 |
|------|------|
| `train.sh` | SLURM 集群训练提交脚本 |

---

## 快速开始

```bash
# 在集群登录节点执行
cd /scratch_root/ed425/HiRISE/HiRISE_diffusion
sbatch scripts/train.sh
```

---

## train.sh 配置

提交前根据实际情况修改脚本顶部的变量：

```bash
PROJECT_ROOT=/scratch_root/ed425/HiRISE/HiRISE_diffusion   # 项目根目录（包含 src/）
DATA_ROOT=/scratch_root/ed425/HiRISE/files                  # .npy 文件所在根目录
CSV_PATH=/scratch_root/ed425/HiRISE/data_record_bin12.csv   # 数据记录 CSV 的绝对路径
CKPT_DIR=/scratch_root/ed425/HiRISE_diffusion/src/models/   # checkpoint 输出目录
```

CSV 中每行 `Path` 字段的值会被拼接到 `DATA_ROOT` 后形成完整路径，例如：

```
DATA_ROOT/ESP_026721_1635/ESP_026721_1635_0_IR10.npy
```

### SLURM 资源参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `--partition` | `root` | 队列名，按集群实际修改 |
| `--cpus-per-task` | `6` | 匹配 DataLoader `num_workers=4` + 余量 |
| `--gres=gpu:1` | 1 张 GPU | 单卡训练 |
| `--mem` | `64G` | 内存 |
| `--time` | `24:00:00` | 最长运行时间，超时会被 SLURM 强制终止 |

> 若 100k 步在 24h 内跑不完，可改为 `--time=48:00:00` 或分多次续训（见下方）。

---

## W&B 可视化

### API Key 安全

**不要把 API Key 硬编码在脚本里**（存在泄露风险）。推荐做法：

```bash
# 在登录节点执行一次，Key 会存入 ~/.netrc
wandb login
```

如果集群无法交互登录，可在提交前临时设置环境变量：

```bash
export WANDB_API_KEY=<your_key>
sbatch scripts/train.sh
```

或在 `~/.bashrc` / `~/.bash_profile` 里持久化，避免明文写入脚本。

### 指标说明

| W&B 指标 | 含义 |
|----------|------|
| `train/loss` | L_joint 平均（每 `log_every` 步） |
| `train/loss_A` | IR10→RED4 方向损失 |
| `train/loss_B` | RED4→IR10 方向损失 |
| `train/lr` | 当前学习率 |
| `train/grad_norm` | 梯度裁剪前的梯度范数（监控训练稳定性） |
| `checkpoint/step` | 保存 checkpoint 时的步数 |

### 禁用 W&B

本地调试或无网络节点时：

```bash
python src/train.py --no_wandb ...
```

---

## train.py 完整参数

```
python src/train.py [选项]

选项：
  --data_root     DATA_ROOT      .npy 文件根目录（覆盖 config.py）
  --csv_path      CSV_PATH       数据记录 CSV 绝对路径（覆盖 config.py）
  --ckpt_dir      CKPT_DIR       checkpoint 输出目录（默认 <project_root>/checkpoints）
  --wandb_project PROJECT        W&B 项目名（默认 HiRISE_diffusion）
  --run_name      NAME           W&B run 名称（默认自动生成）
  --no_wandb                     关闭 W&B 日志
```

优先级：CLI 参数 > `src/config.py` > 项目相对路径默认值。

---

## 断点续训

训练每 `save_every`（默认 5000）步保存两个文件：

```
CKPT_DIR/
    step_0005000.pt   # 历史快照（可回溯任意步）
    step_0010000.pt
    ...
    latest.pt         # 始终指向最新步
```

再次提交同一个 `sbatch` 命令即可自动续训：脚本启动时检测到 `latest.pt` 存在，会从断点步数继续。

---

## 查看 SLURM 日志

```bash
# 实时跟踪当前 job 输出
tail -f /scratch_root/ed425/HiRISE_diffusion/scripts/logs/train_<JOBID>.log

# 查看所有 job 状态
squeue -u ed425

# 取消 job
scancel <JOBID>
```

---

## 训练配置（src/config.py）

主要超参数不通过 CLI 传入，直接编辑 `src/config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 6 | 论文 Table 7 |
| `lr` | 1e-4 | AdamW 初始学习率 |
| `lr_decay` | 0.9 | StepLR 衰减因子 |
| `lr_decay_every` | 2000 | 每 N 步衰减一次 |
| `total_steps` | 100 000 | 总训练步数 |
| `timesteps` | 1000 | DDPM 扩散步数 |
| `base_channels` | 128 | UNet 基础通道数 C |
| `num_res_blocks` | 3 | 每个分辨率级别的 ResBlock 数 |
