# HiRISE 推理原理笔记

`src/inference.py` 实现了基于 CM-Diff 的 SCI（Statistical Constraint Inference）双向图像翻译：

- **Direction 0**：IR10 → RED4
- **Direction 1**：RED4 → IR10

---

## 一、预处理：Per-Scene IR10 Median/MAD 归一化 + clamp

训练和推理都使用 per-scene 的 IR10 median/MAD 归一化：

$$\text{center} = \mathrm{median}(X_{\text{IR10}}) \qquad \text{scale} = \max\bigl(1.4826 \times \mathrm{MAD}(X_{\text{IR10}}),\; 0.05\bigr)$$

$$X_{\text{norm}} = \mathrm{clamp}\!\left(\frac{X - \text{center}}{\text{scale}},\; -10,\; 10\right)$$

统计量**仅从 IR10 像素值计算**，对 IR10 和 RED4 使用相同的仿射变换，保留两个通道之间的真实光谱偏移（spectral offset）。

**为什么用 IR10 统计量而不是各自归一化？**
若对 IR10 和 RED4 各自做 min-max 归一化到 `[-1, 1]`，两者间真实的亮度差异（RED4 系统性地偏暗）会被人为消除，模型无法学习正确的跨通道映射。共享 IR10 统计量后：

- 归一化后 IR10 均值 ≈ 0（by construction，median 减除）
- 归一化后 RED4 均值 ≈ −0.22（波段偏差被真实保留）

**为什么用 median/MAD 而不用 mean/std？**
HiRISE 图像含有噪声列和仪器伪迹，MAD 对异常值鲁棒。系数 1.4826 是正态一致性校正。

**为什么需要 `clamp_min(0.05)` 和 `clamp(-10, 10)`？**
低对比度 patch（均匀沙丘/岩面）的 MAD 可能接近 0，不加限制时 scale = 1e-3 导致归一化值域 ±500，prior 统计（sigma、histogram range）完全失真。`clamp_min(0.05)` 将放大倍率上限设为 20×；`clamp(-10, 10)` 兜底截断残余离群值。**90% 的真实像素在 ±0.66 以内，clamp(-10,10) 从不截断实际地形细节。**

**推理时的归一化（`inference.py main()`）：**
脚本自动定位 IR10 文件来获取统计量：

```python
if "_IR10.npy" in src_path:
    ir10_path = src_path          # direction=0: source 本身就是 IR10
elif "_RED4.npy" in src_path:
    ir10_path = src_path.replace("_RED4.npy", "_IR10.npy")   # direction=1: 加载配对 IR10
```

找不到配对文件时退回到仅用源图统计（精度会略降）。

### 1.2 Method B: DC 减除

归一化之后，额外减去 IR10 归一化后的空间均值 $\mathrm{dc} = \overline{X_{\text{IR10}}^{\text{norm}}}$：

$$X_{\text{final}} = X_{\text{norm}} - \mathrm{dc}$$

这确保每个场景 IR10 均值精确为 0（by construction），使全局先验 $\mu_{\text{IR}} \approx 0$ 对所有场景都有效，SCI 以标准 $\lambda$ 值即可正常工作。

推理输出后加回 dc：$\text{result} = \text{model output} + \mathrm{dc}$

**实际数据范围：** 归一化 + dc 减除后，IR10 和 RED4 均在 [−10, 10] 之间（clamp），典型像素集中在 [−1, 1] 以内。

---

### 1.3 归一化参数的不对称性

center、scale、dc **三个参数全部由 IR10 的像素值计算**，与 RED4 无关：

| 参数 | 计算方式 |
|------|---------|
| center | median(IR10) |
| scale  | 1.4826 × MAD(IR10) |
| dc     | mean((IR10 − center) / scale) |

这导致两个推理方向在真实部署时有根本差异：

| 方向 | 推理时持有的数据 | 能否自主计算归一化参数 | 真实部署 |
|------|----------------|----------------------|---------|
| IR→RED | IR10（source） | ✓ 全部可计算 | 完全自洽 |
| RED→IR | RED4（source） | ✗ 需要 IR10（target）| 参数缺失 |

**评估时为何能正常运行：** 评估时数据集里两张图都在，`batch["norm_stats"]` 里的 center/scale/dc 是从真实 IR10 计算的，相当于隐式使用了目标域信息。对模型比较而言这是合法的，但不代表只拿到 RED4 的真实部署场景能做到。

**RED→IR 部署时 dc = 0：** 当 IR10 配对文件不可用时，令 dc = 0，等效于跳过 dc 减除步骤。此时：
- 模型输出已在归一化空间，直接报告归一化指标（MSE_norm、SSIM_norm、Pearson）是有效的
- 物理 DN 空间的精确还原不可能——因为 center/scale 也缺失，无法恢复绝对亮度
- 若要对 RED→IR 评估反归一化，只能在有 ground-truth IR10 的受控评估环境下进行（eval.py 的做法）

---

## 二、DDPM 逆向过程

### 2.1 x_T 初始化

DDPM 正向过程的边际分布为：

$$x_T = \sqrt{\bar\alpha_T}\, x_0 + \sqrt{1 - \bar\alpha_T}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

因此 $E[x_T] = \sqrt{\bar\alpha_T}\, E[x_0]$。用 `beta_end = 0.02` 时 $\sqrt{\bar\alpha_T} \approx 0.006$，这个均值虽小但不为零。若从纯 $\mathcal{N}(0, I)$ 开始推理，第一步计算 $\tilde{x}_0$ 时需要除以 $\sqrt{\bar\alpha_T} \approx 0.006$，约 167 倍的放大会让链立刻崩溃。

因此用目标域先验均值 $\mu_{\text{target}}$ 来种初始 DC：

```python
mu_target = prior_stats["mu"].item()
x_T = sqrt_ab_T * mu_target + sqrt_one_ab_T * randn(B, 1, H, W)
```

$\sqrt{\bar\alpha_T} \approx 0.006$，仅贡献约 0.6% 的 DC 偏置——足以避免第一步 $\tilde{x}_0$ 的 167× 放大崩溃，又不会让初始化偏向 source 的纹理。**不能**直接用 `x_source` 种，因为 source 和 target 的均值不同（如 direction=0 时 source 是 IR10 均值≈0，target 是 RED4 均值≈−3.35），用 source 会引入明显的 DC bias。

---

### 2.2 DDPM 后验均值（Posterior Mean）

每步逆向计算（`ddpm_step_sci()`）：

**Step 1** — UNet 预测噪声：

$$\hat\varepsilon = \text{UNet}(x_t,\, x_{\text{source}},\, \text{edge},\, \text{direction},\, t)$$

**Step 2** — 预测干净图像 $\tilde{x}_0$（tweedie estimate）：

$$\tilde{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon}{\sqrt{\bar\alpha_t}}$$

**Step 3** — DDPM 后验均值（论文 Eq. 11）：

$$\mu_q = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\, x_t + \sqrt{\bar\alpha_{t-1}}\,\beta_t\, \tilde{x}_0}{1 - \bar\alpha_t}$$

其中 $\bar\alpha_t = \prod_{i=1}^t (1-\beta_i)$，$\alpha_t = 1-\beta_t$。

后验方差 $\sigma_q^2 = \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t$。

---

## 三、SCI 统计约束推理

SCI 在每步 $\tilde{x}_0$ 上施加一个梯度校正，将预测拉向目标域的统计先验。

### 3.1 两个约束损失

**L_scl**（Statistical Constraint Loss，Eq. 14）——约束均值和标准差：

$$L_{\text{scl}} = |\mu_{\text{pred}} - \mu_{\text{prior}}| + |\sigma_{\text{pred}} - \sigma_{\text{prior}}|$$

**L_ccl**（Channel Constraint Loss，Eq. 13）——约束直方图形状（Chi-squared 距离）：

$$L_{\text{ccl}} = \sum_k \frac{(h_{\text{pred},k} - h_{\text{prior},k})^2}{h_{\text{pred},k} + h_{\text{prior},k}}$$

其中 $h_{\text{pred}}$ 通过可微分软直方图（Gaussian bin assignment）从 $\tilde{x}_0$ 计算，$h_{\text{prior}}$ 由训练集统计预先算好存入 `prior_*.pt`。

组合损失：

$$L_{\text{cons}} = \lambda_{\text{scl}} \cdot L_{\text{scl}} + \lambda_{\text{ccl}} \cdot L_{\text{ccl}}$$

### 3.2 梯度校正与更新

计算 $\nabla_{\tilde{x}_0} L_{\text{cons}}$，并对后验均值施加梯度下降校正：

$$\mu_{\text{updated}} = \mu_q - \sigma_q \cdot \nabla_{\tilde{x}_0} L_{\text{cons}}$$

方向是**梯度下降**（减号），使 $\tilde{x}_0$ 的统计特性向先验靠拢。然后采样：

$$x_{t-1} \sim \mathcal{N}(\mu_{\text{updated}},\, \sigma_q^2) \quad (t > 0)$$

$$x_0 = \mu_{\text{updated}} \quad (t = 0, \text{ deterministic})$$

$\tilde{x}_0$ 是唯一需要梯度的变量，其余量均 `.detach()`，因此每步只需一次前向 + 一次反向（仅对 $\tilde{x}_0$，不经过 UNet）。

### 3.3 先验文件

| 文件 | 对应方向 | 内容 |
|------|----------|------|
| `src/output/prior_red.pt` | direction=0（IR→RED） | RED4 训练集的 μ, σ, 直方图, hist\_min/max |
| `src/output/prior_ir.pt`  | direction=1（RED→IR） | IR10 训练集的 μ, σ, 直方图, hist\_min/max |

先验由 `src/compute_prior.py`（或 `scripts/compute_prior.sh`）从训练集计算，读取 `DiffusionDataset` 输出的 `ir`/`red` 字段（已含 per-scene IR10 median/MAD 归一化），无需手动调整。

**当前先验统计量**（clamp_min=0.05 + clamp(-10,10) 归一化空间，重算于 2026-03-29）：

| 文件 | μ | σ | hist range | active bins |
|------|---|---|------------|-------------|
| prior_ir.pt | −0.0075 | 0.2249 | [−10, 10] | 48/256 |
| prior_red.pt | −0.2169 | 0.2815 | [−10, 10] | 41/256 |

---

## 四、与 CM-Diff 原文的区别

| 方面 | CM-Diff 原文 | 本实现 |
|------|-------------|--------|
| 归一化 | 全局 min-max → `[-1, 1]` | Per-scene IR10 median/MAD，clamp_min(0.05)，clamp(-10,10) |
| 两通道归一化统计量 | 各自 min-max | 共享 IR10 统计量，保留波段偏移 |
| 先验 μ 对单场景是否有效 | 是（所有图在同一绝对空间） | 是（μ_IR ≈ 0；RED4 波段偏差被保留为 μ_RED ≈ −0.22） |
| 输出值域 | 固定 `[-1, 1]` | clamp 到 [−10, 10]，典型像素在 [−1, 1] 以内 |
| 推荐 λ | 20 | 待调参（Run 3 训练后重新评估） |
| prior 文件范围 | 固定 `[-1, 1]` | [−10, 10]（由 clamp 范围决定，hist active bins 约 48/256） |

CM-Diff 原文用全局 min-max 的核心优势是所有图像处于同一绝对空间，SCI 先验的 μ/σ 对每张图都直接有效。本实现选择 per-scene IR10 median/MAD 是为了对 HiRISE 仪器噪声鲁棒，同时保留 IR10 与 RED4 之间真实的光谱偏移，使 SCI 先验统计量仍具物理意义。

---

## 五、完整推理数据流

### Direction 0：IR→RED（完全自洽）

```
输入 IR10.npy  (raw DN values)
  │
  ▼  center = median(IR10),  scale = 1.4826 × MAD(IR10)   ← 全部从 source 计算
归一化  IR10_norm = clamp((IR10 − center) / scale, −10, 10)
  │
  ▼  dc = mean(IR10_norm)
dc 减除  IR10_final = IR10_norm − dc   →  mean = 0 (exact)
  │
  ▼  x_T = sqrt(ᾱ_T)·μ_RED + sqrt(1−ᾱ_T)·ε      （prior_red.pt 的 μ ≈ −0.597）
逆向扩散  T=1000 步，每步:
  │     UNet 预测 ε̂  →  x̃₀  →  SCI 梯度校正  →  采样 x_{t-1}
  ▼
RED4_pred_norm  (归一化空间)
  │
  ▼  RED4_pred = (RED4_pred_norm + dc) × scale + center   ← dc/scale/center 均已知
物理 DN 值  RED4.npy
```

### Direction 1：RED→IR

```
输入 RED4.npy  (raw DN values)
  │
  ├─ 评估环境（有配对 IR10）：
  │     center = median(IR10_GT),  scale = 1.4826 × MAD(IR10_GT)
  │     dc     = mean(IR10_GT_norm)            ← 使用 ground-truth IR10 计算
  │
  └─ 真实部署（无配对 IR10）：
        center/scale ≈ 用 RED4 自身统计估算（精度下降）
        dc = 0                                 ← 无 IR10 则跳过 dc 减除
  │
  ▼  RED4_norm = clamp((RED4 − center) / scale, −10, 10)  （−dc，若 dc 可知）
  │
  ▼  x_T = sqrt(ᾱ_T)·μ_IR + sqrt(1−ᾱ_T)·ε       （prior_ir.pt 的 μ ≈ 0）
逆向扩散  T=1000 步，每步:
  │     UNet 预测 ε̂  →  x̃₀  →  SCI 梯度校正  →  采样 x_{t-1}
  ▼
IR10_pred_norm  (归一化空间)
  │
  ├─ 评估环境：IR10_pred = (IR10_pred_norm + dc) × scale + center  → 物理 DN 值
  │
  └─ 真实部署：仅报归一化指标（MSE_norm / SSIM_norm / Pearson），不做物理还原
```

---

## 六、使用方法

### 命令行

```bash
# Direction 0: IR10 → RED4
.venv/Scripts/python.exe src/inference.py \
    --source    data/files/npy_files_b12/ESP_058229_1150/ESP_058229_1150_0_IR10.npy \
    --direction 0 \
    --checkpoint src/output/latest.pt \
    --prior_dir  src/output \
    --output     outputs/pred_RED4.npy

# Direction 1: RED4 → IR10
.venv/Scripts/python.exe src/inference.py \
    --source    data/files/npy_files_b12/ESP_058229_1150/ESP_058229_1150_0_RED4.npy \
    --direction 1 \
    --checkpoint src/output/latest.pt \
    --prior_dir  src/output \
    --output     outputs/pred_IR10.npy
```

### Notebook 中调用

```python
from inference import sample
from compute_prior import load_prior_stats
from config import InferenceConfig

prior_stats = load_prior_stats("src/output/prior_red.pt", device)
cfg = InferenceConfig(lambda_scl=20.0, lambda_ccl=20.0)

with torch.no_grad():
    pred = sample(model, scheduler, ir_tensor, direction=0,
                  prior_stats=prior_stats, cfg_inf=cfg, device=device)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | 必填 | 源图像路径（.npy） |
| `--direction` | 必填 | 0 = IR→RED，1 = RED→IR |
| `--checkpoint` | `src/output/latest.pt` | 模型权重 |
| `--prior_dir` | `src/output` | prior 统计文件目录 |
| `--output` | `../outputs/result.npy` | 输出路径（.npy） |
| `--device` | `cuda` | 无 GPU 时自动降级为 cpu |

`lambda_scl` 和 `lambda_ccl` 在 `src/config.py InferenceConfig` 中设置，默认 0（禁用 SCI）。Method B 重训练完成后建议设为 20。
