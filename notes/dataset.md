# HiRISE 数据集说明

## 1. 数据来源

HiRISE（High Resolution Imaging Science Experiment）是火星侦察轨道器（MRO）上的高分辨率相机，
拍摄火星地表的多光谱图像。每次观测（Observation）对应一个地理位置，包含多个 CCD 通道。

---

## 2. 目录结构

```
data/files/
├── data_record_bin12.csv       # 索引表，Binning=1 或 2 的观测（主要数据集）
├── data_record_bin22.csv       # 索引表，Binning=2 的观测
├── npy_files_b12/              # Binning=1/2 的 npy 文件
│   ├── ESP_058229_1150/
│   │   ├── ESP_058229_1150_0_BG12.npy
│   │   ├── ESP_058229_1150_0_IR10.npy
│   │   ├── ESP_058229_1150_0_RED3.npy
│   │   ├── ESP_058229_1150_0_RED4.npy
│   │   └── ESP_058229_1150_0_RED5.npy
│   └── ...
└── npy_files_b22/              # Binning=2 的 npy 文件
```

文件命名规则：`{Observation}_{Patch}_{CCD}.npy`

---

## 3. CCD 通道说明

HiRISE 相机有多个 CCD 排列成一列，同时拍摄同一地区的不同波段或空间位置：

```
空间排列（扫描方向）：
  ← RED3 | RED4（中心） | RED5 →
          IR10 / BG12（与 RED4 像素对齐）
```

| CCD | 波长 | 形状 | 用途 |
|---|---|---|---|
| **RED4** | 红波段（~600–830 nm） | (256, 256) | **预测目标 y** |
| **IR10** | 近红外（~800–1000 nm） | (256, 256) | **模型主输入（方向A的source）** |
| BG12 | 蓝绿（~400–600 nm） | (256, 256) | 本项目不使用 |
| RED3 | 红波段，左邻 CCD | (1024, 32, 4) | SCI 推理先验来源 |
| RED5 | 红波段，右邻 CCD | (1024, 32, 4) | SCI 推理先验来源 |

---

## 4. npy 文件内容

### BG12 / IR10 / RED4（像素对齐通道）

```
shape : (256, 256)
dtype : float32
值域  : [0, ~0.5]（反射率，已归一化）
```

这三个通道和目标 RED4 **像素级对齐**，可以直接逐像素比较。

示例统计（ESP_058229_1150）：

| CCD | min | max | mean | std |
|---|---|---|---|---|
| BG12 | 0.039 | 0.047 | 0.044 | 0.001 |
| IR10 | 0.096 | 0.119 | 0.108 | 0.003 |
| RED4 | 0.080 | 0.107 | 0.095 | 0.003 |

IR10 的亮度约为 BG12 的 2.5 倍，RED4 介于两者之间，符合火星地表的近红外反射特性。

### RED3 / RED5（空间邻居通道）

```
shape : (1024, 32, 4)
dtype : float32
```

- Axis 0（1024）：扫描行方向，对应空间高度
- Axis 1（32）：CCD 宽度（相邻 CCD 比 RED4 窄）
- Axis 2（4）：4 个子通道，对应 `[BG, IR, RE, RED]`（dataset 只取前3个：BG/IR/RE）

RED3 / RED5 与 RED4 **空间上不完全对齐**（相机几何和卫星运动引起偏移），
因此不能直接逐像素使用，只能作为统计先验参考。

**4 个子通道的含义与 SCI prior 选择：**

| ch | 波段 | mean（示例） | 与 RED4 mean 差距 |
|---|---|---|---|
| ch0 | BG（蓝绿） | 0.044 | 0.051 — 不适合 |
| ch1 | IR（近红外） | 0.107 | 0.012 — 不适合 |
| ch2 | RE（红边） | 0.095 | **0.000** — 适合作 prior |
| ch3 | RED4 配准值 | 0.095 | **0.000** — 丢弃（数据泄露）|

- **ch2（RE）** 是 SCI prior 的正确来源：同波段邻居，和 RED4 分布几乎完全一致
- **ch3 被丢弃**：代码注释 `Drop r4`，该通道是 RED4 在邻居位置的配准值，保留会造成目标泄露
- **ch1（IR）不能用作 RED4 的 prior**：IR 均值比 RED4 高约 13%，分布不同

---

## 5. CSV 索引表结构

CSV 是所有 npy 文件的元信息索引。

| 列 | 含义 |
|---|---|
| `Observation` | 观测 ID（轨道号_纬度），对应文件夹名 |
| `Set` | Observation + Patch 的组合 ID，同一位置同一块的所有 CCD 共享同一 Set |
| `Patch` | 同一 Observation 内的第几个 256×256 块（从 0 开始） |
| `CCD` | 通道名（BG12 / IR10 / RED3 / RED4 / RED5） |
| `Path` | npy 文件相对路径（相对于 `data/` 目录） |
| `Pix_min / Pix_max` | 该 npy 文件的像素值范围 |
| `Binning` | 像素合并因子（1 或 2），影响空间分辨率 |
| `TDI` | 时间延迟积分行数，影响信噪比 |
| 其余列 | 拍摄时的轨道/太阳/温度等元数据 |

**数据集规模（bin12）：**
- 观测数：12,495
- Set 数：17,873
- npy 文件数：89,365
- bin22 额外 Set 数：19,508

---

## 6. DiffusionDataset 输出（`__getitem__` 返回值）

`data/dataset.py` 中的 `DiffusionDataset` 是独立的扩散训练数据集类（不继承 `HiRISEDataset` / `FilteredHiRISEDataset`）。

| 键 | 形状 | 内容 |
|---|---|---|
| `ir` | (1, H, W) | IR10，归一化后 |
| `red` | (1, H, W) | RED4，归一化后（与 IR10 使用相同的统计量） |
| `norm_stats` | (3,) | `[center, scale, dc]`，归一化参数（用于反归一化） |
| `obs_id` | str | 观测 ID |
| `set_name` | int | Set 编号 |
| `date` | str | 拍摄日期 |

### 归一化方式：Per-Scene IR10 Median/MAD + clamp

IR10 和 RED4 使用**同一组归一化统计量**，统计量仅由 IR10 的像素值确定：

$$\text{center} = \mathrm{median}(X_{\text{IR10}}) \qquad \text{scale} = \max\bigl(1.4826 \times \mathrm{MAD}(X_{\text{IR10}}),\; 0.05\bigr)$$

$$\text{ir\_norm} = \mathrm{clamp}\!\left(\frac{X_{\text{IR10}} - \text{center}}{\text{scale}},\; -10,\; 10\right) \qquad \text{red\_norm} = \mathrm{clamp}\!\left(\frac{X_{\text{RED4}} - \text{center}}{\text{scale}},\; -10,\; 10\right)$$

**为什么用 IR10 统计量而不是各自归一化？**

RED4 和 IR10 来自同一地物，但传感器响应不同，RED4 系统性地比 IR10 暗。若对两个通道分别做 min-max 归一化到 `[-1, 1]`，这一真实的**光谱偏移（spectral offset）被人为消除**，模型看不到两个通道的亮度差异，无法学习正确的跨通道映射。使用 IR10 统计量后保留了波段偏移：

- 归一化后 IR10 均值 ≈ 0（by construction，median 减除）
- 归一化后 RED4 均值 ≈ −0.22（波段偏差被保留）

**为什么用 median/MAD 而不用 mean/std？**

HiRISE 图像存在噪声列和仪器伪迹，MAD（Median Absolute Deviation）对异常值鲁棒。系数 1.4826 是正态一致性校正：对于 $X \sim \mathcal{N}(\mu, \sigma^2)$，$1.4826 \times E[\mathrm{MAD}] = \sigma$。

**为什么需要 `clamp_min(0.05)`？**

低对比度 patch（如平坦沙丘、均匀岩面）的 MAD 可能接近 0（例如 0.001）。若不加限制，scale = 0.0015，归一化后值域可达 ±500，导致 prior 统计（sigma ≈ 4.8，hist range ±270）完全失真，SCI 梯度校正反而破坏生成质量。`clamp_min(0.05)` 将最大放大倍率限制在 20×。

**为什么 `clamp(-10, 10)` 不丢失细节？**

实测 prior 百分位数（p5/p95）：IR10 在 ±0.20 以内，RED4 在 −0.66 到 +0.04 以内——90% 的像素远离 ±10 边界。clamp 实际上**从不截断真实地形细节**，只挡住因 scale 极小产生的无意义极端值。IR10 与 RED4 使用同一 scale，两者的空间结构关系（纹理、边缘、crater）在归一化后完整保留。

### DC 减除

归一化之后，额外减去 IR10 归一化后的空间均值 $\mathrm{dc} = \overline{\text{ir\_norm}}$：

$$\text{ir\_final} = \text{ir\_norm} - \mathrm{dc} \qquad \text{red\_final} = \text{red\_norm} - \mathrm{dc}$$

这确保每个场景的 IR10 通道均值精确为 0（by construction），使全局先验 $\mu_{\text{IR}} \approx 0$ 对所有场景有效，SCI 以标准 $\lambda$ 值即可正常工作。dc 保存在 `norm_stats[2]` 中，反归一化时加回：

$$x_{\text{raw}} = (x_{\text{norm}} + \mathrm{dc}) \times \text{scale} + \text{center}$$

CM-Diff 用的是全局 min-max 归一化到 [−1, 1]——对所有场景、所有像素找全局最大值和最小值，然后线性映射。这样归一化后两个通道的值域自然落在 [−1, 1]，全局 μ 自然接近 0（大约在值域中点附近）。

CM-Diff 不需要 per-scene 归一化，因为 SAR + 光学数据的动态范围比 HiRISE 更可预测。HiRISE 场景之间亮度差异极大（极地冰面 vs 暗玄武岩），全局 min-max 会被极端值主导，所以我们用 per-scene median/MAD，然后需要额外的 Method B 来稳定 prior。dc校准是对 per-scene 归一化的补丁，让两个通道的全局 prior 都变成稳定常数。IR=0 只是约定，不是目标。

**实际数据范围（after fix）：**
- IR10 均值 = 0（by construction），典型 p5/p95 ≈ ±0.20
- RED4 集中于 −0.22 附近，p5/p95 ≈ [−0.66, +0.04]

### 为什么 FM 实验中加入 `norm_gain` 可能有帮助

当前 FM 使用 rectified flow：

```
x_t = (1 - t) * noise + t * x_target
v_target = x_target - noise
```

这里 `noise` 是标准高斯，典型尺度约为 1。但开启 dc 后，HiRISE 归一化图像的有效动态范围明显更窄：IR10 的 p5/p95 只有约 ±0.20，RED4 也主要集中在小幅偏移附近。这会让 clean endpoint `x_target` 的尺度远小于 noise endpoint。

加入固定 gain：

```
x_model = norm_gain * (x_robust - dc)
x_raw   = (x_model / norm_gain + dc) * scale + center
```

它不改变物理反归一化结果，只改变模型看到的数值尺度。对 FM 来说，`norm_gain=4` 有几个潜在优势：

1. **让数据端点更接近高斯端点的尺度。**  
   如果 `x_target` 的标准差只有 0.2 左右，而 `noise` 的标准差是 1，flow 的大部分速度会被“从 N(0,1) 收缩到很窄的数据流形”主导。乘以 gain 后，clean endpoint 的方差更接近 noise endpoint，ODE 轨迹更均衡。

2. **让网络更容易学习可见结构。**  
   当目标图像数值幅度很小，结构性差异会被压到很小的 residual 里。gain 把 IR/RED 的局部纹理、边缘和 band offset 放大到 U-Net 更容易使用的范围，同时最后仍可用 `norm_gain` 精确反归一化。

3. **让 SGI 梯度有更合理的数值尺度。**  
   SGI 的 mean、sigma、histogram loss 都在模型归一化空间里计算。如果数据分布太窄，统计损失和 `x_t` 上的梯度可能相对很弱，或者需要非常大的 lambda 才能产生可见影响。gain 后，prior 的 `sigma` 和 histogram support 同步放大，SGI 的梯度尺度更接近 FM velocity 的尺度。

4. **避免把 lambda 调参和归一化尺度混在一起。**  
   没有 gain 时，如果 SGI 不起作用，可能是方法无效，也可能只是数据尺度太小导致 lambda 不在合适量级。固定 gain 后，lambda 的含义更稳定，更容易比较 `lambda_sgi_scl/ccl`、velocity guidance 和 PnP-style re-interpolation。

注意：`norm_gain` 必须在 training、prior computation、evaluation、inference 中保持一致。若训练使用 `dc=True, norm_gain=4`，则 SGI prior 也必须来自同一空间，例如 `prior_red_dc_g4.pt` 和 `prior_ir_dc_g4.pt`。旧 DDPM baseline 默认仍应使用 `dc=False, norm_gain=1`，对应 `prior_red.pt` 和 `prior_ir.pt`。

### DC 减除的代价与局限

#### 1. 归一化参数的不对称性（部署层面）

dc 和 center、scale 都由 **IR10 的像素值计算**，与 RED4 无关：

```
center = median(IR10)
scale  = 1.4826 × MAD(IR10)
dc     = mean((IR10 - center) / scale)
```

这导致两个推理方向的可行性完全不同：

| 方向 | 是否能在部署时计算 dc / scale / center |
|------|--------------------------------------|
| IR→RED（source = IR10）| ✓ 所有统计量来自输入，自洽 |
| RED→IR（source = RED4）| ✗ 需要目标 IR10 才能计算——即需要知道答案才能做归一化 |

评估时两个方向都有 ground-truth，所以 `norm_stats` 可以直接从 batch 里读出来；但真实部署只给 RED4 时，RED→IR 的反归一化无法完成。

#### 2. SCI 对异常场景失效

SCI 把预测拉向全局 prior 的 μ。开启 dc 之后 μ_IR = 0 精确等于每个场景的实际均值，方向 0（IR→RED）的 SCI 目标是全局 μ_RED = −0.597。

问题在于：不是每个场景的 RED4 均值都接近 −0.597。对于那些 RED4 均值偏差较大的场景（比如高亮冰面，均值 ≈ +0.2），SCI 会把预测推向错误方向，使 MSE 随 λ 增大而增大（而不是减小）。

这也解释了为什么在部分场景中 **λ = 0 表现最好**：SCI 的全局 prior 假设该场景"接近平均水平"，但这个假设对异常场景是错的。

#### 3. 模型不学习动态亮度映射

开启 dc 之后，RED4 的场景均值从不同的随机值被统一校正到约 −0.597（固定常数）。模型学到的是这个**固定的 band offset**，而不是"给定这张 IR10，该场景的 RED4 应该有多亮"。

关闭 dc（`--no_dc`）则把亮度预测的责任交给网络：RED4 的均值随场景变化，网络必须从 IR10 的输入推断正确的输出亮度。这使 RED→IR 方向在真实部署时可行（无需目标域统计量），但训练难度更高，且 SCI 需要重新校准（或直接关闭）。

---

## 7. 本项目的使用方式

本项目（HiRISE CM-Diff）只使用以下通道：

| 通道 | 用途 |
|---|---|
| IR10 | 方向 0 的 source，方向 1 的 target；提供归一化统计量 |
| RED4 | 方向 0 的 target，方向 1 的 source |
| BG12 | **不使用** |
| RED3, RED5 | **不使用** |
| x_meta | 不使用 |

UNet 输入 shape：`[B, 3, 256, 256]` = 1ch noisy target + 1ch source + 1ch Sobel edge
