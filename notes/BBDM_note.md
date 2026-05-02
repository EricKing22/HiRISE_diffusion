# BBDM: Model Notes

**论文：** *BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models*
**作者：** Bo Li, Kaitao Xue, Bin Liu（南昌航空大学）；Yu-Kun Lai（卡迪夫大学）
**发表：** CVPR 2023

---

## 1. 问题定义与核心思想

给定配对图像 $(x, y)$，$x$ 来自域 A，$y$ 来自域 B，目标是学习映射 $x \mapsto \hat{y}$。

**核心创新：** 不把 $y$ 当作条件注入 U-Net，而是把 $y$ 设为扩散过程的**终点**。正向过程从 $x_0 = x$（源域）"走到" $x_T = y$（目标域）；逆向过程从 $y$ 出发，一步步去噪还原 $x_0$。这个过程叫**布朗桥（Brownian Bridge）**——两端固定，中间随机。

---

## 2. BBDM vs 标准 DDPM：核心区别

理解 BBDM 最重要的是看它和标准 DDPM 在哪三点上本质不同：

| 方面 | 标准 DDPM（生成） | 条件 DDPM（翻译） | **BBDM** |
|------|-----------------|-----------------|---------|
| 正向过程终点 $x_T$ | 纯高斯噪声 $\mathcal{N}(0,I)$ | 纯高斯噪声 $\mathcal{N}(0,I)$ | **目标域图像 $y$** |
| 逆向起点 | $x_T \sim \mathcal{N}(0,I)$（随机采样） | $x_T \sim \mathcal{N}(0,I)$（随机采样） | **$x_T = y$（确定性，直接用目标图）** |
| 条件 $y$ 如何用 | 不存在 | 每步注入 U-Net（$\epsilon_\theta(x_t, y, t)$） | **仅设置起点，不注入 U-Net**（$\epsilon_\theta(x_t, t)$） |
| 方差调度 | 单调递增 $\beta_t \in [0.0001, 0.02]$ | 单调递增 | **先增后减**（中间峰值，两端为 0） |
| 模型学什么 | $p(x_0)$ 无条件分布 | $p(x_0 \mid y)$ 条件分布（无理论保证） | **$p(x_0 \mid y)$ 直接建桥**（两端钉死，有保证） |
| 多样性来源 | 随机 $x_T$ | 随机 $x_T$ | **中间步的随机噪声**（两端 $x_0, x_T$ 均确定） |

**为什么条件 DDPM"无理论保证"？**

条件 DDPM 的正向过程并不知道 $y$ 的存在，仍然走向 $\mathcal{N}(0,I)$。训练时用的目标是：

$$\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, y, t)\|^2]$$

但 $p(x_t \mid y)$ 从未显式出现在训练目标里。网络学会了"给定噪声图 $x_t$ 和参考图 $y$，预测噪声"，但无法保证逆向过程最终输出的分布就是 $p(x_0 \mid y)$。

BBDM 的解决方案：**让正向过程本身就从 $x_0$ 走到 $y$，两端都被钉死**，不再依赖条件注入机制。

---

## 3. 数学原理

### 3.1 布朗桥过程

标准布朗桥：从 $x_0$ 出发、$x_T = y$ 结束的连续随机过程，中间每步分布为：

$$p(x_t \mid x_0, x_T) = \mathcal{N}\!\left(\left(1 - \frac{t}{T}\right)x_0 + \frac{t}{T}x_T,\; \frac{t(T-t)}{T}I\right)$$

两端方差为 0，中间（$t = T/2$）方差最大为 $T/4$。但这个原始方差会随 $T$ 增大而爆炸，BBDM 做了修正。

### 3.2 前向过程（正向扩散）

令 $m_t = t/T$，BBDM 的前向过程边缘分布为：

$$q_{BB}(x_t \mid x_0, y) = \mathcal{N}\!\big(x_t;\;(1-m_t)x_0 + m_t y,\;\delta_t I\big) \tag{1}$$

**关键：** 均值是 $x_0$ 和 $y$ 的线性插值，权重随时间线性变化：
- $t=0$：均值 $= x_0$，方差 $= 0$（从源图出发）
- $t=T$：均值 $= y$，方差 $= 0$（到达目标图）
- $t=T/2$：方差最大，等于 $\delta_{T/2}$（最"模糊"的中间步）

**方差调度** $\delta_t$（BBDM 的重新设计，替代标准 $\beta_t$）：

$$\delta_t = 2s(m_t - m_t^2) \tag{2}$$

- $s=1$ 时，$\delta_{T/2} = 1/2$（最大方差）
- 两端 $\delta_0 = \delta_T = 0$
- 与标准 DDPM 的单调递增 $\beta_t$ 完全不同，这里是**钟形**

参数 $s$ 控制多样性：$s$ 越大，中间步噪声越多，采样结果越多样但越不保真。

实际用直接采样公式（等价于 Eq.1）：

$$x_t = (1-m_t)x_0 + m_t y + \sqrt{\delta_t}\;\epsilon, \quad \epsilon \sim \mathcal{N}(0,I) \tag{3}$$

### 3.3 转移概率 $q_{BB}(x_t \mid x_{t-1}, y)$

正向逐步转移（相邻步之间）：

$$q_{BB}(x_t \mid x_{t-1}, y) = \mathcal{N}\!\left(x_t;\;\frac{1-m_t}{1-m_{t-1}}x_{t-1} + \left(m_t - \frac{1-m_t}{1-m_{t-1}}m_{t-1}\right)y,\;\delta_{t|t-1}I\right) \tag{4}$$

其中：

$$\delta_{t|t-1} = \delta_t - \delta_{t-1}\frac{(1-m_t)^2}{(1-m_{t-1})^2} \tag{5}$$

当 $t = T$ 时，$m_T = 1$，所以 $x_T = y$（确定性到达目标）。

### 3.4 逆向过程

逆向过程从 $x_T = y$ 出发，每步预测 $x_{t-1}$：

$$p_\theta(x_{t-1} \mid x_t, y) = \mathcal{N}(x_{t-1};\;\mu_\theta(x_t, t),\;\tilde{\delta}_t I) \tag{6}$$

**关键：$y$ 不作为 U-Net 的输入**，只是设置了 $x_T = y$。

用 Bayes 定理推导后验均值（类比 DDPM 的推导）：

$$\tilde{\mu}_t(x_t, x_0, y) = c_{xt}x_t + c_{yt}y + c_{\epsilon t}\left(m_t(y - x_0) + \sqrt{\delta_t}\epsilon\right) \tag{7}$$

由于 $x_0$ 推理时未知，训练网络预测噪声 $\epsilon_\theta(x_t, t)$，再用重参数化代入 Eq.7：

$$\mu_\theta(x_t, y, t) = c_{xt}x_t + c_{yt}y - c_{\epsilon t}\epsilon_\theta(x_t, t) \tag{8}$$

**注意：** $y$ 出现在 $\mu$ 的解析表达式中，但**只是数值参与计算，不是 U-Net 的输入**。

逆向方差（解析解，不学习）：

$$\tilde{\delta}_t = \frac{\delta_{t|t-1} \cdot \delta_{t-1}}{\delta_t} \tag{9}$$

### 3.5 训练目标

ELBO 化简后，等价于最小化：

$$\mathcal{L} = \mathbb{E}_{x_0, y, \epsilon}\!\left[c_{\epsilon t}\left\|m_t(y - x_0) + \sqrt{\delta_t}\,\epsilon - \epsilon_\theta(x_t, t)\right\|^2\right] \tag{10}$$

**直观理解：** 网络需要预测的目标是 $m_t(y - x_0) + \sqrt{\delta_t}\epsilon$——这是"从 $x_0$ 到 $y$ 的方向"与"随机噪声"的叠加，而不只是纯噪声 $\epsilon$。随着 $t$ 增大（接近 $T$），$m_t$ 增大，"走向 $y$ 的方向"占比越来越大。

与标准 DDPM 训练目标对比：

```
标准 DDPM：E[|| ε - ε_θ(x_t, t) ||²]         ← 只预测加进去的噪声
条件 DDPM：E[|| ε - ε_θ(x_t, y, t) ||²]       ← 同上，但网络额外看到 y
BBDM：     E[|| m_t(y-x_0) + √δ_t·ε - ε_θ(x_t, t) ||²]  ← 预测噪声+方向混合
```

---

## 4. 采样算法

**训练（Algorithm 1）：**
```
repeat:
  取配对数据 (x_0, y)
  随机时间步 t ~ Uniform(1,...,T)
  采样噪声 ε ~ N(0,I)
  正向扩散：x_t = (1-m_t)x_0 + m_t·y + √δ_t·ε
  梯度下降：最小化 ||m_t(y-x_0) + √δ_t·ε - ε_θ(x_t, t)||²
until 收敛
```

**推理（Algorithm 2）：**
```
设 x_T = y  ← 直接用目标域图像作为起点
for t = T, T-1, ..., 1:
  z ~ N(0,I)（t>1 时），否则 z=0
  x_{t-1} = c_xt · x_t + c_yt · y - c_εt · ε_θ(x_t, t) + √δ̃_t · z
return x_0
```

**可加速采样**（类 DDIM）：用长度为 $S$ 的子序列 $\{\tau_1,...,\tau_S\}$ 代替全部 $T$ 步，$S=200$ 时效果接近 $S=1000$。

---

## 5. 整体架构

```
输入图像 I_A (域A)
    ↓
VQGAN 编码器 L_A     ←— 预训练，训练时冻结
    ↓
L_A = x_0  (源域潜变量)

目标图像 I_B (域B)
    ↓
VQGAN 编码器 L_B     ←— 预训练，训练时冻结
    ↓
L_B = y    (目标域潜变量) = 布朗桥终点

BBDM（布朗桥扩散，在潜空间运行）：
  训练：x_0 → (布朗桥前向) → x_T = y
  推理：x_T = y → (逐步去噪) → x̂_0

x̂_0（预测的源域潜变量 = 翻译结果潜变量）
    ↓
VQGAN 解码器 L_B^{-1}  ←— 用目标域的解码器
    ↓
输出图像 Î_{A→B}
```

**关键实现细节：**
- 使用与 LDM 相同的预训练 VQGAN（下采样因子 $f=4$，$256\times256 \to 64\times64$ 潜空间）
- 不同域有**独立的** VQGAN（$\mathcal{L}_A$ 和 $\mathcal{L}_B$）——域之间编码器/解码器不共享
- 扩散 U-Net 结构与 LDM 完全相同，只改变了扩散过程的定义
- U-Net 输入：只有 $x_t$（当前潜变量）和时间步 $t$，**不接收 $y$ 或任何额外条件**

---

## 6. BBDM vs DDPM：详细对比表

### 6.1 正向过程

| | 标准 DDPM | BBDM |
|---|---|---|
| 公式 | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | $x_t = (1-m_t)x_0 + m_t y + \sqrt{\delta_t}\epsilon$ |
| $x_T$ 分布 | $\approx \mathcal{N}(0,I)$ | $= y$（确定，方差为0） |
| 方差调度形状 | 单调递增 | 钟形（$t=T/2$ 峰值） |
| 需要 $y$ 参与？ | 否 | 是（$y$ 是终点） |

### 6.2 逆向过程

| | 标准 DDPM | 条件 DDPM | BBDM |
|---|---|---|---|
| 起点 | $x_T \sim \mathcal{N}(0,I)$ | $x_T \sim \mathcal{N}(0,I)$ | $x_T = y$（确定） |
| U-Net 输入 | $(x_t, t)$ | $(x_t, y, t)$ | $(x_t, t)$ |
| $y$ 出现在哪 | 不存在 | U-Net 输入（每步） | $\mu$ 的解析表达式（$c_{yt}y$ 项） |
| 采样公式 | $x_{t-1} = \frac{1}{\sqrt\alpha_t}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon) + \sqrt\beta_t z$ | 同上 | $x_{t-1} = c_{xt}x_t + c_{yt}y - c_{\epsilon t}\epsilon_\theta + \sqrt{\tilde\delta_t}z$ |
| 方差 $\tilde\sigma$ | $\beta_t$（固定，从调度表） | $\beta_t$ | $\tilde\delta_t = \frac{\delta_{t\|t-1}\delta_{t-1}}{\delta_t}$（固定，从公式） |

### 6.3 训练目标

| | 训练目标 | 注释 |
|---|---|---|
| 标准 DDPM | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ | 只预测加入的高斯噪声 |
| 条件 DDPM | $\|\epsilon - \epsilon_\theta(x_t, y, t)\|^2$ | 同上，$y$ 辅助 |
| BBDM | $\|m_t(y-x_0) + \sqrt{\delta_t}\epsilon - \epsilon_\theta(x_t, t)\|^2$ | 预测"方向+噪声"混合信号 |

**理解 BBDM 训练目标的关键：**

$m_t(y-x_0)$ 是从 $x_0$ 指向 $y$ 的**方向向量**，乘以时间权重 $m_t$。网络需要同时预测这个方向和随机噪声 $\epsilon$，两者都会随时间变化：
- $t$ 很小（接近 0）：$m_t \approx 0$，主要预测噪声，类似标准 DDPM
- $t$ 很大（接近 $T$）：$m_t \approx 1$，主要预测 $y - x_0$，学习域间的整体差异

---

## 7. 实验结果

### 7.1 数据集与任务

| 数据集 | 翻译任务 | 规模 |
|--------|---------|------|
| CelebAMask-HQ | 语义分割图 → 人脸照片 | 30K 配对图像 |
| edges2shoes | 边缘线稿 → 鞋子照片 | 50K 配对图像 |
| edges2handbags | 边缘线稿 → 手袋照片 | 138K 配对图像 |
| faces2comics | 人脸照片 → 漫画风格 | 自建数据集 |

### 7.2 定量结果（CelebAMask-HQ）

| 方法 | FID ↓ | LPIPS ↓ | Diversity ↑ |
|------|--------|---------|------------|
| Pix2Pix | 56.997 | 0.431 | 0（确定性） |
| CycleGAN | 78.234 | 0.490 | 0（确定性） |
| DRIT++ | 77.794 | 0.431 | 35.759 |
| SPADE | 44.171 | 0.376 | 0（确定性） |
| OASIS | 27.751 | 0.384 | 39.662 |
| CDE（条件扩散） | 24.404 | 0.414 | 50.278 |
| LDM（条件扩散） | 22.816 | 0.371 | 20.304 |
| **BBDM（ours）** | **21.350** | **0.370** | 29.859 |

Diversity = 对同一输入生成 5 个样本，计算每像素标准差的均值。

### 7.3 关键消融：采样步数 vs 效果

| 采样步数 | FID ↓ | LPIPS ↓ | Diversity ↑ |
|---------|--------|---------|------------|
| 20 步 | 33.409 | 0.362 | 17.587 |
| 50 步 | 25.188 | 0.372 | 23.191 |
| 100 步 | 23.503 | 0.378 | 26.157 |
| **200 步** | **21.350** | **0.370** | **29.859** |
| 1000 步 | 21.348 | 0.375 | 29.924 |

200 步和 1000 步几乎一样——推荐默认使用 200 步。

### 7.4 多样性参数 $s$ 的影响

| $s$ | FID ↓ | LPIPS ↓ | Diversity ↑ |
|-----|--------|---------|------------|
| 0.5 | 22.627 | 0.387 | 27.791 |
| **1**（默认） | **21.350** | **0.370** | 29.859 |
| 2 | 23.278 | 0.380 | 37.063 |
| 4 | 24.490 | 0.384 | 39.573 |

$s$ 增大 → 多样性增加，但质量下降。

---

## 8. 与 HiRISE 项目的关联

### 8.1 BBDM 相较 CM-Diff 的优势（对于 HiRISE）

| 方面 | CM-Diff（当前方案） | BBDM |
|------|-------------------|------|
| 双向翻译 | ✅ 单网络同时训练两方向 | ❌ 每方向需单独训练 |
| 条件机制 | 条件 DDPM（每步注入 CFC） | **布朗桥**（无需注入，天然保证） |
| 对低域差任务 | 条件注入可能过拟合到 source | **桥式扩散更自然**（IR10↔RED4 域差小）|
| 推理步数 | 1000 步（SCI 需要中间 $\tilde x_0$） | **200 步足够** |
| 分布漂移问题 | 存在（需 SCI 修正） | **两端确定，无分布漂移** |

### 8.2 为什么 IR10↔RED4 适合 BBDM？

IR10 和 RED4 是同一卫星同一次观测的相邻光谱 CCD，两者：
- 空间完全对齐（不需要配准）
- 统计分布接近（均值差约 0.6σ）
- 域差远小于 CelebAMask-HQ 的语义图 → 人脸照片

这正是 BBDM 的最优场景：**域差越小，布朗桥中间步的噪声越少，收敛越快，效果越好**。

### 8.3 潜在实验设计（消融建议）

1. **BBDM（单方向）** vs **CM-Diff（双向）**：在相同 HiRISE 数据上比较
   - BBDM 单向 IR10→RED4 作为基线
   - 测试 BBDM 是否比 CM-Diff 收敛更快、分布漂移更少
2. **去掉 SCI**：若换成 BBDM，则 SCI（直方图修正）理论上不再必要——验证是否确实不需要
3. **差分图 + BBDM**：将 PanDiff 的差分图思路（预测 $\Delta = RED4 - IR10$）应用到 BBDM 的 $y$ 设计上

### 8.4 局限与注意事项

- BBDM 要求**配对数据**（HiRISE 满足）
- BBDM **不支持单网络双向联合训练**（IR10→RED4 和 RED4→IR10 需要两个独立模型）——这丢失了 CM-Diff 的 BDT 优势
- 原论文用 VQGAN 潜空间；HiRISE 可以直接在像素空间（或用更小 patch）运行 BBDM，不需要预训练 VQGAN
