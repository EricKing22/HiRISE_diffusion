# CM-Diff: Model Notes

**论文：** *CM-Diff: A Single Generative Network for Bidirectional Cross-Modality Translation Diffusion Model Between Infrared and Visible Images*
**arXiv：** 2503.09514v2 (Aug 2025)

---

## 1. 问题定义

给定配对的红外（IR）和可见光（VIS）图像，训练**单个模型**实现双向翻译：
- IR → VIS
- VIS → IR

已有方法要么用两个独立网络（加循环一致性损失），要么对每个方向单独训练。CM-Diff 用单个 U-Net + DDPM 同时实现两个方向，不需要循环一致性。

---

## 2. 背景：DDPM

### 2.1 正向过程（加噪）

标准 DDPM 正向过程逐步向干净图像 $x_0$ 添加高斯噪声，共 $T$ 步：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

其中：
- $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$ — 累积噪声调度乘积
- $\beta_t$ 从 $\beta_1 = 0.0001$ 线性增长到 $\beta_T$（CM-Diff 用 $0.01$，标准 DDPM 用 $0.02$）
- $\epsilon \sim \mathcal{N}(0, I)$ — 标准高斯噪声

### 2.2 噪声调度表

训练前预先定义好的固定常数表，规定每个时间步注入多少噪声。**与图片内容无关，对所有图片一样：**

| 时间步 | $\beta_t$ | 噪声量 |
|---|---|---|
| $t=1$ | $0.0001$ | 几乎不加噪声 |
| $t=500$ | $\approx 0.005$ | 中等噪声 |
| $t=1000$ | $0.01$ | 最多噪声 |

### 2.3 逆向过程（采样）

U-Net 不直接预测 $\mu$，而是**预测噪声 $\epsilon$**，再由公式推导出 $\mu$：

$$\hat{\epsilon} = \text{U-Net}(x_t,\, t) \quad \in \mathbb{R}^{H \times W \times C}$$

$$\mu = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\hat{\epsilon}\right)$$

每步采样：

$$x_{t-1} = \mu + \sqrt{\beta_t}\,\epsilon_{\text{new}}, \qquad \epsilon_{\text{new}} \sim \mathcal{N}(0, I)$$

**关于维度：**
- $\mu \in \mathbb{R}^{H \times W \times C}$：每个像素每个通道各有一个预测均值
- $\sqrt{\beta_t}$：**标量**，全图所有像素所有通道共用同一个标准差（从调度表查 $t$ 对应的值）
- $\epsilon_{\text{new}} \in \mathbb{R}^{H \times W \times C}$：每个位置独立采样的随机扰动

$$x_{t-1}[i,j,c] = \underbrace{\mu[i,j,c]}_{\text{网络预测的"应该在哪"}} + \underbrace{\sqrt{\beta_t}}_{\text{调度表固定噪声量}} \cdot \underbrace{\epsilon_{\text{new}}[i,j,c]}_{\text{随机扰动}}$$

**注意：** 最后一步 $t=1$ 时不加随机噪声，直接输出 $\mu$ 作为最终图像。

### 2.4 原始 DDPM vs Improved DDPM

| | 原始 DDPM (Ho 2020) | Improved DDPM (Nichol 2021) |
|---|---|---|
| U-Net 输出 | 1个头：$\hat{\epsilon} \in \mathbb{R}^{H \times W \times C}$ | 2个头：$\hat{\epsilon}$ + $v \in \mathbb{R}^{H \times W \times C}$ |
| 方差来源 | 直接查调度表 $\beta_t$（固定标量） | 网络学习，每步插值出 $\sigma_t$ |
| 方差公式 | $\sigma_t^2 = \beta_t$ | $\sigma_t^2 = \exp\!\left(v \log\beta_t + (1-v)\log\tilde{\beta}_t\right)$ |
| 采样步数 | 需要多步（1000步）才有好质量 | 可以更少步达到同等质量 |
| 训练难度 | 简单稳定 | 需要混合损失函数，稍复杂 |

其中 Improved DDPM 的 $v$ 是 U-Net 第二个输出头预测的插值系数，$\tilde{\beta}_t$ 是方差的下界：

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\,\beta_t$$

### 2.5 CM-Diff 使用原始 DDPM 的原因与代价

CM-Diff **沿用原始 DDPM，固定方差，U-Net 只预测 $\epsilon$**，论文未解释原因。

**合理推测：** BDT + CFC + SCI 三个新模块已经使训练足够复杂，选择固定方差是为了控制变量和保持训练稳定性。

**代价：** 固定方差对 IR 和 VIS 两个统计特性差异很大的模态使用同一套 $\beta_t$，无法自适应。这正是论文需要引入 SCI 来修正分布偏差的根本原因之一——**SCI 在推理时弥补了固定方差带来的分布漂移**。

---

## 3. 整体架构

CM-Diff 由**单个 U-Net 主干**（$\mathcal{Z}_N$）和三个核心组件构成：

1. **TDG（Translation Direction Guidance）** — 通道位置隐式编码 + 方向标签嵌入，告诉模型翻译方向
2. **CFC（Cross-Modality Feature Control）** — 跨模态特征控制，持续注入源域语义特征
3. **SCI（Statistical Constraint Inference）** — 推理时统计约束，修正颜色分布漂移

另有两个轻量级模态编码器（独立于 $\mathcal{Z}_N$）：
- $\mathcal{Z}^\text{IR}_E$：提取 IR 图像特征
- $\mathcal{Z}^\text{VIS}_E$：提取 VIS 图像特征

以及预训练的边缘检测网络 **DexiNed**（推理时冻结）。

---

## 4. Bidirectional Diffusion Training (BDT)

### 4.1 Translation Direction Guidance (TDG)

**隐式通道位置编码：**

IR 图像原本是单通道，**复制成3通道**使 $C_\text{IR} = C_\text{VIS} = 3$，两个方向的输入均为9通道：

| 通道 | IR → VIS | VIS → IR |
|---|---|---|
| 1–3 | 加噪的 VIS 图像（去噪目标，RGB） | 加噪的 IR 图像（去噪目标，**1ch 复制成 3**） |
| 4–6 | 原始 IR 图像（源条件，**1ch 复制成 3**） | 原始 VIS 图像（源条件，RGB） |
| 7–9 | $E^\text{IR} = \text{DexiNed}(x^\text{IR})$ 边缘图 | $E^\text{VIS} = \text{DexiNed}(x^\text{VIS})$ 边缘图 |

拼接公式：

$$\mathcal{Z}^{t}_{\text{IR}\to\text{VIS}} = q(x^\text{VIS}_t \mid x^\text{VIS}_0) \oplus x^\text{IR}_0 \oplus E^\text{IR}$$

$$\mathcal{Z}^{t}_{\text{VIS}\to\text{IR}} = q(x^\text{IR}_t \mid x^\text{IR}_0) \oplus x^\text{VIS}_0 \oplus E^\text{VIS}$$

目标模态数据**始终在通道1–3**，U-Net 通过通道位置隐式感知当前翻译方向。两个方向通道数完全相同（均为9通道），因此 **U-Net 结构无需任何改动**。

对于IR→VIS，边缘图可能是多尺度输出，也可能是单通道复制，论文未明确说明。

**显式方向标签嵌入：**

方向标签 $C$（0或1）通过可学习的 Embedding 查找表 $\tau_\phi$ 映射为向量：

$$J = \tau_\phi(C)$$

$\tau_\phi$ 等价于 `nn.Embedding(num_classes=2, dim=d)`，不是对标量做线性投影：

```
C=0 (IR→VIS) → 查表第0行 → [a1, a2, ..., ad]   （独立可学习向量）
C=1 (VIS→IR) → 查表第1行 → [b1, b2, ..., bd]   （独立可学习向量）
```

**J 注入 U-Net（TDG 完整流程）：**

时间步 $t$ 和方向标签 $C$ 分别编码后相加，通过 AdaGN 注入**每个 ResBlock**：

```
t（整数 1~T）
  → 正弦位置编码（固定，不可学习）→ 高维向量
  → 两层 MLP + SiLU（可学习）
  → emb_t  [d 维]

C（0 或 1）
  → nn.Embedding 查找表（可学习）
  → emb_J  [d 维]

combined = emb_t + emb_J   （直接相加）

每个 ResBlock：
  → GroupNorm(F)
  → [scale, shift] = Linear(combined)
  → 输出 = scale * GroupNorm(F) + shift   （AdaGN）
```

网络在每一层都同时知道当前去噪进度（来自 $t$）和翻译目标模态（来自 $J$）。消融实验中去掉 $J$ 后 PSNR 下降 7%。

### 4.2 Cross-Modality Feature Control (CFC)

**动机：** 源图虽然已拼接在输入通道 4–6，但随着 U-Net 层层下采样，这个信息会被稀释。CFC 在多个分辨率层级上持续注入源模态语义特征，确保去噪过程始终受到源图控制。

**两个独立的模态编码器（只有编码器部分）：**
- $\mathcal{Z}^\text{IR}_E$：提取 IR 图像特征（用于 IR → VIS 方向）
- $\mathcal{Z}^\text{VIS}_E$：提取 VIS 图像特征（用于 VIS → IR 方向）

U-Net 采用 "CNN + Attention" 混合策略：高分辨率层（$256{\times}256$、$128{\times}128$）只用卷积，低分辨率层（$32{\times}32$、$16{\times}16$、$8{\times}8$）加 Attention Block。CFC **只在 Attention Block 处注入**（编码器和解码器各三处，共六处）。

**下标含义：**
- $n$：U-Net 层级（$n=1$ 对应 $32{\times}32$，$n=2$ 对应 $16{\times}16$，$n=3$ 对应 $8{\times}8$）
- $i$：同一层内第 $i$ 个 attention block

**如何修改 Attention Block：**

标准 Self-Attention 的 Q/K/V 全来自同一特征图，CFC 将 K/V 改为来自模态编码器：

```
原始 Self-Attention：
  F_g → Conv_q → Q
  F_g → Conv_k → K
  F_g → Conv_v → V
  → softmax(Q*K^T / sqrt(d)) * V → FC → + F_g

CFC Cross-Attention：
  F_g → Conv_q → Q          （主U-Net噪声特征图）
  F_d → Conv_kv → K, V      （模态编码器源图特征）
  → Cross-Attention(Q,K,V) → FC → + F_g
```

模态编码器在三个分辨率各产生一个特征图，解码器复用编码器阶段已计算的 $F_d$：

```
源图 [256×256, 3ch]
  → 模态编码器 layer1 → F_d [32×32]  → 注入编码器+解码器 32×32 层
  → 模态编码器 layer2 → F_d [16×16]  → 注入编码器+解码器 16×16 层
  → 模态编码器 layer3 → F_d [8×8]    → 注入编码器+解码器 8×8  层
```

**Cross-Attention 公式：**

$$Q^i_n = \text{Conv}_q\!\left(F^n_{g_i}\right), \qquad K^i_n,\, V^i_n = \text{Conv}_{kv}\!\left(F^n_{d_i}\right)$$

$$F^n_{g_{i+1}} = F^n_{g_i} + \text{softmax}\!\left(\frac{Q^i_n \cdot (K^i_n)^\top}{\sqrt{d}}\right) V^i_n$$

**维度与 Flatten 过程（以 $32{\times}32$，$C=256$，$d=64$ 为例）：**

```
F_gi  [32, 32, 256] → Conv_q  → Q [32, 32, 64]
F_di  [32, 32, 256] → Conv_kv → [32, 32, 128] split → K [32,32,64], V [32,32,64]

Flatten → Q,K,V: [1024, 64]
MHA: softmax(Q*K^T / sqrt(d_head)) * V → [1024, d] → reshape [32, 32, d]
残差连接加回 F_gi
```

### 4.3 训练损失

**所有损失函数总览：**

| 损失 | 阶段 | 作用 | 本质 | 参数 |
|---|---|---|---|---|
| $\mathcal{L}_{\text{IR}\to\text{VIS}}$ | 训练 | 学习 VIS 数据分布 | 噪声预测 MSE | $\lambda=1.0$ |
| $\mathcal{L}_{\text{VIS}\to\text{IR}}$ | 训练 | 学习 IR 数据分布 | 噪声预测 MSE | $\lambda=1.0$ |
| $\mathcal{L}_\text{joint}$ | 训练 | 两个方向梯度叠加，单模型同时学两个分布 | 加权求和 | $\lambda_{1,2}=1.0$ |
| $\mathcal{L}_\text{scl}$ | 推理 | 对齐生成图的全局均值和标准差至目标域 | L1 统计距离 | $\lambda=20.0$ |
| $\mathcal{L}_\text{ccl}$ | 推理 | 对齐生成图的直方图形状至目标域 | $\chi^2$ 直方图距离 | $\lambda=20.0$ |
| $\mathcal{L}_\text{cons}$ | 推理 | 每步去噪时修正采样均值 | $\mathcal{L}_\text{scl}+\mathcal{L}_\text{ccl}$ 加权 | — |

**训练 vs 推理损失的本质区别：**

```
训练损失（L_joint）：
  backward() → 更新 U-Net 参数 phi
  目的：让模型学会预测噪声（学习数据分布）

推理损失（L_cons）：
  backward() → 只读取 x_0_tilde 的梯度，不更新任何参数
  目的：在每步采样时把生成轨迹推向目标域统计分布
```

---

#### $\mathcal{L}_{\text{IR}\to\text{VIS}}$：IR → VIS 方向噪声预测损失

标准 DDPM MSE 损失，目标是让 U-Net 准确预测加入 VIS 图中的噪声：

$$\mathcal{L}_{\text{IR}\to\text{VIS}}(\phi) = \mathbb{E}_{t_1,\, x^\text{IR}_0,\, \epsilon_1}\!\left[\left\|\epsilon_1 - \epsilon_\phi\!\left(\mathcal{Z}^{t_1}_{\text{IR}\to\text{VIS}},\, J^\text{VIS},\, t_1\right)\right\|_2^2\right]$$

**输入：**
- $\mathcal{Z}^{t_1}_{\text{IR}\to\text{VIS}}$：9通道拼接输入（加噪VIS图 + 原始IR图 + IR边缘图）
- $J^\text{VIS} = \tau_\phi(C^\text{VIS})$：IR→VIS 方向标签嵌入
- $t_1 \sim \text{Uniform}(\{1,\ldots,T\})$：随机采样的时间步

**标签：** $\epsilon_1 \sim \mathcal{N}(0, I)$，正向加噪时实际使用的噪声（已知）

---

#### $\mathcal{L}_{\text{VIS}\to\text{IR}}$：VIS → IR 方向噪声预测损失

结构与上完全对称，目标是让 U-Net 准确预测加入 IR 图中的噪声：

$$\mathcal{L}_{\text{VIS}\to\text{IR}}(\phi) = \mathbb{E}_{t_2,\, x^\text{VIS}_0,\, \epsilon_2}\!\left[\left\|\epsilon_2 - \epsilon_\phi\!\left(\mathcal{Z}^{t_2}_{\text{VIS}\to\text{IR}},\, J^\text{IR},\, t_2\right)\right\|_2^2\right]$$

**输入：**
- $\mathcal{Z}^{t_2}_{\text{VIS}\to\text{IR}}$：9通道拼接输入（加噪IR图 + 原始VIS图 + VIS边缘图）
- $J^\text{IR} = \tau_\phi(C^\text{IR})$：VIS→IR 方向标签嵌入
- $t_2 \sim \text{Uniform}(\{1,\ldots,T\})$：独立随机采样的时间步（与 $t_1$ 无关）

**标签：** $\epsilon_2 \sim \mathcal{N}(0, I)$，独立采样的噪声（与 $\epsilon_1$ 无关）

---

#### $\mathcal{L}_\text{joint}$：联合训练损失

$$\mathcal{L}_\text{joint}(\phi) = \lambda_{\text{IR}\to\text{VIS}}\,\mathcal{L}_{\text{IR}\to\text{VIS}} + \lambda_{\text{VIS}\to\text{IR}}\,\mathcal{L}_{\text{VIS}\to\text{IR}}$$

两个权重均为 $1.0$，实际上就是直接相加：

$$\mathcal{L}_\text{joint} = \mathcal{L}_{\text{IR}\to\text{VIS}} + \mathcal{L}_{\text{VIS}\to\text{IR}}$$

**作用：** 两个方向的梯度在同一次 `backward()` 中叠加，同时更新同一组 U-Net 参数 $\phi$，强迫单个模型同时学会 IR 和 VIS 两个数据分布。

**$t_1, t_2$ 独立采样的意义：** 同一个 batch 里两个方向可以处于不同噪声程度，增加训练多样性。

### 4.4 训练算法（Algorithm 1）

```
repeat:
  从数据集取一对图：x_VIS_0, x_IR_0
  独立采样 t1, t2  ~  Uniform({1,...,T})
  独立采样 ε1, ε2  ~  N(0, I)

  x_VIS_t1  =  √ᾱ_t1 · x_VIS_0  +  √(1−ᾱ_t1) · ε1   （正向加噪 VIS）
  x_IR_t2   =  √ᾱ_t2 · x_IR_0   +  √(1−ᾱ_t2) · ε2   （正向加噪 IR）

  J_VIS = τ_φ(C_VIS),   J_IR = τ_φ(C_IR)              （方向嵌入）
  E_VIS = DexiNed(x_VIS_0),   E_IR = DexiNed(x_IR_0)  （边缘图）

  对以下目标做梯度下降：
    λ_IR→VIS · ‖ε1 − ε_φ(Z_t1_IR→VIS, J_VIS, t1)‖²
  + λ_VIS→IR · ‖ε2 − ε_φ(Z_t2_VIS→IR, J_IR,  t2)‖²

until converged
```

---

## 5. Statistical Constraint Inference (SCI)

**动机：** BDT 训练完成后，直接用祖先采样（ancestral sampling）会产生颜色偏移，原因是逆向马尔可夫链的随机漂移。SCI 在**推理时**注入目标域统计信息修正这一问题，不需要重新训练。

### 5.1 基础逆向扩散（Baseline）

将标准 DDPM 逆向步骤改写为条件化版本，引入跨模态嵌入 $\mathcal{Z}^t$ 和方向标签 $J$：

$$x^\text{VIS}_{t-1} = \mu_\phi\!\left(\mathcal{Z}^{t_1}_{\text{IR}\to\text{VIS}},\, J^\text{VIS},\, t_1\right) + \Sigma_\phi(\cdots)\,\epsilon_1$$

$$x^\text{IR}_{t-1} = \mu_\phi\!\left(\mathcal{Z}^{t_2}_{\text{VIS}\to\text{IR}},\, J^\text{IR},\, t_2\right) + \Sigma_\phi(\cdots)\,\epsilon_2$$

相比单向 DDPM，$\mu_\phi$ 的输入额外包含了 CFC 注入的源模态特征和方向标签，去噪过程受源图语义控制。

### 5.2 前置概念：Classifier Guidance（梯度引导推理）

SCI 的梯度修正方法来自 **Classifier Guidance**（Dhariwal & Nichol 2021）框架。

**核心思想：** 标准 DDPM 从 $p(x)$ 无条件采样。若要从条件分布 $p(x \mid y)$ 采样，由贝叶斯公式：

$$\log p(x \mid y) = \log p(x) + \log p(y \mid x) + \text{const}$$

对 $x$ 求梯度，逆向步骤的均值被修正为：

$$\mu_\text{updated} = \mu + \Sigma \cdot \nabla_x \log p(y \mid x)$$

这个修正**不改变模型权重**，只是在每步采样时注入一个"推力"，把采样轨迹推向满足条件 $y$ 的区域。

**$\mu_\text{updated}$ 的推导（Dhariwal & Nichol 2021，Appendix H）：**

> **前置理解：score function 是什么**
>
> 一个分布 $p(x)$ 的 score function 定义为 $\nabla_x \log p(x)$，即对数概率关于 $x$ 的梯度。
> DDPM 的采样本质上是沿 score function 的方向移动（Langevin dynamics）：
> score 越大的地方概率越高，采样点会被"吸引"过去。
> **修改 score = 修改采样落点的分布。**

---

**Step 1：对贝叶斯公式两边取 score（对 $x$ 求梯度）**

从贝叶斯公式出发：

$$\log p(x \mid y) = \log p(x) + \log p(y \mid x) + \text{const}$$

两边对 $x$ 求梯度，const 消失，得到条件分布的 score：

$$\underbrace{\nabla_x \log p(x \mid y)}_{\text{条件 score（我们想要的）}} = \underbrace{\nabla_x \log p(x)}_{\text{无条件 score（DDPM 已有）}} + \underbrace{\nabla_x \log p(y \mid x)}_{g\text{，classifier 梯度}}$$

含义：要让采样落在"满足条件 $y$"的区域，只需在原来的 score 上加一项 $g$，$g$ 来自"条件有多可能成立"对 $x$ 的梯度。

---

**Step 2：DDPM 反向步骤的无条件分布是高斯，写出其 score**

DDPM 的每一步反向过程是：

$$p(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\; \mu,\; \Sigma)$$

高斯分布的 log 为 $\log \mathcal{N}(x;\mu,\Sigma) = -\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu) + \text{const}$，对 $x$ 求梯度：

$$\nabla_x \log \mathcal{N}(x;\,\mu,\Sigma) = -\Sigma^{-1}(x - \mu)$$

这是"无条件 score"，对应 Step 1 左边的第一项。

---

**Step 3：把 $g$ 加到 score 上，得到条件 score**

由 Step 1，条件 score 是无条件 score 加 $g$：

$$\nabla_x \log p(x \mid y) = -\Sigma^{-1}(x - \mu) + g$$

---

**Step 4：新 score 对应的是哪个高斯？（配方求新均值）**

Step 3 得到的新 score 仍然是 $x$ 的线性函数（$\Sigma^{-1}$ 固定），所以它对应的还是一个高斯分布，只是均值发生了偏移。设新均值为 $\mu_\text{updated}$，令：

$$-\Sigma^{-1}(x - \mu_\text{updated}) \stackrel{!}{=} -\Sigma^{-1}(x - \mu) + g$$

展开左边：$-\Sigma^{-1}x + \Sigma^{-1}\mu_\text{updated}$

展开右边：$-\Sigma^{-1}x + \Sigma^{-1}\mu + g$

$x$ 项相消，只剩常数项相等：

$$\Sigma^{-1}\mu_\text{updated} = \Sigma^{-1}\mu + g$$

两边左乘 $\Sigma$：

$$\boxed{\mu_\text{updated} = \mu + \Sigma \cdot g}$$

**直觉：** 原来从 $\mathcal{N}(\mu, \Sigma)$ 采样，加入条件后等价于从 $\mathcal{N}(\mu + \Sigma g,\, \Sigma)$ 采样——均值沿 $g$ 方向移动了 $\Sigma g$，方差不变。

---

**Step 5：CM-Diff 中 $g$ 的来源及符号**

论文（Eq. 9）定义：

$$\log p(w \mid \tilde{x}_0) \propto -\mathcal{L}_\text{cons}$$

因此条件梯度为：

$$g = \nabla_{\tilde{x}_0} \log p(w \mid \tilde{x}_0) = -\nabla_{\tilde{x}_0}\mathcal{L}_\text{cons}$$

代入 $\mu_\text{updated} = \mu + \Sigma g$：

$$\mu_\text{updated} = \mu - \Sigma \cdot \nabla\mathcal{L}_\text{cons}$$

PyTorch 中 `L_cons.backward()` 计算的是 $+\nabla\mathcal{L}_\text{cons}$（损失上升方向），所以代码里应使用 $\mu - \Sigma \cdot \texttt{grad}$（减号）。论文 Algorithm 2 写的 $+\nabla\mathcal{L}_\text{cons}$ 是符号约定不一致，实际效果是将均值推向损失减小、更接近目标域分布的方向。

**为什么对 $\tilde{x}_0$ 而不是 $x_t$ 求梯度：**

$x_t$ 在 $t$ 大时几乎是纯噪声，对其计算统计特性无意义。先用 U-Net 预测干净图像估计 $\tilde{x}_0$，其像素值与真实图像在同一范围内，统计梯度信号更可靠：

$$\tilde{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(x_t - \sqrt{1 - \bar{\alpha}_t}\,\epsilon_\theta(x_t, t)\right)$$

**梯度的实际计算（PyTorch autograd）：**

```python
# 以 VIS→IR 方向为例：
# 输入 9 通道 = [noisy IR (ch1-3)] ⊕ [source VIS (ch4-6)] ⊕ [edge map (ch7-9)]
#   ch1-3: x_t，当前步的含噪 IR（推理目标）
#   ch4-6: x_VIS，固定的源 VIS 图像（条件输入）
#   ch7-9: E_VIS，从源 VIS 提取的边缘图（DexiNed，固定）
# J_IR:   方向嵌入（label embedding，指示"我要生成 IR"）

with torch.no_grad():
    z_t = encoder(x_t, x_VIS, E_VIS)          # CFC：提取跨模态特征
    eps_pred = unet(x_t, x_VIS, E_VIS, J_IR, t)  # 预测噪声，权重冻结

# 从 x_t 和预测噪声还原出预测干净图 x_0_tilde（仍在 IR 域）
x_0_tilde = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)

# 开启梯度追踪：后续对 x_0_tilde 求 L_cons 的偏导
x_0_tilde = x_0_tilde.detach().requires_grad_(True)

# L_scl 和 L_ccl 均以 x_0_tilde 为输入，与 IR 目标域先验比较。
# 梯度方向 = "让预测结果更像真实 IR 域分布"
L_cons = lambda_scl * L_scl(x_0_tilde, prior_stats_IR) \
       + lambda_ccl * L_ccl(x_0_tilde, prior_hist_IR)

L_cons.backward()
grad = x_0_tilde.grad      # shape [B, C, H, W]，与 x_0_tilde 同尺寸

mu_updated = mu + Sigma * grad   # 通过 Classifier Guidance 修正采样均值
```

**后验分布公式：**

$$q(x_{t-1} \mid x_t, \tilde{x}_0) = \mathcal{N}\!\left(x_{t-1};\; \mu_q(x_t, \tilde{x}_0),\; \Sigma_q(t)\right)$$

$$\mu_q(x_t, \tilde{x}_0) = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})\,x_t + \sqrt{\bar{\alpha}_{t-1}}\,\beta_t\,\tilde{x}_0}{1 - \bar{\alpha}_t}$$

$$\Sigma_q(t) = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\,I$$

最终修正后的采样均值：

$$\mu_\text{updated} = \mu_q + \Sigma_q\,\nabla_{\tilde{x}_0}\,\mathcal{L}_\text{cons}$$

### 5.3 推理时约束损失（Inference-time Constraint Losses）

推理损失**不参与模型训练，不更新任何参数**。作用是在每步去噪时对 $\tilde{x}_0$ 计算梯度，将采样均值推向目标域的统计分布，修正 BDT 固定方差带来的颜色偏移。

$$\mathcal{L}_\text{cons} = \lambda_\text{ccl}\,\mathcal{L}_\text{ccl} + \lambda_\text{scl}\,\mathcal{L}_\text{scl} \qquad (\lambda_\text{ccl} = \lambda_\text{scl} = 20.0)$$

---

#### $\mathcal{L}_\text{scl}$：Statistical Constraint Loss（统计约束损失）

**作用：** 对齐生成图和目标域的**一阶（均值）和二阶（标准差）统计量**，校正全局亮度和对比度偏移。

$$\mathcal{L}_\text{scl} = \sum_{\delta \in \{R,G,B\}} \left(\left|\mu^\delta_\text{pred} - \mu^\delta_\text{prior}\right| + \left|\sigma^\delta_\text{pred} - \sigma^\delta_\text{prior}\right|\right)$$

- $\mu^\delta_\text{pred}$，$\sigma^\delta_\text{pred}$：$\tilde{x}_0$ 在通道 $\delta$ 的均值和标准差（每步动态计算）
- $\mu^\delta_\text{prior}$，$\sigma^\delta_\text{prior}$：训练集目标域的统计量（预先计算，推理时固定）

> **注意：** 这里的 $\mu^\delta_\text{pred}$ 是 $\tilde{x}_0$ 的**像素通道均值**（标量），不是 DDPM 后验分布的$x_{t-1}$均值图 $\mu_\theta$（与 $\tilde{x}_0$ 同尺寸的张量）。两者名称相同但含义完全不同。

**PyTorch 计算：**

```python
# x_0_tilde shape: [B, C, H, W]
mean_pred = x_0_tilde.mean(dim=[-2, -1])   # [B, C]，每通道像素均值
std_pred  = x_0_tilde.std(dim=[-2, -1])    # [B, C]，每通道像素标准差

# prior_mean, prior_std 从训练集目标域预先统计，推理时固定
L_scl = F.l1_loss(mean_pred, prior_mean) + F.l1_loss(std_pred, prior_std)
```

计算在 `x_0_tilde`（预测干净图）上进行，而非 $x_{t-1}$（仍含噪声的中间结果），因为含噪图的像素统计量无实际意义。

**可微性：** 均值和标准差都是对像素值的可微操作，autograd 直接处理。

**局限：** 只描述分布的"重心"和"宽度"，无法捕捉分布形状（如双峰、长尾）。

---

#### $\mathcal{L}_\text{ccl}$：Channel Constraint Loss（通道约束损失）

**作用：** 对齐生成图和目标域的**直方图分布形状**，捕捉 $\mathcal{L}_\text{scl}$ 无法覆盖的复杂分布特性，抑制局部颜色伪影。

$$\mathcal{L}_\text{ccl} = \sum_{\delta \in \{R,G,B\}} \sum_{i=1}^{B} \frac{\left(h^\delta_{\text{pred},i} - h^\delta_{\text{prior},i}\right)^2}{h^\delta_{\text{pred},i} + h^\delta_{\text{prior},i} + \varepsilon}$$

- $h^\delta_{\text{pred},i}$：$\tilde{x}_0$ 在通道 $\delta$ 第 $i$ 个 bin 的像素数量（软直方图）
- $h^\delta_{\text{prior},i}$：训练集目标域直方图（预先计算，固定）
- $B$：histogram bin 数量；$\varepsilon = 10^{-6}$：避免除零

**为什么用 $\chi^2$ 距离：**

$\chi^2$ 距离是统计学中比较两个离散分布的标准度量，通用形式为：

$$\chi^2(P, Q) = \sum_i \frac{(P_i - Q_i)^2}{Q_i}$$

CM-Diff 使用对称版本（分母取两个分布的平均），避免除以零且对方向不敏感：

$$\chi^2_\text{sym}(P, Q) = \sum_i \frac{(P_i - Q_i)^2}{P_i + Q_i + \varepsilon}$$

分母 $P_i + Q_i$ 的归一化效果：某个 bin 本来像素就少（稀疏区域），微小差异会被放大，对"不该出现的颜色"惩罚更重；相比 L1/L2 直接比 bin 计数更敏感于分布形状。论文消融实验证明 $\chi^2$ 的 FID 最低。

**可微性（软直方图）：** 硬直方图不可微，改用高斯核软分配：

$$h^\delta_i = \sum_{\text{pixels}} \exp\!\left(-\frac{(x^\delta_\text{pixel} - c_i)^2}{2\sigma^2}\right)$$

其中 $c_i$ 是第 $i$ 个 bin 的中心，梯度可从直方图反传到每个像素值。

---

#### $\mathcal{L}_\text{cons}$：综合约束损失

$$\mathcal{L}_\text{cons} = 20.0 \cdot \mathcal{L}_\text{ccl} + 20.0 \cdot \mathcal{L}_\text{scl}$$

两个损失互补：$\mathcal{L}_\text{scl}$ 校正全局统计，$\mathcal{L}_\text{ccl}$ 校正分布形状。$\lambda = 20$ 是消融实验的最优值：过小约束不足，过大抑制生成多样性。

### 5.4 推理算法（Algorithm 2，VIS → IR 方向）

```
初始化 DexiNed（冻结）
x_IR_T  ~  N(0, I)               # 从纯噪声开始，IR 域
E_VIS  =  DexiNed(x_VIS_0)       # 从源 VIS 提取边缘图
J_IR   =  τ_φ(C_IR)              # VIS→IR 方向嵌入

for t = T, T−1, ..., 1:
    # UNet 输入 9 通道：
    #   ch 1-3：x̂_IR_t      （含噪 IR，推理目标）
    #   ch 4-6：x_VIS_0      （源 VIS，固定条件）
    #   ch 7-9：E_VIS        （VIS 边缘图，固定）
    μ, Σ  =  μ_φ(x̂_IR_t),  Σ_φ(x̂_IR_t)

    # 预测干净图像（IR 域）
    x̃_IR_0  =  (x̂_IR_t − √(1−ᾱ_t) · ε_φ(x̂_IR_t, x_VIS_0, E_VIS, J_IR, t)) / √ᾱ_t

    # 对 x̃_IR_0 计算约束损失，与 IR 目标域先验比较（不更新模型权重）
    L_cons  =  λ_scl · L_scl(x̃_IR_0, prior_IR)  +  λ_ccl · L_ccl(x̃_IR_0, hist_IR)

    # 用梯度修正采样均值，推动生成结果更符合 IR 域统计分布
    x̂_IR_{t−1}  ~  N( μ + ∇_{x̃_IR_0} L_cons ,  Σ )

return x̂_IR_0
```

$\mathcal{L}_\text{cons}$ 的梯度针对 $\tilde{x}_0$ 计算，不针对 $x_t$。类似于 DPS（Chung et al. 2022），但用域统计量替代了学习的先验。

### 5.5 实现说明（HiRISE 适配，`src/inference.py`）

论文针对 RGB（3通道）图像设计，HiRISE 数据集为单通道（IR10 和 RED4 均为灰度图），实现上有以下差异：

**通道数简化：**

论文的损失函数对 R/G/B 三通道分别求和：

$$\mathcal{L}_\text{scl} = \sum_{\delta \in \{R,G,B\}} |\mu^\delta_\text{pred} - \mu^\delta_\text{prior}| + |\sigma^\delta_\text{pred} - \sigma^\delta_\text{prior}|$$

单通道版本退化为一项，在 `sci_l_scl()` 和 `sci_l_ccl()` 中直接对整个张量计算，无需通道循环。

**边缘图：** 论文使用预训练的 DexiNed 深度边缘检测器，本实现使用 Sobel 算子（`diffusion/process.py: sobel_edge()`），输出 1 通道边缘图。完整复现时可替换为 DexiNed。

**后验均值公式（`ddpm_step_sci()` 中实现）：**

$$\mu_q = \frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})\,x_t + \sqrt{\bar{\alpha}_{t-1}}\,\beta_t\,\tilde{x}_0}{1 - \bar{\alpha}_t}$$

$$\Sigma_q = \tilde{\beta}_t = \frac{\beta_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \quad \text{（调度器中预计算为 } \texttt{posterior\_var}\text{）}$$

**软直方图（`soft_histogram()`）：** 使用高斯核软分配，bin 宽度 = $(x_\max - x_\min) / B$：

$$h_i = \frac{\sum_j \exp\!\left(-\frac{(x_j - c_i)^2}{2\,\Delta^2}\right)}{\text{归一化}}, \quad \Delta = \text{bin\_width}$$

梯度通过指数函数对每个像素值可微。

**先验统计的准备（`compute_prior_stats()`）：** 在推理前，需在训练集目标域上预先计算 $\mu_\text{prior}$、$\sigma_\text{prior}$、$h_\text{prior}$ 并保存为 `.pt` 文件：

```python
from inference import compute_prior_stats, save_prior_stats

# 以 RED4（目标域）为例：
tensors = [dataset[i]["red"] for i in range(len(dataset))]   # 所有 RED4 图像
stats   = compute_prior_stats(tensors, bins=256)
save_prior_stats(stats, "checkpoints/prior_stats_red.pt")
```

**推理命令示例：**

```bash
cd src
python inference.py \
    --source    ../data/example_ir.npy \
    --direction 0 \
    --checkpoint ../checkpoints/latest.pt \
    --prior      ../checkpoints/prior_stats_red.pt \
    --output     ../outputs/result.npy
```

---

## 6. U-Net 架构细节

- **主干：** 带跳跃连接的 U-Net
- **基础通道数：** 128
- **编码器通道：** 64 → 128 → 128 → 256 → 256 → 512 → 512（bottleneck）
- **解码器：** 与编码器对称，拼接跳跃连接
- **ResBlocks：** 每个分辨率层级3个残差块
- **Attention：** 多头注意力，仅在 $8{\times}8$、$16{\times}16$、$32{\times}32$ 处，64通道
- **条件注入：** 时间嵌入 + 方向标签嵌入通过 AdaGN 注入每个 ResBlock
- **CFC 交叉注意力：** Attention Block 处，Q 来自主 U-Net，K/V 来自模态编码器
- **输入：** 9通道（3 加噪目标 + 3 源图 + 3 边缘图），分辨率 $256{\times}256$

---

## 7. 实现细节

| 超参数 | 值 |
|---|---|
| 图像分辨率 | $256 \times 256$ |
| 扩散时间步 $T$ | $1000$ |
| Beta 调度 | 线性，$\beta_1=0.0001$，$\beta_T=0.01$ |
| U-Net 基础通道数 | $128$ |
| Batch size | $6$ |
| 优化器 | AdamW |
| 学习率 | $10^{-4}$，每2000步 $\times 0.9$ |
| 总训练步数 | $100{,}000$ |
| $\lambda_{\text{IR}\to\text{VIS}}$，$\lambda_{\text{VIS}\to\text{IR}}$ | $1.0$ |
| $\lambda_\text{ccl}$，$\lambda_\text{scl}$ | $20.0$ |
| Histogram bins $B$ | 未明确指定，推测 ~256 |
| 边缘检测器 | DexiNed（预训练，冻结） |
| 硬件 | 4× NVIDIA V100 |

**噪声调度说明：** 将 $\beta_T$ 从标准 DDPM 的 $0.02$ 降至 $0.01$，使完全加噪图像保留更多结构信息，有助于条件生成。

---

## 8. 关键设计选择

| 设计 | 原因 |
|---|---|
| 单 U-Net 双向 | 消除双网络+循环一致性的需要，减少参数和训练时间 |
| 通道位置隐式编码 | 无需架构改动即可区分目标模态 |
| 方向标签嵌入 | 显式监督信号；消融实验：去掉后 PSNR 下降 7% |
| DexiNed 替代 Sobel/Canny | 学习型边缘在嘈杂 IR 图像中更好地保留语义轮廓 |
| $\chi^2$ 直方图距离 | 对稀疏极端强度差异比 Euclidean/Bhattacharyya 更敏感 |
| SCI 仅在推理时 | 不修改训练好的模型；可针对不同数据集调整统计量 |
| $\lambda=20$ | 最优：过小颜色漂移，过大抑制多样性（消融实验验证） |

---

## 9. 评估指标

- **FID**：$\text{FID}(v', v) = \|\mu_v - \mu_{v'}\|^2 + \text{tr}\!\left(\Sigma_v + \Sigma_{v'} - 2(\Sigma_v \Sigma_{v'})^{1/2}\right)$，越低越好
- **SSIM**：$\dfrac{(2\mu_v\mu_{v'}+c_1)(2\sigma_{v,v'}+c_2)}{(\mu_v^2+\mu_{v'}^2+c_1)(\sigma_v^2+\sigma_{v'}^2+c_2)}$，范围 $[0,1]$，越高越好
- **LPIPS**：$\sum_i \sum_{h_i,w_i} \omega_i \cdot \|f_i(v')_{h_i,w_i} - f_i(v)_{h_i,w_i}\|^2 / (H_i W_i)$，越低越好
- **PSNR**：$10\log_{10}\!\left(\dfrac{MN \cdot I_{\max}^2}{\sum_{i,j}(v_{ij}-v'_{ij})^2}\right)$（dB），越高越好

---

## 10. 数据集

| 数据集 | 总对数 | 训练 | 测试 | 分辨率 | 红外类型 |
|---|---|---|---|---|---|
| AVIID | 3,343 | 2,674 | 669 | 434/512×434/512 | 热红外 |
| VEDAI | 1,209 | 1,089 | 120 | 512/1024 | 近红外（NIR） |
| M3FD | 4,500 | 3,780 | 210 | 1024×768 | 热红外 |

---

## 11. 核心贡献总结

1. **BDT（双向扩散训练）：** 单 DDPM 同时学习 IR↔VIS，用通道位置作为隐式方向信号，用可学习的标签嵌入作为显式方向信号，两个方向损失同时优化。

2. **CFC（跨模态特征控制）：** 两个轻量级模态编码器在多分辨率处通过 Cross-Attention 注入源域特征，保证生成内容与源图语义对齐。

3. **SCI（统计约束推理）：** 推理时梯度引导，用 $\chi^2$ 直方图距离和 L1 统计距离将采样轨迹推向目标域分布，有效消除颜色偏移，无需重新训练。
