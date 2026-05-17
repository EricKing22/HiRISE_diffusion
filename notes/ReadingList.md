# HiRISE CM-Diff 设计决策

## 模型适配（HiRISE vs 原论文）

| 项目 | 原 CM-Diff | HiRISE 版本 |
|---|---|---|
| 模态 | IR (1ch 复制→3ch) ↔ VIS (3ch RGB) | IR10 (1ch) ↔ RED4 (1ch) |
| 输入通道 | 9ch = target(3) + source(3) + edge(3) | 3ch = target(1) + source(1) + edge(1) |
| 输出通道 | 3ch | 1ch |
| Edge map | DexiNed（为 RGB 设计） | Sobel（单通道灰度图） |
| 方向 A (J=0) | IR → VIS | IR10 → RED4 |
| 方向 B (J=1) | VIS → IR | RED4 → IR10 |
| SCI 先验来源 | 训练集目标域全局统计 | RED3 + RED5（同观测局部邻居，更准确） |
| BG12 | 作为 IR 通道之一 | 不使用 |
| x_neigh (RED3, RED5) | 无对应 | 仅用于 SCI 推理时提供局部先验统计 |

## 设计动机

- 双向扩散（BDT）保留：IR10↔RED4 互转，物理上对称，适合 HiRISE 多光谱对齐任务
- 1:1 通道不复制：两个模态均为灰度图，无需 padding 或通道复制，参数更少
- 局部先验优于全局先验：RED3/RED5 是同一地点同次观测的相邻 CCD，统计先验比训练集均值更准确

---

# 背景阅读 TODO

本项目相关研究谱系，按重要性整理。核心谱系（第一部分）展示了从分类器引导到本项目 SGI 的方法演化路径，是理解 SCI/SGI 模块设计动机的必读文献。

---

## 核心谱系：引导扩散 / 引导流场匹配（精读）

本项目 SCI/SGI 模块的理论根基可追溯至以下研究脉络：

```
Classifier Guidance（NeurIPS 2021）——梯度引导扩散的奠基
    ↓ 推广到任意可微约束函数
Universal Guidance（ICLR 2024）——任意函数引导 + x̂₀ 估计中转技巧
    ↓ 形式化为逆问题的贝叶斯后验采样
DPS（ICLR 2023）——每步估计 x̂₀ 施加约束，严格推导
    ↓ 迁移到 Flow Matching 的确定性 ODE 积分
PnP-Flow（ICLR 2025）——FM 框架下的 PnP 引导（实用实现）
On the Guidance of FM（ICML 2025）——FM 引导的完整理论框架（理论基础）
    ↓ 本项目：把统计分布约束作为引导函数，动态邻居先验替代固定先验
SGI（本项目贡献）——Statistical Guided Inference
```

CM-Diff 的 SCI 模块是 DPS 思路在统计约束上的具体实现；本项目 SGI 进一步将引导函数从固定训练集先验改为动态观测时邻居先验，并迁移到 FM 框架。

---

### CM-Diff ⭐⭐⭐（本项目直接基础）
- **标题：** CM-Diff: A Single Generative Network for Bidirectional Cross-Modality Translation Diffusion Model Between Infrared and Visible Images
- **作者：** Bin Hu, Chenqiang Gao*, Shurui Liu, Fang Chen, Junjie Guo, Fangcen Liu, Junwei Han
- **机构：** 重庆邮电大学 + 中山大学 + UC Merced + 西北工业大学
- **发表：** AAAI 2025（arXiv: 2503.05941）
- **核心思路：** 单网络双向跨模态扩散翻译，核心是三个模块：

  **① BDT（Bidirectional Diffusion Training）**
  - 隐式通道位置编码：输入 $Z_t = [q(x_0^{VIS}|t)] \oplus x_0^{IR} \oplus E_{IR}$，noisy 目标占前 3ch，source 占中间 3ch，edge map 占后 3ch——通道位置本身隐式告知方向
  - 显式方向标签嵌入：direction label $J = \tau_\phi(C)$ 经 label embedding 层编码，注入 timestep embedding，无需改变网络结构即可切换方向
  - 两路 MSE 损失同时优化：$L_{joint} = \lambda_1 L_{IR\to VIS} + \lambda_2 L_{VIS\to IR}$，每步 batch 随机抽取两个方向各自的噪声预测误差

  **② CFC（Cross-modality Feature Control）**
  - 两个独立的模态编码器（$Z_E^{IR}$, $Z_E^{VIS}$）从 source 图像提取特征 $F_d$
  - 将 $F_d$ 作为 K, V 注入主 U-Net 每层的 cross-attention，$Q$ 来自主网络 noise feature map $F_g$
  - 公式：$F_{g}^{i+1} = F_g^i + \text{softmax}(\frac{Q^i (K^i)^T}{\sqrt{d}}) V^i$
  - 作用：source 的结构/语义信息通过 attention 直接引导 target 生成，比拼接更精确

  **③ SCI（Statistical Constraint Inference）**
  - **这是与下方 DPS/Universal Guidance 谱系的直接交叉点**
  - 推理时通过统计约束修正采样漂移：先估计中间干净图像 $\tilde{x}_0 = \frac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta)$（即 DPS 的 x̂₀ 估计技巧）
  - $L_{scl}$（统计约束）：对每通道的 mean 和 std 与训练集先验做 L1 对齐
  - $L_{ccl}$（通道直方图约束）：对每通道直方图 bin 计数做加权 L2 匹配（$\chi^2$ 距离变体）
  - 梯度 $\nabla_{\tilde{x}_0}(L_{scl} + L_{ccl})$ 注入反向过程修正均值 $\mu$，不改变模型参数
  - 效果：修正模态分布漂移，减少颜色失真和噪声伪影
  - **局限：** 先验来自训练集全局统计（固定），不能根据具体观测场景自适应调整——这是 SGI 要解决的问题

- **数据集（原论文）：** VEDAI、AVIID、M3FD（航拍 IR-VIS 配对数据集）
- **关键结果：** AVIID IR→VIS：PSNR 23.67 / SSIM 0.835 / FID 22.80；VIS→IR：PSNR 27.96 / SSIM 0.871 / FID 21.33（全面优于 CycleGAN/MUNIT/UNIT/T2V-DDPM 等基线）
- **HiRISE 适配改动：** 见文件顶部设计决策表（通道 9→3、边缘 DexiNed→Sobel、BG12 移除、邻居先验 RED3/RED5 用于 SCI）
- [x] 已读（notes/papers/CM-Diff.pdf）

---

### Classifier Guidance ⭐⭐⭐（引导扩散的奠基方法）
- **标题：** Diffusion Models Beat GANs on Image Synthesis
- **作者：** Prafulla Dhariwal, Alex Nichol（OpenAI）
- **发表：** NeurIPS 2021（arXiv: 2105.05233）
- **核心贡献（本谱系相关部分）：** 该论文的架构改进（多尺度 attention、更多 attention head 等）使扩散模型在 FID 上首次超越 GAN，但对本谱系更重要的是它系统提出了**分类器引导（Classifier Guidance）**这一框架，奠定了"将外部梯度信号注入扩散反向过程"的基本范式。

  **分类器引导的推导**
  - 目标：从条件分布 $p(x|y)$ 采样（其中 $y$ 为类别标签）
  - 贝叶斯分解：$\log p(x_t | y) = \log p(x_t) + \log p(y | x_t) + \text{const}$
  - 对应的条件得分函数（score function）：
    $$\nabla_{x_t} \log p(x_t | y) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{扩散模型得分}} + \underbrace{\nabla_{x_t} \log p(y | x_t)}_{\text{分类器梯度}}$$
  - 实现：在带噪样本 $\{x_t, y\}$ 上训练一个噪声感知分类器 $p_\phi(y|x_t)$，在推理时将其梯度注入反向过程：
    $$\hat\epsilon_\theta(x_t, y) = \epsilon_\theta(x_t) - \sqrt{1-\bar\alpha_t} \cdot \gamma \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$
  - 其中 $\gamma$ 为引导强度超参数（scale）：$\gamma > 1$ 增强条件一致性但降低多样性，$\gamma < 1$ 相反
  - 修改后的均值 $\mu$：在 DDPM 的 $\mu_\theta(x_t)$ 基础上加上 $\Sigma \cdot \gamma \cdot \nabla_{x_t} \log p_\phi(y|x_t)$

  **对后续工作的意义**
  - 证明了：扩散模型的采样可以被"劫持"——在不重新训练生成模型的前提下，通过注入外部梯度信号来引导生成方向
  - 核心约束：**分类器必须在噪声样本上训练**（在干净图像上训练的分类器梯度无效），且引导函数限定为分类概率
  - 同年提出的 Classifier-Free Guidance（Ho & Salimans, 2022）用条件/无条件混合取代外部分类器，规避了训练额外分类器的需要，但失去了使用任意外部函数的灵活性
  - Universal Guidance 的出发点正是：能否把"引导函数"推广到分类器以外的任意可微函数？

- **对 HiRISE 的意义：** SCI 中 $\nabla_{\tilde{x}_0}(L_{scl} + L_{ccl})$ 本质上就是一个引导梯度，$L_{scl} + L_{ccl}$ 类比 $\log p(y|x_t)$——只是 $y$ 不是类别标签，而是统计分布约束。理解 Classifier Guidance 能帮助清晰地写出 SGI 的动机和形式化。
- [ ] 阅读（notes/papers/classifier_guidance.pdf）

---

### Universal Guidance ⭐⭐⭐（推广到任意可微约束）
- **标题：** Universal Guidance for Diffusion Models
- **作者：** Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, Tom Goldstein（马里兰大学）
- **发表：** ICLR 2024（arXiv: 2302.07121）
- **核心问题：** Classifier Guidance 要求将分类器训练在噪声样本 $x_t$ 上，这对于"人脸识别模型"、"目标检测器"、"风格损失"等现成模型不可行。能否在完全**不重新训练任何模型**的前提下，用任意可微函数 $g(x)$ 引导预训练扩散模型？

  **算法核心：x̂₀ 中转技巧**
  - 问题：引导函数 $g$ 在训练时只见过干净图像，直接计算 $\nabla_{x_t} g(x_t)$ 没有意义（$x_t$ 是带噪图像）
  - 解决方案：先将 $x_t$ "去噪"得到干净图像估计 $\hat x_0$，在 $\hat x_0$ 上计算 $g$，再通过链式法则将梯度传回 $x_t$：
    $$\hat x_0(x_t) = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t)}{\sqrt{\bar\alpha_t}} \quad \text{（Tweedie 公式）}$$
    $$\nabla_{x_t} g(\hat x_0) = \frac{\partial g}{\partial \hat x_0} \cdot \frac{\partial \hat x_0}{\partial x_t} = \frac{\partial g}{\partial \hat x_0} \cdot \left(-\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}} \cdot \frac{\partial \epsilon_\theta}{\partial x_t}\right)$$
    简化近似（忽略 $\epsilon_\theta$ 对 $x_t$ 的 Jacobian）：$\nabla_{x_t} g(\hat x_0) \approx \frac{1}{\sqrt{\bar\alpha_t}} \cdot \nabla_{\hat x_0} g(\hat x_0)$
  - 引导步骤（每个扩散时间步 $t$ 执行一次或多次）：
    1. 前向去噪：$\hat x_0 \leftarrow \frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t)}{\sqrt{\bar\alpha_t}}$
    2. 计算引导梯度：$\Delta = \nabla_{x_t} g(\hat x_0)$
    3. 修正：$x_t \leftarrow x_t - \rho_t \cdot \Delta$（梯度下降一步）
    4. 重复步骤 1-3 共 $R$ 次（称为 **recurrence**，增加约束满足程度）
    5. 执行正常的去噪步 $x_{t-1} \sim p_\theta(x_{t-1}|x_t)$

  **Recurrence 技巧**
  - 每个时间步内多次迭代步骤 1-3，使 $x_t$ 充分靠近满足约束的流形，再执行去噪
  - 相当于在每步内做一个局部优化，解决"单步梯度更新量不足"的问题
  - 引导强度 $\rho_t$ 可随 $t$ 衰减（早期大步，后期小步）

  **支持的引导函数类型（论文中展示）**
  - **人脸识别**：$g(x) = 1 - \cos(\text{ArcFace}(x), \text{ArcFace}(x_{ref}))$，将生成的人脸拉向参考身份
  - **CLIP 文本引导**：$g(x) = 1 - \cos(\text{CLIP}_{img}(x), \text{CLIP}_{txt}(\text{prompt}))$
  - **风格迁移**：$g(x) = \|\text{Gram}(x) - \text{Gram}(x_{style})\|_F^2$
  - **目标检测**：$g(x) = -\text{conf}(\text{YOLO}(x), \text{class})$，引导生成指定类别的物体
  - **分割掩码**：$g(x) = \text{CrossEntropy}(\text{SAM}(x), \text{mask}_{target})$
  - **关键洞察**：所有这些函数都只需在干净图像上可微，不需要知道噪声——这一点通过 $\hat x_0$ 中转得到保证

  **与 Classifier Guidance 的关系**
  - Classifier Guidance 是 Universal Guidance 的特例：当 $g(\hat x_0) = -\log p_\phi(y|\hat x_0)$ 且分类器在干净图像上训练时，两者等价
  - Universal Guidance 的 $\hat x_0$ 中转技巧是关键扩展，规避了"噪声感知分类器"的训练要求

  **对 CM-Diff SCI 的直接联系**
  - SCI 中的操作与 Universal Guidance 几乎一一对应：
    - $\tilde x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}$ ↔ $\hat x_0$ 的 Tweedie 估计
    - $L_{scl}(\tilde x_0) + L_{ccl}(\tilde x_0)$ ↔ 引导函数 $g(\hat x_0)$
    - $\nabla_{\tilde x_0}$ 注入修正均值 ↔ 步骤 3 的梯度修正
  - 差异：SCI 没有使用 recurrence（每步只一次），且先验是固定的训练集统计

- **对 HiRISE 的意义：** Universal Guidance 提供了完整的形式化框架，可以直接用于撰写 SGI 的方法描述。SGI 的引导函数 $g(x) = L_{stat}(x;\,\mu_{neigh},\sigma_{neigh},H_{neigh})$ 是一个 Universal Guidance 实例，其中参考统计量来自观测时的邻居波段（RED3, RED5）而非固定先验。
- [ ] 阅读（notes/papers/universal_guidance.pdf）

---

### DPS ⭐⭐⭐（逐步 x̂₀ 估计施加约束，严格贝叶斯推导）
- **标题：** Diffusion Posterior Sampling for General Noisy Inverse Problems
- **作者：** Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, Jong Chul Ye（KAIST + Los Alamos National Laboratory）
- **发表：** ICLR 2023（arXiv: 2209.14687）
- **核心问题：** 给定噪声观测 $y = \mathcal{A}(x_0) + n$（其中 $\mathcal{A}$ 为退化算子、$n$ 为噪声），如何用预训练扩散模型作为隐式先验，从后验分布 $p(x_0|y)$ 中采样？

  **严格贝叶斯推导**
  - 后验得分分解（Bayes 定理 + 得分函数形式）：
    $$\nabla_{x_t} \log p(x_t|y) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{扩散模型先验得分}} + \underbrace{\nabla_{x_t} \log p(y|x_t)}_{\text{似然梯度（不可解析）}}$$
  - 第一项由预训练扩散模型给出：$\nabla_{x_t} \log p(x_t) = -\frac{\epsilon_\theta(x_t)}{\sqrt{1-\bar\alpha_t}}$
  - 第二项 $\nabla_{x_t} \log p(y|x_t)$ 需要对 $x_0$ 积分（边缘化），在一般情况下**不可解析**：
    $$p(y|x_t) = \int p(y|x_0)\,p(x_0|x_t)\,dx_0$$

  **核心近似：p(y|x_t) ≈ p(y|x̂₀)**
  - **关键近似**：用 $x_t$ 的 Tweedie 估计 $\hat x_0(x_t)$ 替代积分：
    $$p(y|x_t) \approx p(y|\hat x_0(x_t)), \quad \hat x_0(x_t) = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t)}{\sqrt{\bar\alpha_t}}$$
  - 对于高斯噪声观测 $y = \mathcal{A}(x_0) + n$（$n \sim \mathcal{N}(0, \sigma_n^2 I)$）：
    $$\nabla_{x_t} \log p(y|\hat x_0) = -\frac{1}{\sigma_n^2} \nabla_{x_t} \|y - \mathcal{A}(\hat x_0)\|_2^2$$
  - 由链式法则，$\nabla_{x_t} \|y - \mathcal{A}(\hat x_0)\|_2^2 = \mathbf{J}_{\hat x_0}^T \nabla_{\hat x_0} \|y - \mathcal{A}(\hat x_0)\|_2^2$，其中 $\mathbf{J}_{\hat x_0} = \partial \hat x_0 / \partial x_t$

  **修正后的反向过程**
  - DDPM 的均值修正：
    $$\tilde\mu_t = \mu_\theta(x_t) - \zeta_t \cdot \nabla_{x_t} \|y - \mathcal{A}(\hat x_0)\|_2^2$$
  - 其中 $\zeta_t$ 为步长调度，控制数据一致性强度
  - 完整算法（伪代码）：
    ```
    for t = T, T-1, ..., 1:
        ε̂ = ε_θ(x_t, t)                       # 模型预测噪声
        x̂₀ = (x_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t    # Tweedie 估计
        grad = ∇_{x_t} ‖y - A(x̂₀)‖²         # 数据一致性梯度
        x_{t-1} = DDPM_step(x_t, ε̂) - ζ_t·grad  # 引导采样
    ```

  **与 Universal Guidance 的关系与区别**
  - **相同**：都在每步估计 $\hat x_0$，都通过 $\hat x_0$ 将引导信号传回 $x_t$
  - **不同**：
    - DPS 是严格的贝叶斯推导（posterior sampling），有理论依据
    - Universal Guidance 是启发式方法，适用范围更广（任意引导函数）
    - DPS 考虑了观测噪声 $\sigma_n^2$，Universal Guidance 没有显式处理噪声模型
    - DPS 明确使用 Jacobian $\mathbf{J}_{\hat x_0}$，Universal Guidance 通常用近似 $\nabla_{x_t} \approx \nabla_{\hat x_0}$
    - DPS 针对线性/非线性算子 $\mathcal{A}$（inpainting、super-resolution、deblur），Universal Guidance 针对更广泛的"风格/语义"约束

  **近似误差分析**
  - 近似 $p(y|x_t) \approx p(y|\hat x_0)$ 在大 $t$（高噪声）时误差大：$x_t$ 太嘈杂，Tweedie 估计 $\hat x_0$ 不准确
  - 在小 $t$（低噪声）时近似良好：$x_t \approx x_0$，$\hat x_0 \approx x_t \approx x_0$
  - 实践中步长 $\zeta_t$ 需要随 $t$ 调整，通常在早期（大 $t$）使用小步长

  **应用场景（论文验证）：** inpainting、super-resolution（4×, 8×）、deblurring（Gaussian/motion blur）、MRI reconstruction（4×, 8×）、compressed sensing（25%, 10%）

  **对 HiRISE/SGI 的直接意义：**
  - CM-Diff 的 SCI 是 DPS 的一个特例：$\mathcal{A}(x_0) = I$（恒等映射，observation = 统计量 $y = [\mu, \sigma, H]$），且 DPS 中的 $\|y - \mathcal{A}(\hat x_0)\|^2$ 对应 SCI 中的 $L_{scl} + L_{ccl}$
  - DPS 的理论框架可以用来为 SGI 提供严格的贝叶斯依据：$y$ 是观测到的邻居波段统计量 $[\mu_{RED3}, \sigma_{RED3}, ..., \mu_{RED5}, \sigma_{RED5}]$，$\mathcal{A}$ 是统计量提取函数
  - PnP-Flow 将这一框架迁移到 FM，理解 DPS 是理解 PnP-Flow 的前提
- [ ] 阅读（notes/papers/DPS.pdf）

---

### PnP-Flow ⭐⭐⭐（DPS 迁移到 Flow Matching 的 ODE 积分）
- **标题：** PnP-Flow: Plug-and-Play Image Restoration with Flow Matching
- **发表：** ICLR 2025
- **核心问题：** DPS 等方法是在 DDPM 的随机微分方程（SDE）框架下开发的——如何将同样的"逐步 $\hat x_0$ 估计 + 数据一致性修正"思路迁移到 Flow Matching（FM）的确定性常微分方程（ODE）框架？

  **Flow Matching 的基本设置**
  - FM 定义了源分布 $p_0$（数据）和目标分布 $p_1$（高斯噪声）之间的线性插值路径：
    $$x_t = (1-t)\,x_0 + t\,x_1, \quad x_0 \sim p_{data},\; x_1 \sim \mathcal{N}(0, I)$$
  - 条件速度场（conditional velocity field）：$v(x_t|x_0,x_1) = x_1 - x_0$（线性路径的导数为常数）
  - 训练目标：$\mathcal{L}_{FM} = E_{t,x_0,x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$
  - 推理：解 ODE $\frac{dx_t}{dt} = v_\theta(x_t, t)$，从 $x_1 \sim \mathcal{N}(0,I)$ 积分到 $x_0$（$t: 1 \to 0$）

  **FM 中的 x̂₀ 估计（"Tweedie 公式"的 FM 类比）**
  - DDPM 中：$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t)}{\sqrt{\bar\alpha_t}}$（根据前向过程方程反解）
  - FM 中（线性插值路径）：$x_t = x_0 + t\,(x_1 - x_0)$，即 $x_t = x_0 + t\cdot v$
  - 因此：
    $$\hat x_0(x_t) = x_t - t\cdot v_\theta(x_t, t)$$
  - 这是 FM 框架下的干净图像估计，是 DPS 近似 $p(y|x_t) \approx p(y|\hat x_0)$ 在 FM 中的直接对应

  **PnP-Flow 的引导 ODE**
  - 目标：解引导 ODE $\frac{dx_t}{dt} = v_\theta(x_t, t) + \lambda_t \cdot g_t(x_t)$，其中 $g_t$ 为数据一致性修正项
  - 每个 ODE 积分步（Euler 离散化，步长 $\Delta t$）：
    1. 估计干净图像：$\hat x_0 \leftarrow x_t - t\cdot v_\theta(x_t, t)$
    2. 计算引导梯度（DPS-style）：$\nabla_t = \nabla_{x_t}\|y - \mathcal{A}(\hat x_0)\|^2$
    3. 修正速度场：$\tilde v_t = v_\theta(x_t, t) - \lambda_t \cdot \nabla_t$
    4. Euler 步：$x_{t-\Delta t} = x_t - \Delta t \cdot \tilde v_t$
  - **Plug-and-Play 部分**：$v_\theta$ 可替换为任意预训练 FM 模型（"plug in"），无需重新训练
  - PnP 先验通过 FM 模型隐式体现，数据一致性通过 $\nabla_t$ 显式注入——两者"解耦"

  **FM 与 DDPM 引导的关键差异**
  | | DDPM/DPS | FM/PnP-Flow |
  |---|---|---|
  | 过程类型 | 随机（SDE）| 确定性（ODE）|
  | $\hat x_0$ 估计 | $\frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}$ | $x_t - t\cdot v_\theta(x_t, t)$ |
  | 噪声调度 | 非线性（$\bar\alpha_t$）| 线性（$t$）|
  | 引导注入点 | 修正 $\mu_\theta$（均值）| 修正 $v_\theta$（速度） |
  | 步数 | 通常 1000 步（可加速至 50–200）| 通常 20–100 步（ODE 更高效）|
  | 确定性 | 每次运行结果不同（SDE）| 相同初始值得到相同结果（ODE）|

  **对 HiRISE/SGI 的直接意义：**
  - 本项目已在 FM 框架上训练了单向 FM 模型（git log 显示 "Add support for baseline uni-directional FM models"）
  - PnP-Flow 的框架直接提供了如何在推理时将统计引导函数 $g(\hat x_0) = L_{stat}(\hat x_0;\,\mu_{neigh},\sigma_{neigh},H_{neigh})$ 注入 FM ODE 的具体算法
  - $\hat x_0 = x_t - t \cdot v_\theta(x_t, t)$ 是实现 SGI 时的核心估计公式，对应 SCI 中的 $\tilde x_0$ 估计
  - FM 的确定性 ODE 有利于引导效果的稳定性（DDPM SDE 的随机性会引入额外方差）
- [ ] 阅读（notes/papers/PnP-Flow.pdf）

---

### On the Guidance of Flow Matching ⭐⭐⭐（FM 引导的理论基础）
- **标题：** On the Guidance of Flow Matching
- **作者：** Ruiqi Feng, Tailin Wu, Chenglei Yu, Wenhao Deng, Peiyan Hu
- **机构：** AI4Science Lab，西湖大学（Westlake University）
- **发表：** ICML 2025（arXiv: 2502.02150）| [代码](https://github.com/AI4Science-WestlakeU/flow_guidance)
- **核心问题：** 扩散模型（DDPM/SDE）的引导方法（DPS、Classifier Guidance 等）不能直接迁移到 Flow Matching（FM/ODE）框架——FM 的引导问题在形式上与扩散模型有本质差异，此前几乎完全未被系统研究。

  **为什么 FM 引导比 DDPM 引导更难？**

  DDPM 的条件得分函数可通过 Bayes 分解简洁写出：
  $$\nabla_{x_t}\log p_t(x_t|y) = \nabla_{x_t}\log p_t(x_t) + \nabla_{x_t}\log p_t(y|x_t)$$
  第一项由扩散模型直接给出（$= -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$），第二项用 $\hat x_0$ 近似（DPS 技巧）。

  FM 的后验速度场（posterior velocity field）的推导则**需要对路径分布做积分**：
  $$v(x_t|y, t) = \mathbb{E}\left[v(x_t|x_0, x_1, t)\;\middle|\;x_t,\,y\right] = \mathbb{E}\left[x_1 - x_0\;\middle|\;x_t,\,y\right]$$
  这里 $(x_0, x_1)$ 是产生 $x_t$ 的路径端点对（在线性插值路径下 $x_t = (1-t)x_0 + tx_1$）。条件化在 $y$ 上破坏了对 $(x_0, x_1)$ 的联合分布，使得期望**无法解析计算**。

  等价形式（对应 Bayes 分解的 FM 版本）：
  $$v(x_t|y, t) = v(x_t, t) + \underbrace{\frac{d}{dt}\log p_t(y|x_t)}_{\text{不仅是空间梯度，还涉及时间导数}}$$
  与 DDPM 相比，多出了沿轨迹方向的**时间导数**项——这正是 FM 引导比 DDPM 难的根本原因。

  **论文提出的四类引导方法**

  | 类别 | 方法 | 特点 |
  |------|------|------|
  | **① 训练无关（Training-free）精确引导** | Monte Carlo 估计后验速度 | 渐近精确，无需额外训练，代价是多次前向采样 |
  | **② 训练型（Training-based）精确引导** | 学习条件速度场修正项 | 额外训练一个轻量网络预测 $v(x_t\|y,t) - v(x_t,t)$，新损失函数覆盖已有方法为特例 |
  | **③ 近似引导（梯度类）** | 类 DPS 的速度场修正 | 覆盖 DPS、ΠGDM、LGD 等为特例；理论分析近似误差来源 |
  | **④ 近似引导（无梯度类）** | 基于重要性加权的重采样 | 不需要引导函数的梯度，适用于不可微约束 |

  **类 DPS 的 FM 近似引导（最实用，与 PnP-Flow 直接相关）**

  FM 版的后验速度近似：用 $\hat x_0 = x_t - t\cdot v_\theta(x_t, t)$ 估计干净图像，再对引导函数求梯度：
  $$\tilde v(x_t, t) = v_\theta(x_t,t) + \lambda_t\,\nabla_{x_t}\log p(y|\hat x_0(x_t))$$
  这与 PnP-Flow 的做法一致，但本文给出了该近似在理论上何时成立（小 $t$ 时近似准确，大 $t$ 时误差大）以及如何选取步长 $\lambda_t$。

  **理论贡献：近似误差分析**
  - 论文严格推导了各类近似引导的误差界，明确指出：
    - 大 $t$（噪声多）时 $\hat x_0$ 估计不准，梯度引导偏差大
    - 小 $t$（噪声少）时近似几乎精确
    - Monte Carlo 训练无关引导误差随样本数增大趋于 0（渐近精确）
  - 给出了在不同场景下选择哪类方法的理论指导

  **验证实验：** 合成数据集（高斯混合、月亮形等）、图像逆问题（inpainting、super-resolution、deblurring）、离线强化学习（offline RL reward guidance）

- **与 PnP-Flow 的关系：**
  - PnP-Flow 是本文 ③ 类近似（梯度类）的特例，本文提供了其理论依据
  - 本文的 ① 类 Monte Carlo 精确引导理论上更准，但计算代价更高
  - 本文统一了 FM 引导的理论谱系，PnP-Flow 是其中一个实用实现点

- **对 HiRISE/SGI 的直接意义：**
  - SGI 将统计约束 $g(\hat x_0) = L_{stat} + L_{hist}$ 作为引导函数——本文明确说明这属于 ③ 类近似引导，理论上在小 $t$（低噪声阶段）最准
  - 论文给出的 $\lambda_t$ 步长衰减策略（大 $t$ 用小步长、小 $t$ 用大步长）可直接用于 SGI 的超参调优
  - 如果 SGI 效果不理想，可按本文尝试 ① 类 Monte Carlo 引导：对每步采样多条路径估计后验，代价换精度
  - 论文的误差分析解释了为什么 SCI（DDPM 版本）和 SGI（FM 版本）都在大 $t$ 时引导不稳定——这是理论上的必然，建议在大 $t$ 时降低引导强度
- [ ] 阅读（notes/papers/FM_guidance.pdf）

---

### SGI（本项目贡献：统计引导推理 + 动态邻居先验）
- **全称：** Statistical Guided Inference（统计引导推理）
- **地位：** 上述谱系在 HiRISE 多光谱预测任务上的具体实例化，是本 IRP 的核心方法贡献

  **方法定位（相对于谱系）**
  - 基础框架：PnP-Flow（FM 的 ODE 引导）
  - 引导函数来源：Universal Guidance（任意可微统计约束）+ DPS（贝叶斯后验解释）
  - 先验改进：CM-Diff SCI 的核心创新——将固定训练集先验替换为动态邻居先验

  **引导函数的形式化**
  - 设在推理时，邻居波段 RED3 和 RED5 可直接观测，提取统计量作为动态先验：
    $$\text{prior} = \{\mu_{k}, \sigma_{k}, H_{k}\}_{k \in \{RED3, RED5\}}$$
    其中 $H_k$ 为归一化直方图（$B$ 个 bin）
  - 目标先验（RED4 的估计先验）由相邻波段插值/均值给出：
    $$\mu^*_{RED4} \approx \frac{1}{2}(\mu_{RED3} + \mu_{RED5}), \quad \sigma^*_{RED4} \approx \frac{1}{2}(\sigma_{RED3} + \sigma_{RED5})$$
  - 引导函数（对应 DPS 中的 $\|y - \mathcal{A}(\hat x_0)\|^2$）：
    $$g(\hat x_0) = \underbrace{\|\mu(\hat x_0) - \mu^*\|_1 + \|\sigma(\hat x_0) - \sigma^*\|_1}_{L_{stat}} + \underbrace{\sum_b w_b\,(\hat H_b(\hat x_0) - H^*_b)^2}_{L_{hist}}$$
  - 在 FM 的每个 ODE 步中：
    $$\hat x_0 = x_t - t\cdot v_\theta(x_t,t), \quad x_{t-\Delta t} = x_t - \Delta t\left[v_\theta(x_t,t) + \lambda_t \nabla_{x_t} g(\hat x_0)\right]$$

  **相对于 CM-Diff SCI 的改进**
  | | CM-Diff SCI | SGI |
  |---|---|---|
  | 先验来源 | 训练集全局统计（固定）| RED3/RED5 邻居（观测时动态）|
  | 扩散框架 | DDPM SDE | FM ODE |
  | $\hat x_0$ 估计 | $\frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}$ | $x_t - t\cdot v_\theta(x_t,t)$ |
  | 理论根基 | 启发式 | 贝叶斯后验（DPS）+ Universal Guidance |
  | 先验自适应性 | 全局（整个训练集均值）| 局部（同观测帧、同地点邻居 CCD）|
  | 推理确定性 | 随机（SDE）| 确定性（ODE，可重现）|

  **为什么动态先验更好（HiRISE 特有的物理理由）**
  - HiRISE 的 CCD 阵列在时序扫描中 RED3/RED4/RED5 是连续读出的相邻列，对应**同一地面区域**
  - 地面目标（矿物、地形）的光谱比值在局部范围内高度稳定，$\mu_{RED4}/\mu_{RED3}$ 变化远小于全球训练集统计方差
  - 固定全局先验（CM-Diff SCI）对高反射率极区地形或低反射率暗物质区域表现差；动态邻居先验自动适应当前场景亮度级别

- **当前状态：** 实验中（基于 FM 模型实现中）

---

## Cycle-consistency 与自适应引导（按匹配度排序）

这组论文用于支撑一个新的 SGI 方向：不再把生成图像拉向固定训练集 global prior，而是利用双向模型构造 source-adaptive cycle loss。对本项目而言，核心形式是

$$
x_{\text{target}} = G_{s\to t}(y_{\text{source}}),\quad
\hat y_{\text{source}} = G_{t\to s}(x_{\text{target}}),\quad
L_{\text{cycle-stat}} = d(\phi(\hat y_{\text{source}}), \phi(y_{\text{source}}))
$$

其中 $\phi$ 可先取 mean/std/low-pass 统计量，而不是 full-pixel reconstruction。这个方向与 DPS/Universal Guidance/PnP-Flow 的关系是：cycle consistency 不作为训练 loss，而作为推理时的可微观测一致性函数。

### CycleNet ⭐⭐⭐（diffusion 中最直接的 cycle-consistency 参考）
- **标题：** CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation
- **作者：** Sihan Xu*, Ziqiao Ma*, Yidong Huang, Honglak Lee, Joyce Chai
- **发表：** NeurIPS 2023（arXiv: 2310.13165）
- **链接：** https://cyclenetweb.github.io/ | https://arxiv.org/abs/2310.13165
- **核心思路：** 将 cycle consistency 引入预训练 diffusion image manipulation：源图经过文本引导编辑后，再通过反向编辑重建源图，用循环一致性正则改善 unpaired I2I 编辑的一致性和质量。
- **与 HiRISE 的相关性：**
  - ✅ 最直接证明“cycle consistency 可以和 diffusion 采样/编辑结合”，不是只属于 GAN。
  - ✅ 目标也是在无显式配对监督或弱监督下维持输入一致性，和 RED→IR→RED 的约束形式接近。
  - ⚠️ 该论文主要用于 text-guided editing，不是物理波段翻译；本项目应借鉴 cycle 约束形式，而不是文本/attention 机制。
- **对本项目新想法的启发：** 可作为“cycle consistency in diffusion generation”的主要引用，支撑把 $\hat y_{\text{source}} \approx y_{\text{source}}$ 作为推理时约束。
- [ ] 阅读

### CycleDiff ⭐⭐⭐（diffusion unpaired I2I 的最新 cycle 方法）
- **标题：** CycleDiff: Cycle Diffusion Models for Unpaired Image-to-image Translation
- **作者：** Shilong Zou, Yuhang Huang, Renjiao Yi, Chenyang Zhu, Kai Xu
- **发表：** IEEE Transactions on Image Processing 2026（DOI: 10.1109/TIP.2026.3657240；arXiv: 2508.06625）
- **链接：** https://arxiv.org/abs/2508.06625 | https://pubmed.ncbi.nlm.nih.gov/41615977/
- **核心思路：** 针对 unpaired I2I，将 translation process 更深地嵌入 diffusion process，并处理“translation 在 clean signal 上、diffusion 在 noisy signal 上”二者不对齐的问题。
- **与 HiRISE 的相关性：**
  - ✅ 任务层面是 cross-domain image translation，比文本编辑更接近 IR10↔RED4。
  - ✅ 明确指出 diffusion 过程与翻译过程的对齐问题，这正对应我们在 FM 中用 $\hat x_0$ 或 clean estimate 做 guidance 的必要性。
  - ⚠️ 重点是训练/模型设计，不是 inference-time gradient guidance。
- **对本项目新想法的启发：** 可用于论证“cycle translation 约束需要作用在 clean estimate 上”，支持在 FM ODE 每步先估计 clean target，再通过反方向模型计算 source reconstruction statistics。
- [ ] 阅读

### Dual Diffusion Implicit Bridges (DDIB) ⭐⭐⭐（ODE 翻译与天然 cycle consistency）
- **标题：** Dual Diffusion Implicit Bridges for Image-to-Image Translation
- **作者：** Xuan Su, Jiaming Song, Chenlin Meng, Stefano Ermon
- **发表：** arXiv 2022（arXiv: 2203.08382）
- **链接：** https://arxiv.org/abs/2203.08382 | https://huggingface.co/papers/2203.08382
- **核心思路：** 使用两个独立训练的 diffusion models，通过 ODE encoder/decoder 在两个域之间翻译；由于两步都是 ODE 映射，理论上 cycle consistent，误差主要来自 ODE solver 离散化。
- **与 HiRISE 的相关性：**
  - ✅ 与本项目 FM ODE 采样形式很接近，尤其是“通过确定性 ODE 翻译并可反向重建”的观点。
  - ✅ 支持把 RED→IR→RED 看成一种桥接/一致性检查，而不是固定全局统计先验。
  - ⚠️ DDIB 使用两个域模型和 latent bridge，不是单个 bidirectional FM U-Net。
- **对本项目新想法的启发：** 可作为 ODE-based cycle consistency 的理论背景；本项目可进一步把 cycle error 变成 sampling-time gradient guidance。
- [ ] 阅读

### CycleDiffusion ⭐⭐（latent cycle + plug-and-play guidance）
- **标题：** Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance
- **作者：** Chen Henry Wu, Fernando De la Torre
- **发表：** arXiv 2022（arXiv: 2210.05559）
- **链接：** https://arxiv.org/abs/2210.05559 | https://github.com/ChenWu98/cycle-diffusion
- **核心思路：** 提出 diffusion latent code 的统一 Gaussian 表示和 DPM-Encoder；在两个相关域的 diffusion models 中观察到 shared latent space，并用 CycleDiffusion 做 unpaired I2I 和 zero-shot editing；同时讨论 latent code 上的 plug-and-play guidance。
- **与 HiRISE 的相关性：**
  - ✅ 同时涉及 cycle translation 与 guidance，是“cycle + guidance”组合的早期参考。
  - ✅ 强调相关域之间可能共享 latent space，适合 IR10/RED4 这种相邻波段。
  - ⚠️ 主要在 latent space 和预训练 diffusion 上工作，不直接处理 FM velocity guidance。
- **对本项目新想法的启发：** 可引用其“相关域共享表示 + cycle translation”的观察，但本项目应强调自己的 guidance 发生在 FM ODE 的 image-space clean estimate 上。
- [ ] 阅读

### CycleFlow ⭐⭐（Flow Matching 中引入 cycle consistency）
- **标题：** CycleFlow: Leveraging Cycle Consistency in Flow Matching for Speaker Style Adaptation
- **作者：** Ziqi Liang, Xulong Zhang, Chang Liu, Xiaoyang Qu, Weifeng Zhao, Jianzong Wang
- **发表：** ICASSP 2025（arXiv: 2501.01861）
- **链接：** https://arxiv.org/abs/2501.01861 | https://largeaudiomodel.com/publication/2025/liang2025cycleflow/
- **核心思路：** 在非平行语音转换中，将 cycle consistency 引入 conditional flow matching，用双 CFM 结构改善 speaker timbre/pitch adaptation，并缓解 train-inference mismatch。
- **与 HiRISE 的相关性：**
  - ✅ 关键词层面最接近：cycle consistency + conditional flow matching + 双向转换。
  - ✅ 可作为“FM 框架中使用 cycle consistency 是合理的”引用。
  - ⚠️ 任务是语音转换，不是图像；主要是训练正则，不是采样时梯度引导。
- **对本项目新想法的启发：** 支撑在 FM 而不只是 DDPM/GAN 中使用 cycle consistency；但本项目贡献应强调 image-domain bidirectional FM 和 inference-time cycle-stat guidance。
- [ ] 阅读

### CycleGAN ⭐⭐（cycle consistency 的经典源头）
- **标题：** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- **作者：** Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
- **发表：** ICCV 2017（arXiv: 1703.10593）
- **链接：** https://arxiv.org/abs/1703.10593 | https://github.com/junyanz/CycleGAN
- **核心思路：** 同时训练 $G:X\to Y$ 和 $F:Y\to X$，用 $F(G(x))\approx x$ 与 $G(F(y))\approx y$ 的 cycle consistency loss 约束 unpaired image translation。
- **与 HiRISE 的相关性：**
  - ✅ 是 cycle consistency 在图像翻译中的标准引用。
  - ✅ 形式上与 RED→IR→RED 完全对应。
  - ⚠️ GAN-based、训练时 loss、双生成器；本项目是 bidirectional FM 单模型 + inference-time guidance。
- **对本项目新想法的启发：** 用作概念源头引用，但不应作为方法主体依据；需要明确本项目不是 GAN 训练，也不依赖 adversarial loss。
- [ ] 阅读

### FlowCycle ⭐（flow-based editing 的弱相关补充）
- **标题：** FlowCycle: Pursuing Cycle-Consistent Flows for Text-based Editing
- **作者：** Yanghao Wang, Zhen Wang, Long Chen
- **发表：** ICLR 2026 withdrawn submission（arXiv: 2510.20212）
- **链接：** https://openreview.net/forum?id=0rHEudxV8K | https://arxiv.org/abs/2510.20212
- **核心思路：** 面向 text-based image editing，用 cycle-consistent process 优化 target-aware intermediate state，使编辑前后保持一致性。
- **与 HiRISE 的相关性：**
  - ✅ 涉及 flow model 与 cycle-consistent optimization，可作为补充背景。
  - ⚠️ withdrawn submission，且任务是文本编辑；引用时需要谨慎，最多作为相关工作线索。
- **对本项目新想法的启发：** 说明 flow-based image editing 也在探索 cycle-consistent 过程，但不宜作为核心论据。
- [ ] 阅读

---

## GAN-based 双向翻译方法（Cycle Consistency）

### CycleGAN
- **标题：** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- **作者：** Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros (2017)
- **链接：** https://arxiv.org/abs/1703.10593 | https://github.com/junyanz/CycleGAN
- **简介：** 同时训练两个生成器（X→Y 和 Y→X）加对抗损失，引入循环一致性损失（翻译后再翻译回来应还原原图）。无需配对数据即可实现双向翻译，但会导致生成结果模糊和细节丢失。CM-Diff 正是为了替代这种双网络+循环一致性的架构。
- [ ] 阅读

### UNIT
- **标题：** Unsupervised Image-to-Image Translation Networks
- **作者：** Ming-Yu Liu, Thomas Breuel, Jan Kautz (2017)
- **链接：** https://arxiv.org/abs/1703.00848 | https://github.com/mingyuliutw/UNIT
- **简介：** 基于共享隐空间假设，两个域的图像可以映射到同一个潜在表示。使用一对 VAE-GAN 模型，强制编码器共享，解码器各自重建，无需配对样本。
- [ ] 阅读

### MUNIT
- **标题：** Multimodal Unsupervised Image-to-Image Translation
- **作者：** Xun Huang, Ming-Yu Liu, Serge Belongie, Jan Kautz (2018)
- **链接：** https://arxiv.org/abs/1804.04732 | https://github.com/NVlabs/MUNIT
- **简介：** 将图像表示分解为域无关的内容码和域特定的风格码。翻译时将源图的内容码与目标域随机采样的风格码重新组合，从而实现多模态（一对多）的翻译输出。
- [ ] 阅读

### DCLGAN
- **标题：** Dual Contrastive Learning for Unsupervised Image-to-Image Translation
- **作者：** Junlin Han, Mehrdad Shoeiby, Lars Petersson, Mohammad Ali Armin (2021)
- **链接：** https://arxiv.org/abs/2104.07689 | https://github.com/JunlinHan/DCLGAN
- **简介：** 在 CUT 的基础上扩展为双向对比学习框架，同时学习 X→Y 和 Y→X 两个映射，使用两个编码器缓解模式崩溃问题。
- [ ] 阅读

---

## GAN-based 单向翻译方法

### CUT
- **标题：** Contrastive Learning for Unpaired Image-to-Image Translation
- **作者：** Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu (2020)
- **链接：** https://arxiv.org/abs/2007.15651 | https://github.com/taesungp/contrastive-unpaired-translation
- **简介：** 用基于 patch 的对比损失（NCE）替代循环一致性损失，最大化输入和输出中对应 patch 之间的互信息。负样本来自同一张图内的其他 patch，训练比 CycleGAN 更快更轻量。
- [ ] 阅读

### I2V-GAN
- **标题：** I2V-GAN: Unpaired Infrared-to-Visible Video Translation
- **作者：** Shuang Li et al. (2021)
- **链接：** https://arxiv.org/abs/2108.00913 | https://github.com/BIT-DA/I2V-GAN
- **简介：** 针对视频级 IR→VIS 翻译，结合对抗、循环感知和跨域/域内相似性约束，在保证空间外观质量的同时维持帧间时序一致性。还发布了 IRVI 基准数据集（24,352 帧配对红外/可见光视频片段）。
- [ ] 阅读

### ROMA
- **标题：** ROMA: Cross-Domain Region Similarity Matching for Unpaired Nighttime Infrared to Daytime Visible Video Translation
- **作者：** Zhenjie Yu et al. (2022)
- **链接：** https://arxiv.org/abs/2204.12367
- **简介：** 针对夜间红外→白天可见光这一极端域差问题，利用红外帧的结构信息和跨域区域相似性匹配强制时空一致性，使用多尺度区域判别器提升细节质量。
- [ ] 阅读

---

## DDPM-based 方法

### T2V-DDPM
- **标题：** T2V-DDPM: Thermal to Visible Face Translation using Denoising Diffusion Probabilistic Models
- **作者：** Nithin Gopalakrishnan Nair, Vishal M. Patel (2023)
- **链接：** https://arxiv.org/abs/2209.08814 | https://github.com/Nithin-GK/T2V-DDPM
- **简介：** 首批将 DDPM 应用于 IR→VIS 人脸翻译的工作之一，专注低光/夜间场景。提出了针对配对翻译设定的加速推理策略，大幅减少采样步数。CM-Diff 的 baseline 对比方法之一。
- [ ] 阅读

### UNIT-DDPM
- **标题：** UNIT-DDPM: UNpaired Image Translation with Denoising Diffusion Probabilistic Models
- **作者：** Hiroshi Sasaki, Chris G. Willcocks, Toby P. Breckon (2021)
- **链接：** https://arxiv.org/abs/2104.05358
- **简介：** 首个将 DDPM 应用于无配对图像翻译的方法，完全用去噪得分匹配目标替代对抗训练。通过 Langevin 动力学的去噪马尔可夫链蒙特卡洛过程以源图为条件进行翻译，训练稳定。
- [ ] 阅读

---

## 遥感扩散模型（Pansharpening 相关）

### PanDiff ⭐⭐（中度相关）
- **标题：** PanDiff: A Novel Pansharpening Method Based on Denoising Diffusion Probabilistic Model
- **作者：** Qingyan Meng, Wenxu Shi, Sijia Li, Linlin Zhang（中科院航天信息研究所）
- **发表：** IEEE TGRS 2023
- **链接：** https://ieeexplore.ieee.org/document/10136205/
- **核心思路：**
  - **差分图学习（Difference Map）：** 不直接生成 HRMS，而是预测 $\Delta MS = HRMS - IMS$（HRMS 与双线性插值放大后的 LRMS 之差）。大幅降低 DDPM 的生成难度——差分图值域集中在 0 附近，比完整图像更容易学习。
  - **MIM（Modal Intercalibration Module）：** 对两路条件输入（PAN 1ch + LRMS 多ch）分别做通道自适应权重校准再拼接注入 U-Net，防止因模态特性差异导致条件信息被"均等对待"。
  - 输入结构：PAN 和 LRMS 作为 U-Net 的条件引导，而非直接预测目标——与 CM-Diff 中 source 拼接进 U-Net 的做法类似。
- **损失：** 标准 DDPM 噪声预测 MSE（ε 预测），在差分图上做扩散。
- **数据集：** GaoFen-2、QuickBird、WorldView-3（遥感全色/多光谱数据集）。
- **与 HiRISE 的相关性：**
  - ✅ **差分图学习直接可借鉴**：IR10 和 RED4 是相邻光谱波段，亮度统计高度接近（均值差约 0.6σ），预测残差 $\Delta = RED4 - IR10$ 比直接预测 RED4 更稳定，可作为 HiRISE 的消融实验选项。
  - ✅ **MIM 的模态校准思路**：HiRISE 中 IR10 与 RED4 虽然都是单通道，但光谱响应不同，在拼接前加入自适应权重校准可能改善特征提取。
  - ⚠️ 任务差异：Pansharpening 是空间分辨率提升（4× 上采样），HiRISE 是同分辨率跨波段预测，空间细节注入机制不完全适用。
- [x] 已读（notes/papers/PanDiff.pdf）

### SSDiff ⭐⭐（中度相关）
- **标题：** SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening
- **作者：** Yu Zhong, Xiao Wu, Liang-Jian Deng, Zihan Cao（电子科技大学）
- **发表：** NeurIPS 2024
- **链接：** https://arxiv.org/abs/2404.11537 | https://github.com/Z-ypnos/SSDiff_main
- **核心思路：**
  - **子空间分解双分支：** 将 pansharpening 分解为空间分支（处理 PAN 高频细节）和光谱分支（处理 LRMS 光谱特征）两个正交子空间，分别由独立的 U-Net 编解码器处理。
  - **APFM（Alternating Projection Fusion Module）：** 用向量投影（矩阵分解）代替 self-attention，将空间分支的特征 $F_{spa}$ 投影进光谱分支的子空间，实现轻量级跨分支融合。相当于把 Q-K-V self-attention 替换为确定性向量投影，参数量更少。
  - **FMIM（Frequency Modulation Inter-branch Module）：** 在两分支交界处做频率调制，平衡低频（光谱分支主导）和高频（空间分支主导）信息的分布，避免频率失衡导致去噪效果变差。
  - **x₀ 预测而非 ε 预测：** $L_{simple} = E[\|x_0 - x_\theta(x_t, c, t)\|_1]$，直接预测干净图像而非噪声，在遥感图像上更稳定（差分图小值域，噪声预测容易过拟合）。
  - **扩散目标：** 与 PanDiff 相同，扩散在 $\Delta = HrMSI - \uparrow LrMSI$ 上进行，而非完整 HrMSI。
  - **L-BAF（LoRA-like branch-wise alternating fine-tuning）：** 150k 步后交替单独微调空间/光谱分支，使各分支学习到更具判别性的专属特征。
- **数据集：** WorldView-3、WorldView-2、GaoFen-2、QuickBird。
- **与 HiRISE 的相关性：**
  - ✅ **x₀ 预测 loss 是最直接可实验的改动**：当前 CM-Diff 使用 ε 预测，切换为 x₀ 预测（L1）对 HiRISE 这种值域集中的遥感任务可能更稳定，消融实验成本低。
  - ✅ **差分图学习**：同 PanDiff，预测 $\Delta = RED4 - IR10$ 而非直接预测 RED4 可能更容易收敛。
  - ⚠️ **双分支设计不适用**：IR10 和 RED4 均为单通道，不存在"空间 vs 光谱"的分离，APFM/FMIM 的设计针对 PAN（高空间）↔LRMS（高光谱）的互补关系，在 HiRISE 中没有对应结构。
  - ⚠️ 同 PanDiff，任务是空间上采样而非等分辨率光谱预测。
- [x] 已读（notes/papers/SSDiff.pdf）

---

## 基础扩散模型论文（必读）

### DDPM
- **标题：** Denoising Diffusion Probabilistic Models
- **作者：** Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
- **链接：** https://arxiv.org/abs/2006.11239
- **简介：** 现代扩散生成模型的奠基论文。建立了扩散概率模型与去噪分数匹配 + Langevin 动力学之间的联系，证明优化重加权变分下界可达高质量图像生成（CIFAR-10 FID 3.17）。引入了 U-Net 预测噪声、正弦时间嵌入、固定 beta 调度等核心设计，CM-Diff 直接基于此框架。
- [ ] 阅读

### Improved DDPM
- **标题：** Improved Denoising Diffusion Probabilistic Models
- **作者：** Alex Nichol, Prafulla Dhariwal (2021)
- **链接：** https://arxiv.org/abs/2102.09672
- **简介：** 对原始 DDPM 的关键改进：引入可学习的逆向方差（第二个输出头预测插值系数 v），使采样步数减少一个数量级仍能保持质量；提出余弦噪声调度。CM-Diff 未采用此改进，固定方差带来的分布漂移问题由 SCI 在推理时补救。
- [ ] 阅读

---

## notes 新增论文（已读）

### BBDM ⭐⭐⭐（高度相关）
- **标题：** BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models
- **作者：** Bo Li, Kaitao Xue, Bin Liu（南昌航空大学）；Yu-Kun Lai（卡迪夫大学）
- **发表：** CVPR 2023
- **核心思路：** 将图像翻译建模为**布朗桥（Brownian Bridge）随机过程**，而非条件生成过程。扩散的起点 $x_T = y$（目标域图像），终点 $x_0 = x$（源域图像），中间过程在两端之间"架桥"，不再需要将条件图像注入 U-Net 每一步——彻底规避了条件扩散的理论保证缺失问题。扩散在 LDM 的潜空间内进行以加速训练和推理。
- **前向过程：** $q_{BB}(x_t | x_0, y) = \mathcal{N}(x_t;\,(1-m_t)x_0 + m_t y,\,\delta_t I)$，其中 $m_t = t/T$，$\delta_t = 2s(m_t - m_t^2)$，中间步方差先升后降，两端方差为 0。
- **反向过程：** 从 $x_T = y$ 出发，逐步去噪回 $x_0$；$y$ 只用于设置反向扩散起点，**不作为每步条件输入**。
- **训练目标：** ELBO 简化为噪声预测 MSE：$E\,[c_{\epsilon t}\,\|\,m_t(y-x_0)+\sqrt{\delta_t}\,\epsilon - \epsilon_\theta(x_t,t)\|^2]$
- **对 HiRISE 的参考价值：**
  - IR10 与 RED4 是高度相关的相邻光谱波段，与 RGB↔边缘这类大域差任务不同，两者统计分布接近——**布朗桥"短路"起点与终点的思路非常契合**。
  - 可作为 CM-Diff 的对比基线：BBDM 无需方向标签，天然支持配对图像翻译，但不支持单网络双向联合训练。
  - 若 CM-Diff 收敛困难，可尝试用 BBDM 作为消融方向（去掉 BDT，保留桥式扩散）。
  - BBDM 的可学习采样多样性参数 $s$ 与 SCI 的精度-多样性权衡类似，可借鉴其调参策略。
- [x] 已读（notes/papers/BBDM.pdf）

---

### DiffV2IR ⭐（低相关，参考价值有限）
- **标题：** DiffV2IR: Visible-to-Infrared Diffusion Model via Vision-Language Understanding
- **作者：** Lingyan Ran, Lidong Wang, Guangcong Wang, Peng Wang, Yanning Zhang（西北工业大学 + 大湾区大学）
- **发表：** CVPR 2025
- **核心思路：** 针对**可见光→热红外（VIS→LWIR/MWIR）翻译**任务，提出两个关键模块：
  - **PLM（Progressive Learning Module）**：三阶段渐进式训练——① LoRA 微调 Stable Diffusion 使其理解红外模态（在 IR-500K 上微调，prompt="an infrared image"）；② paired V2IR 配对监督学习跨模态映射；③ 小数据集风格化细化以支持指定风格的红外输出。
  - **VLUM（Vision-Language Understanding Module）**：使用 BLIP 对可见光图像生成文字描述（场景、光照、阴影等语义），将语言嵌入与可见光图像、分割图三路 classifier-free guidance 联合引导去噪，权重分别为 $s_V, s_S, s_T$。
  - **IR-500K 数据集**：自建，50 万张多波段红外图像，来源覆盖多种场景和传感器。
- **架构：** 基于 Stable Diffusion（LDM），LoRA 适配，U-Net 支持三路条件嵌入。
- **损失：** Latent diffusion 标准 DDPM 噪声预测 + classifier-free guidance。
- **与 HiRISE 的差异（为何相关性低）：**
  - 目标任务是 VIS→热红外（物理机制完全不同），而 HiRISE IR10↔RED4 是同一卫星传感器相邻波段，不涉及热辐射建模。
  - VLUM 的语义理解（物体类别、阴影、光照）在火星地表的矿物/地形纹理翻译中几乎无用——HiRISE 的翻译本质是光谱统计映射，而非语义映射。
  - PLM 的三阶段渐进训练是针对"预训练模型不理解红外"的补丁，HiRISE 从头训练不需要此迁移路径。
- **可借鉴之处（有限）：**
  - PLM 的渐进式课程思路：先在大量数据上学粗糙映射，再在目标分布上细化——若 HiRISE 未来要适配其他火星传感器（如 CTX/CRISM），可参考此渐进迁移范式。
  - IR-500K 数据规模对比说明了预训练数据量的重要性（HiRISE 仅 14287 组，数据瓶颈显著）。
- [x] 已读（notes/papers/DiffV2IR.pdf）
