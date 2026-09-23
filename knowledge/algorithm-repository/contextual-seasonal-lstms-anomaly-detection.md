---
title: CS-LSTMs — Contextual and Seasonal LSTMs for Time Series Anomaly Detection
type:
  - deep-learning
  - sequence-modeling
  - forecasting
  - signal-processing
  - uncertainty-quantification
  - preprocessing
  - time-series
domain:
  - signal-processing
  - generic-ml
source: Zhang, L., Li, Q., Yang, Y., Chen, J., Zeng, R., Lyu, C., & Ji, S. (2026). Contextual and Seasonal LSTMs for Time Series Anomaly Detection. ICLR 2026. arXiv:2602.09690
---
# CS-LSTMs — Contextual and Seasonal LSTMs for Time Series Anomaly Detection

- **来源**: Zhang, L., Li, Q., Yang, Y., Chen, J., Zeng, R., Lyu, C., & Ji, S. (2026). Contextual and Seasonal LSTMs for Time Series Anomaly Detection. *ICLR 2026*.
- **DOI·arXiv**: arXiv:2602.09690 · OpenReview: `2VtveTkmzW`
- **方法类别**: 预测式（prediction-based）单变量时序异常检测 / 频域 + 时域双分支深度学习 + 小波噪声分解
- **代码**: https://github.com/NESA-Lab/Contextual-and-Seasonal-LSTMs-for-TSAD
- **相关文献**: FCVAE（Wang et al., WWW 2024，频率增强条件 VAE，本文主要对比方法）；DeepAR（概率预测 NLL 范式）；Donoho & Johnstone（小波软阈值）；Kim et al.（AAAI 2022，point adjustment 评测批判）

## 问题设定与动机

输入是单变量时间序列（UTS）$x = (x_1, \dots, x_T) \in \mathbb{R}^T$，任务是给每个时刻 $t$ 输出异常判定 $\hat{y}_t \in \{0,1\}$。UTS 在 Web 系统与云服务器监控中是主要信号形态，异常定位直接关系到可靠性管理。

论文指出两类难检异常，二者的共同点是**异常与否取决于局部上下文，而不是数值的绝对大小**：

1. **小点异常（small point anomalies）**：短促的尖峰。在长窗口下它与正常波动的幅度相当，任何基于"窗口重构误差"或"全局幅度阈值"的方法都容易漏检。
2. **缓慢上升异常（slowly rising anomalies）**：片段级异常，值在若干十步内逐渐偏离周期模式。它与真实趋势形态难以区分，且偏离量在段初期极小，检测天然滞后。

已有方法存在两个盲区：

| 盲区 | 表现 | 后果 |
|------|------|------|
| 只看频率分量 | 频域方法（如 FCVAE 的全局/局部频率模块）刻画周期结构，但不建模相邻时刻的局部依赖 | 短促尖峰在频谱上能量低，被当成噪声忽略 |
| 只用时域信息 | 时域重构/预测方法（Anomaly Transformer、Informer、TFAD 等）拟合逐点数值 | 周期本身在演化（非静态）时，模型学到的相位与幅度偏离真实周期 |
| 频域但假设周期静态 | 用固定周期长度做季节分解（STL、Pooling 下采样） | 缓慢上升异常被吸收进"趋势项"，判为正常 |

CS-LSTMs 的应对是三段式流水线：**小波噪声分解 → 频域/时域双分支预测 → 掩码概率损失打分**。核心取舍是：在训练之前先把噪声压掉，避免模型把异常当作"正常模式"学进去；再用两个互补分支分别负责周期演化和局部上下文。

## 数学设定

### 1. 小波噪声分解（去噪 + 掩码）

论文采用加性分解假设

$$
x = \underbrace{(\text{trend} + \text{season})}_{\text{保留}} + \underbrace{\text{noise}}_{\text{滤除}},
$$

只过滤噪声项，不做趋势/季节的显式拆分（这一点把它与 STL 类方法区分开）。一维离散小波变换给出多尺度系数：

$$
\{cA,\; cD^{(L)}, \dots, cD^{(1)}\} = \mathrm{wavedec}(x, \psi, L),
$$

其中 $cA$ 是第 $L$ 层近似系数，$cD^{(i)}$ 是第 $i$ 层细节系数。对正交小波，$\mathrm{waverec}$ 可精确重构。

**噪声尺度（MAD 估计）**。逐层用中位数绝对偏差估计噪声标准差：

$$
\sigma_i = \frac{\mathrm{median}\!\left(|cD^{(i)}|\right)}{\Phi^{-1}(0.75)}, \qquad \Phi^{-1}(0.75) \approx 0.6745,
$$

$\Phi^{-1}$ 是标准正态分位数函数。选 MAD 而非样本标准差，是因为细节系数里混有异常与真实结构，均值和标准差会被它们拉偏，中位数不受影响。

**通用阈值（universal / VisuShrink）**：

$$
\lambda_i = \sigma_i \sqrt{2 \ln n},
$$

$n$ 为信号长度。理论依据是并集界 + 高斯尾界：设 $M_n = \max_j |\epsilon_j|$ 为 $n$ 个 $\mathcal{N}(0,\sigma^2)$ 噪声幅值，则

$$
\Pr\!\left(M_n > \sigma\sqrt{2\ln n}\right) \le \sqrt{\frac{1}{\pi \ln n}} \xrightarrow{n \to \infty} 0 .
$$

从极值理论看 $M_n = \sigma\sqrt{2\ln n} + O\!\left(\frac{\log\log n}{\sqrt{\log n}}\right)$，服从 Gumbel 极限分布。该阈值是渐近最优的保守取法，代价是可能过平滑——论文承认这一点，并给出逐层版本 $\lambda_i = \sigma_i\sqrt{2\ln n}$ 作为替代。

**软阈值**：

$$
\hat{c}D^{(i)} = \mathrm{sign}\!\left(cD^{(i)}\right) \cdot \max\!\left(|cD^{(i)}| - \lambda_i,\; 0\right),
$$

**重构**：

$$
\hat{x} = \mathrm{waverec}\!\left(cA,\; \hat{c}D^{(L)}, \dots, \hat{c}D^{(1)}\right).
$$

软阈值（而非硬阈值）的作用是把模长小于 $\lambda$ 的系数整体压零，同时对大系数做连续收缩，避免硬阈值在系数边界处引入新的跳变（正是小点异常的形态）。去噪序列 $\hat{x}$ 只承担两个角色：**模型输入**（作为时域协变量），以及**疑似异常点的回归目标**（§5 的掩码 $m_t = 0$ 处）。正常点的回归目标仍是原始观测 $x_t$，两者由掩码逐点混合成 $x_t m_t + \hat{x}_t(1-m_t)$，所以模型学到的是正常模式的生成规律，而不是"去噪序列的规律"。

### 2. S-LSTM：季节分支（非重叠窗 + 频域演化）

历史序列 $x_{t-H:t}$ 切成 $m$ 个**等长非重叠**窗，窗长 $w$，$H = m \cdot w$。第 $j$ 个窗 $x_j \in \mathbb{R}^{w}$ 做实数傅里叶变换：

$$
F_j = \mathrm{rFFT}(x_j) \in \mathbb{C}^{w_s}, \qquad w_s = \left\lfloor w/2 \right\rfloor + 1 .
$$

幅度谱与相位谱分别为

$$
A_j = |F_j| \in \mathbb{R}_{\ge 0}^{w_s}, \qquad \varphi_j = \arg F_j \in (-\pi, \pi]^{w_s}.
$$

论文的窗级输入是频域向量（$z_s \in \mathbb{R}^{n \times w_s}$ 记法，$n$ 即窗数）；复现实现里用实部虚部拼接 $z_j^{(s)} = [\operatorname{Re}F_j \,;\, \operatorname{Im}F_j] \in \mathbb{R}^{2w_s}$，与幅度相位表示等价且无需处理相位缠绕。得到序列

$$
z^{(s)} = \left(z_1^{(s)}, \dots, z_m^{(s)}\right) \in \mathbb{R}^{m \times d_s},
$$

用**单层 LSTM** 建模相邻频域向量之间的演化：

$$
\begin{aligned}
f_t &= \sigma\!\left(W_f [h_{t-1}; z_t] + b_f\right), &
i_t &= \sigma\!\left(W_i [h_{t-1}; z_t] + b_i\right), \\
o_t &= \sigma\!\left(W_o [h_{t-1}; z_t] + b_o\right), &
\tilde{C}_t &= \tanh\!\left(W_c [h_{t-1}; z_t] + b_c\right), \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t, &
h_t &= o_t \odot \tanh(C_t).
\end{aligned}
$$

预测头输出**下一个窗的频谱**，再逆变换回时域：

$$
\hat{F}_{m+1} = \mathrm{head}_\theta(h_m), \qquad \mu_s = \mathrm{irFFT}\!\left(\hat{F}_{m+1}\right) \in \mathbb{R}^{w}.
$$

这里的设计要点：不假设周期静态。把"周期模式如何随时间演化"交给频域上的序列建模，等价于允许每个频率分量的幅度与相位缓慢漂移——缓慢上升异常正是破坏了这种漂移的连续性。直接在时域预测下一个窗的 $w$ 个点要求模型精确外推相位，学习难度更高；在频域预测再逆变换是可学习性更好的等价参数化。

### 3. C-LSTM：上下文分支（重叠窗 + 段级建模）

同一段历史切成**重叠**窗：窗长 $w_c$，步长 $s_c < w_c$，窗数

$$
n_c = \left\lfloor \frac{H - w_c}{s_c} \right\rfloor + 1 .
$$

重叠带来的效果是把"点级预测"变成"段级预测"：相邻窗共享大部分样本，模型比较的是**段与段**的变化，而不是点与点的跳变。每个窗同样做频域特征化（论文两个分支都过 FFT，C 分支窗更短、更细），得到

$$
z^{(c)} = \left(z_1^{(c)}, \dots, z_{n_c}^{(c)}\right), \qquad \mu_c \in \mathbb{R}^{w},
$$

LSTM 结构与 S 分支同构，输出 $\mu_c, \sigma_c$。C 分支的敏感尺度是短期突变的尺度——小点异常的持续时间恰好落在这个尺度上，消融实验（$w/o$ C-LSTM）在 Yahoo 上 Best-F1 从 0.885 降到 0.864、WSD 上从 0.910 降到 0.856，说明该分支贡献集中在短时异常。

### 4. 协变量注入

时域信息以协变量形式并入两个分支的输入：把窗内原始值直接拼到频域特征上，

$$
z_j = \big[\underbrace{\operatorname{Re}F_j,\ \operatorname{Im}F_j}_{\text{频域周期信息}},\ \underbrace{x_j}_{\text{时域协变量}}\big] \in \mathbb{R}^{2w_s + w}.
$$

频域分量编码周期性，时域分量保留值域与局部变化幅度（频域表示丢掉的信息）。消融（$w/o$ Covariate）在 Yahoo 上 Best-F1 从 0.885 降到 0.826，WSD 从 0.910 降到 0.840——两个数据集上的跌幅都明显大于去掉单个分支，说明数值层面的信息不能只靠频谱复原。

### 5. 带异常掩码的负对数似然

模型同时回归**均值 $\mu_t$ 与方差 $\sigma_t^2$**，损失是高斯负对数似然（NLL）。标准形式为

$$
\mathrm{NLL} = \sum_t \left[\frac{1}{2}\log \sigma_t^2 + \frac{(y_t - \mu_t)^2}{2\sigma_t^2}\right] + C .
$$

论文 Eq. (6) 去掉常数项与因子 $\tfrac12$，并引入异常掩码：

$$
\boxed{\ \mathcal{D}(\mu, \sigma, x, \hat{x}) = \sum_t \left[ \log \sigma_t^2 + \frac{\left(x_t m_t + \hat{x}_t (1 - m_t) - \mu_t\right)^2}{\sigma_t^2} \right], \qquad \sigma_t > 0\ }
$$

其中 $m_t \in \{0,1\}$ 指示**正常点**（$m_t = 1$ 为正常，论文 Appendix F），$\hat{x}_t$ 是小波去噪值。掩码的作用：正常点用真实观测 $x_t$ 回归，疑似异常点用去噪估计 $\hat{x}_t$ 回归，模型不会去拟合异常形态。记残差 $r_t = x_t m_t + \hat{x}_t(1-m_t) - \mu_t$，论文给出两项的最优点可分离：

$$
\mu_t^\star = x_t m_t + \hat{x}_t (1 - m_t), \qquad \sigma_t^{\star} = |r_t| \ \ \text{（并设下界 } \sigma_{\min} > 0 \text{）}.
$$

性质：$\sigma_t \ge \sigma_{\min}$ 时损失光滑，对 $\mu$ 严格凸、对 $\sigma$ 强制（coercive），两块分别优化即可收敛。总损失是两分支之和

$$
\mathcal{L} = \mathcal{L}_s + \mathcal{L}_c, \qquad
\mathcal{L}_s = \mathcal{D}(\mu_s, \sigma_s, x, \hat{x}), \quad
\mathcal{L}_c = \mathcal{D}(\mu_c, \sigma_c, x, \hat{x}),
$$

两支的最优解独立，不共享参数。

**方差头的热启动**。NLL 里 $\log\sigma^2$ 要跨若干数量级收敛（噪声尺度 $\sigma \sim 0.1$ 时 $\log\sigma^2 \approx -4.6$），短训练预算下方差头来不及走到位，会产生"分数全部虚高"的假象。用小波细节系数的 MAD 估计做常数初始化

$$
\mathrm{bias}(\text{head}_{\log\sigma^2}) = \log\!\left(\hat{\sigma}_{\text{noise}}^2\right), \qquad \hat{\sigma}_{\text{noise}} = \frac{\mathrm{median}(|cD^{(1)}|)}{0.6745},
$$

即让初始预测方差等于噪声分解给出的噪声功率。这是工程技巧，不在论文正文中，但与本方法的噪声分解步骤同源。

### 6. 异常分数与判定阈值

打分是**预测值与真实观测的偏差**。论文正文只给出文字判据（"预测与观测的偏差超出正常范围时判定为异常"），结论中提到的 "normal distance" 严重度指标**没有给出显式公式**。文献中与之等价、且被本文实现沿用的形式是标准化偏差：

$$
z_t = \frac{\left|x_t - \mu_t\right|}{\sigma_t}, \qquad \hat{y}_t = \mathbb{1}\!\left[z_t > \tau\right],
$$

两个分支可先融合再打分：$\mu_t = \tfrac12(\mu_{s,t} + \mu_{c,t})$，$\sigma_t^2 = \tfrac12(\sigma_{s,t}^2 + \sigma_{c,t}^2)$。阈值 $\tau$ 有三种取法：

| 取法 | 公式 | 适用 |
|------|------|------|
| 固定倍数 | $\tau = 3$ | 分数已由 NLL 校准（$\sigma_t$ 即预测标准差），无需标签 |
| Best-F1 | $\tau^\star = \arg\max_\tau F_1(\tau)$，在分位数网格上遍历 | 离线评测（论文报告口径） |
| 验证集分位数 | $\tau = Q_{1-\alpha}(z_{\text{val, 正常段}})$ | 需要干净的验证段，控制误报率 |

**评测口径的坑**：论文的 Best-F1 与 Delay-F1 都采用 **point adjustment**（段内任一点命中则整段判为命中），Delay-F1 额外要求命中发生在等待时间 $k$ 内（论文按数据集取 Yahoo 3、KPI 7、WSD 40、NAB 150）。point adjustment 会显著虚高 F1——本条目附带的合成数据实验里，同一模型无 PA 的 Best-F1 为 0.472、加 PA 后跳到 0.946。报告结果时应同时给出两种口径（批判见 Kim et al., AAAI 2022）。

## 关键假设

| # | 假设 | 违反后果 |
|---|------|---------|
| 1 | 异常性是**局部上下文**属性，不是绝对幅度属性 | 全局阈值类判据失效；必须依赖预测残差 |
| 2 | 噪声近似加性且 i.i.d. 高斯，细节系数 MAD 能稳健估计其尺度 | VisuShrink 阈值不再最优，过平滑或去噪不足 |
| 3 | 周期模式的演化在频域上**连续可推演**（周期允许漂移，但不允许跳变） | 周期突变时 S 分支外推失效，误报集中在突变点 |
| 4 | 训练段内异常占比足够低，去噪 + 掩码能阻止异常被学成正常 | 模型把异常形态内化，异常分数趋零，漏检 |
| 5 | 两分支互补：S 管频域周期演化、C 管段级局部上下文 | 融合退化为单分支；消融显示 WSD 上 w/o S-LSTM 跌幅最大（0.910 → 0.762） |
| 6 | 观测噪声近似高斯，NLL 形式正确 | 重尾噪声下方差估计不稳，分数分布尾部虚高 |
| 7 | 窗长 $w$ 与主周期匹配：太短则频谱分辨率不足，太长则把点异常均值化掉 | 论文的窗长研究（Fig. 7）显示合理区间内波动约 2%，越界后明显退化 |
| 8 | 训练/测试同分布 | 跨域迁移实验（Table 3）显示可迁移但会掉点：Yahoo 上训练的模型直接测 KPI 得 F1 0.929，反向 Yahoo 只得 0.670 |

## 适用场景

- **云监控 / 运维 KPI**：单变量指标（CPU、QPS、延迟、错误率）的在线异常定位，也是论文的主战场（KPI、WSD 数据集）
- **长周期 + 局部突变混合的传感器信号**：周期本身在漂移（季节性设备的负载周期），同时存在短促脉冲故障
- **缓慢退化类故障**：设备效率、电池容量、摩擦系数等随时间的渐进偏离，对应论文的"缓慢上升异常"
- **标注稀缺场景**：预测式方法只需正常段训练，评测才需要标签；NLL 输出的 $\sigma_t$ 直接给出置信区间
- **需要跨域快速部署**：论文的迁移实验显示在 Yahoo 上训练后直接测 WSD 得 F1 0.883、测 NAB 得 0.986
- **算力受限的边缘部署**：参数量约 600K（对比方法约 1.4M），推理 512 次耗时 4.62 ms（GPU）/ 3.82 ms（CPU，3090 24G），论文报告推理时间较当时 SOTA 降低约 40%

### 不适用

- **多变量 / 多通道耦合异常**：论文只处理 UTS；通道间相关异常需换成 multivariate 框架（如 FCVAE 的多变量变体、Graph-based 方法）
- **无周期性且无趋势的平稳噪声**：S 分支的频域演化建模没有可学的结构，退化为纯噪声拟合
- **异常即"新周期"的场景**：若异常本身表现为周期长度切换，假设 3 被违反，S 分支会持续误报直到重新收敛
- **训练数据本身含大量未标注异常**：掩码靠去噪残差近似，而残差阈值只认幅度——本条目合成实验里训练区间注入 58 个异常（占该区间 3.72%），κ=3 的残差掩码只标出其中 2 个（召回 0.034；测试段 5/113，召回 0.044），漏掉的异常以观测值身份进入回归目标被模型学成正常。异常率越高这个漏检越致命；真实基准的异常率在 0.68%（Yahoo）到 9.89%（NAB）之间，且多为幅度更小的缓慢上升段，掩码的可达召回只会更低
- **需要毫秒级在线推理的极高频流**：滑窗 + 双分支 + 小波分解的常数开销高于单分支阈值法
- **需要异常类型解释**：方法只输出分数与偏差方向，不区分异常类别
- **强非高斯、重尾噪声**（如带突发丢包的网络计数）：高斯 NLL 不匹配，建议换 Student-t 或分位数损失

## 实现要点

### 关键超参数（论文 Table 8）

| 参数 | 取值 | 说明 |
|------|------|------|
| `batch_size` | 512 | 训练批大小 |
| `max_epochs` | 30 | 训练轮数 |
| `seasonal_window_size` ($w$) | 48 | S 分支窗长，非重叠 |
| `total_window_size` ($H$) | 240 | 历史长度 |
| `context_window_size` | 4 | C 分支窗/步长相关参数（论文未明确区分窗长与步长） |
| `d_model` | 256 | 隐层维度 |
| `num_layers` | 1 | LSTM 层数（单层） |

论文强调只有两个量需要调，且都写成比例形式：

$$
\text{total\_window\_size} = m \times \text{seasonal\_window\_size}, \quad m \in (5, 7),
$$
$$
\text{contextual\_window\_size} = \text{seasonal\_window\_size} \parallel n, \quad n \in (5, 7).
$$

即历史长度取 5~7 个季节窗，上下文窗取季节窗的 $1/7 \sim 1/5$ 的粒度（论文原文写作整除形式，未给出逐项换算的完整说明）。

### 调优经验

1. **先定 $w$，再定 $H$**：$w$ 要能覆盖一个完整主周期（论文窗长研究显示，窗覆盖整个周期时 S-LSTM 最优；继续加长会让窗口混入更多噪声而退化）。$H = m w$ 取 5~7 个窗，$m$ 小则频域序列太短、LSTM 学不到演化。
2. **两个分支的窗长必须不同**：两支用同样的窗就失去了互补性。C 分支要短、要重叠，它的价值在于段级比较。
3. **方差头做常数初始化**：用 $\log\hat{\sigma}_{\text{noise}}^2$ 初始化，否则短训练预算下 $\log\sigma^2$ 走不到位，所有分数一起虚高，阈值完全失效。
4. **噪声分解替代 STL / Pooling**：论文对比三种去噪策略（Table 6），噪声分解在 Yahoo/WSD 上 Best-F1 最高（0.885 / 0.910），且训练开销最低（Table 7 每 epoch 52.4 s，Pooling 59.3 s，STL 458.4 s）。STL 慢一个数量级且效果更差，原因是它把缓慢上升异常吸收进趋势项。
5. **损失函数必须用 NLL**：换成 MSE 或 MAE 全面崩（Table 9，Yahoo Best-F1 掉到 0.722 / 0.712，KPI Delay-F1 从 0.879 掉到 0.495 / 0.757）。方差项既是损失权重（自动降权高噪声点），也是打分时的归一化尺度，一举两得。
6. **掩码的构造依赖去噪残差**：论文没有给出判定正常/异常的规则，本条目复现采用 $|x_t - \hat{x}_t| \le \kappa \cdot \mathrm{MAD}(\text{residual})$（$\kappa = 3$）。这是复现近似，不是论文原文，且实测偏保守：在合成数据上对注入异常的召回仅 0.034（训练区间）/ 0.044（测试区间）。残差阈值只认幅度，缓慢上升段在早期几乎不产生残差，是漏检的主要来源。
7. **数据预处理**：论文只说明做归一化与缺失值插补，未给公式。复现建议对**训练段**统计量做 z-score（本条目代码即如此），若对全序列标准化会引入测试信息。归一化对频域特征尤其关键：未标准化时 rFFT 的直流分量比其余分量大两个数量级，优化条件数极差。
8. **阈值先看校准再谈检测**：先在训练段正常点上验证平均分数 ≈ 1（本条目合成实验为 0.528，同段注入点 1.013，测试段正常点 1.082）。若正常点分数远大于 1，说明方差头未收敛，此时调阈值无意义。注意校准统计量要只取正常点——训练段现在含未标注异常，混进去会把均值抬高。
9. **滑窗打分要平均而不是取最后一个窗**：相邻窗对同一点的预测取均值能平滑掉单窗的相位误差。打分步长可以大于 1（本条目取 2），代价是定位精度。
10. **报告口径**：同时给出无 PA 的 F1 与 Best-F1（PA）。论文报的是后者，但两者差距可能极大（本条目 0.472 vs 0.946），只报 PA 会误导。

## 代码

下面是最小可运行实现（单文件，CPU，约 4 秒跑完）。流程完整覆盖：小波去噪 → 非重叠/重叠窗构造 → 双分支前向 → 频域预测 + 逆 FFT → 掩码 NLL 反向（参考值按 Eq.(6) 逐点混合 $x_t m_t + \hat{x}_t(1-m_t)$，`masked_nll` 内直接消费掩码）→ 滑窗异常打分 → Best-F1 / point-adjustment 评测。合成数据含周期 + 趋势 + 噪声，并在**训练区间与测试区间都**注入小点异常与缓慢上升异常（训练区间要含未标注异常，Eq.(6) 的掩码才有对象可掩）。代码内附两段掩码证据：**损失层面的接线自检**（同一批输入、同一组权重，只改掩码，打印三个互不相同的损失值与混合参考值的差异范数），以及**训练轨迹层面的 A/B**（同一数据、同一初始化，真实掩码 vs 掩码强制全 1，对照两次训练的末损失与 Best-F1）。已实测输出见代码块后的运行结果。

```python
"""
CS-LSTMs - Contextual and Seasonal LSTMs for Time Series Anomaly Detection
Minimal runnable reproduction (smoke scale).

Pipeline: wavelet noise decomposition
          -> S-LSTM (non-overlapping windows, frequency domain)
          -> C-LSTM (overlapping windows, frequency domain)
          -> masked negative log-likelihood (mean + variance); the mask mixes the
             regression reference r = x*m + x_hat*(1-m) (paper Eq. 6): normal points
             regress on the observation, suspected anomalies on the denoised value
          -> anomaly score |x - mu| / sigma

Reference: Zhang, L., Li, Q., Yang, Y., Chen, J., Zeng, R., Lyu, C., & Ji, S. (2026).
Contextual and Seasonal LSTMs for Time Series Anomaly Detection. ICLR 2026. arXiv:2602.09690

Run: python cs_lstms_smoke.py     (CPU only, < 90 s)

The synthetic series injects anomalies in the training region too (realistic 1%-5%
rate), so the Eq. (6) mask is consumed while training. A control run with the mask
forced to all-1 isolates the mask's effect end to end. The mask-construction rule
(kappa=3 * MAD of the denoising residual) is a reproduction approximation: the
paper does not state how the mask is built.
"""

import time

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')  # 无显示环境：禁止 plt.show() 阻塞
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

try:
    import pywt
    HAS_PYWT = True
except Exception:  # 未安装 PyWavelets 时退回手写 Haar
    HAS_PYWT = False


# ---------------------------------------------------------------------------
# 1. 小波噪声分解（wavedec -> MAD 噪声估计 -> 通用阈值 -> 软阈值 -> waverec）
# ---------------------------------------------------------------------------

def _haar_dwt(x, level):
    """手写多级 Haar 分解，返回 (cA, [cD_L, ..., cD_1])（pywt 同序）。"""
    a = np.asarray(x, dtype=np.float64)
    pad = (-a.size) % (2 ** level)          # 补齐到 2^level 的整数倍
    if pad:
        a = np.concatenate([a, np.repeat(a[-1:], pad)])
    details = []
    for _ in range(level):
        even, odd = a[0::2], a[1::2]
        details.append((odd - even) / np.sqrt(2.0))
        a = (even + odd) / np.sqrt(2.0)
    return a, details[::-1]


def _haar_idwt(cA, details, out_len):
    """手写 Haar 重构，details 顺序为 [cD_L, ..., cD_1]。"""
    a = np.asarray(cA, dtype=np.float64)
    for d in details:                        # 由最粗层逐级重建
        even = (a - d) / np.sqrt(2.0)
        odd = (a + d) / np.sqrt(2.0)
        a = np.empty(even.size + odd.size)
        a[0::2], a[1::2] = even, odd
    return a[:out_len]


def wavelet_denoise(x, wavelet='db4', level=3):
    """软阈值小波去噪：sigma_i = median(|cD_i|)/0.6745, lambda_i = sigma_i*sqrt(2*log n)。"""
    x = np.asarray(x, dtype=np.float64)
    if HAS_PYWT:
        coeffs = pywt.wavedec(x, wavelet, level=level)   # [cA, cD_L, ..., cD_1]
        cA, details = coeffs[0], list(coeffs[1:])
    else:
        cA, details = _haar_dwt(x, level)
    n = max(x.size, 2)
    for i, d in enumerate(details):
        sigma = np.median(np.abs(d)) / 0.6745          # MAD 稳健尺度估计
        lam = sigma * np.sqrt(2.0 * np.log(n))         # 通用阈值 (VisuShrink)
        details[i] = np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)  # 软阈值
    if HAS_PYWT:
        return pywt.waverec([cA] + details, wavelet)[:x.size]
    return _haar_idwt(cA, details, x.size)


# ---------------------------------------------------------------------------
# 2. 合成数据：周期 + 趋势 + 噪声 + 小点异常 + 缓慢上升异常
# ---------------------------------------------------------------------------

def make_series(n=3000, period=24, train_lo=40, train_hi=1600, anomaly_from=1800,
                n_point=8, n_ramp=3, n_point_train=10, n_ramp_train=1, seed=0):
    """周期 + 趋势 + 噪声，并在训练区间与测试区间**都**注入两类异常。

    训练区间注入不可省：掩码 NLL（Eq. 6）的存在理由就是训练数据含未标注异常；
    训练区间若完全干净，该机制在训练中退化，复现只剩名义。训练区间异常率按真实
    TSAD 基准量级取 1%~5%（见 main 里的实测打印）。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = (10.0 + 0.0008 * t
            + 3.0 * np.sin(2 * np.pi * t / period)
            + 1.2 * np.sin(2 * np.pi * t / (period * 7)))
    x = base + 0.25 * rng.normal(size=n)
    label = np.zeros(n, dtype=np.int64)

    # (a) 小点异常：长窗口下看似正常的短尖峰
    for lo, hi, k in ((train_lo, train_hi, n_point_train), (anomaly_from, n, n_point)):
        idx = rng.choice(np.arange(lo, hi), size=k, replace=False)
        x[idx] += rng.choice([-1.0, 1.0], size=idx.size) * rng.uniform(4.0, 6.0, size=idx.size)
        label[idx] = 1
    # (b) 缓慢上升异常：段内逐渐偏离周期模式
    for lo, hi, k in ((train_lo, train_hi - 80, n_ramp_train), (anomaly_from, n - 80, n_ramp)):
        for s in rng.choice(np.arange(lo, hi), size=k, replace=False):
            seg = int(rng.integers(30, 60))
            x[s:s + seg] += np.linspace(0.0, 3.0, seg)
            label[s:s + seg] = 1
    return x, label


def noise_scale(x, wavelet='db4', level=3):
    """小波细节系数 MAD 噪声尺度估计（最细层 cD_1），即论文的 sigma_i。"""
    x = np.asarray(x, dtype=np.float64)
    if HAS_PYWT:
        d1 = pywt.wavedec(x, wavelet, level=level)[-1]
    else:
        d1 = _haar_dwt(x, level)[1][-1]
    return float(np.median(np.abs(d1)) / 0.6745)


def anomaly_mask(x, x_hat, kappa=3.0):
    """掩码 m_t=1 表示正常点。论文未给出掩码构造规则，此处为复现近似：
    残差 |x - x_hat| 超过 kappa 倍 MAD 尺度则视为疑似异常（m_t=0）。"""
    res = np.abs(x - x_hat)
    scale = np.median(res) / 0.6745 + 1e-8
    return (res <= kappa * scale).astype(np.float64)


# ---------------------------------------------------------------------------
# 3. 窗口与频域特征
# ---------------------------------------------------------------------------

def freq_feat(wins, use_covariate=True):
    """窗 -> [Re(FFT), Im(FFT), 时域原值(协变量)]。"""
    f = np.fft.rfft(wins, axis=-1)
    parts = [f.real, f.imag]
    if use_covariate:
        parts.append(wins)          # 时域信息作为协变量并入两个分支
    return np.concatenate(parts, axis=-1).astype(np.float32)


def make_sample(x_in, x_ref, x_den, msk, t, H, w_s, w_c, stride_c, use_covariate=True):
    """窗级样本：(z_s, z_c, 观测窗, 去噪窗, 掩码窗)。

    参考值不在此处混合——由 masked_nll 按论文 Eq.(6) 用掩码现算
    r_t = x_t*m_t + x_hat_t*(1-m_t)，保证 mask 真正进入损失而非旁路元数据。
    """
    hist = x_in[t - H:t]
    z_s = freq_feat(hist.reshape(H // w_s, w_s), use_covariate)                    # 非重叠窗
    segs = np.stack([hist[i:i + w_c] for i in range(0, H - w_c + 1, stride_c)])
    z_c = freq_feat(segs, use_covariate)                                          # 重叠窗
    return z_s, z_c, x_ref[t:t + w_s], x_den[t:t + w_s], msk[t:t + w_s]


# ---------------------------------------------------------------------------
# 4. 双分支模型：S-LSTM / C-LSTM
# ---------------------------------------------------------------------------

class Branch(nn.Module):
    """S/C 分支：LSTM 在频域推演下一个窗的频谱，逆 FFT 回时域得 mu，并预测对数方差。"""

    def __init__(self, feat_dim, hidden, out_len, var_init=1.0):
        super().__init__()
        self.out_len = out_len
        self.nmode = out_len // 2 + 1
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=1, batch_first=True)
        self.head_spec = nn.Linear(hidden, 2 * self.nmode)   # 下一个窗的 Re/Im
        self.head_lv = nn.Linear(hidden, out_len)
        nn.init.zeros_(self.head_lv.weight)
        nn.init.constant_(self.head_lv.bias, float(np.log(var_init)))  # 方差头热启动

    def forward(self, z):
        out, _ = self.lstm(z)
        h = out[:, -1, :]
        spec = self.head_spec(h)
        c = torch.complex(spec[:, :self.nmode], spec[:, self.nmode:])
        mu = torch.fft.irfft(c, n=self.out_len, dim=-1)   # 频谱 -> 时域预测窗
        return mu, self.head_lv(h).clamp(-12.0, 4.0)


class CSLSTMs(nn.Module):
    def __init__(self, feat_s, feat_c, hidden=16, out_len=8, var_init=1.0):
        super().__init__()
        self.s_branch = Branch(feat_s, hidden, out_len, var_init)   # seasonal
        self.c_branch = Branch(feat_c, hidden, out_len, var_init)   # contextual

    def forward(self, z_s, z_c):
        mu_s, lv_s = self.s_branch(z_s)
        mu_c, lv_c = self.c_branch(z_c)
        return mu_s, lv_s, mu_c, lv_c


def masked_nll(mu, log_var, x_obs, x_den, mask):
    """论文 Eq.(6) 的掩码 NLL：mask=1 的正常点回归观测 x，mask=0 的疑似异常点回归去噪估计 x_hat。

    r_t = x_t*m_t + x_hat_t*(1-m_t)
    L   = mean_t [ log sigma_t^2 + (r_t - mu_t)^2 / sigma_t^2 ]

    log_var 是 log sigma^2；mask 在此处直接参与参考值的构造（不是旁路元数据）。
    """
    ref = x_obs * mask + x_den * (1.0 - mask)      # 掩码混合参考值，Eq.(6) 的括号项
    var = torch.exp(log_var) + 1e-6
    return (log_var + (ref - mu) ** 2 / var).mean()


# ---------------------------------------------------------------------------
# 5. 训练
# ---------------------------------------------------------------------------

def train(model, data, cfg, H, device, tag=''):
    z_s, z_c, obs, den, msk = data
    n = z_s.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    bs = cfg['batch_size']
    history = []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(n)
        total, nb, n_masked = 0.0, 0, 0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            n_masked += int((msk[j] == 0).sum())     # 本 epoch 训练批里真正走"去噪参考值"的目标点数
            mu_s, lv_s, mu_c, lv_c = model(z_s[j], z_c[j])
            loss = (masked_nll(mu_s, lv_s, obs[j], den[j], msk[j])
                    + masked_nll(mu_c, lv_c, obs[j], den[j], msk[j]))   # L = L_s + L_c
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        history.append(total / nb)
        print(f"  {tag}epoch {epoch}/{cfg['epochs']}  masked-NLL = {history[-1]:.4f}  "
              f"masked pts in batches = {n_masked}")
    return history


# ---------------------------------------------------------------------------
# 6. 异常打分与评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def anomaly_score(model, x_in, x_ref, x_den, msk, lo, hi, cfg, H, device):
    """滑窗打分：score_t = |x_t - mu_t| / sigma_t，跨窗取均值；输出覆盖 [lo, hi)。

    打分始终用**观测值** x_t 与预测均值比较（论文评分口径），与训练时掩码混合的参考值无关。
    返回 (fused, seasonal, contextual) 三组分数，后两组对应论文的 w/o C-LSTM / w/o S-LSTM 消融。
    """
    model.eval()
    w_s = cfg['seasonal_window']
    n_out = hi - lo - w_s + 1
    acc = {k: np.zeros(n_out) for k in ('fused', 's', 'c')}
    cnt = np.zeros(n_out)
    for t in range(lo, hi - w_s + 1, cfg['score_stride']):
        seg = min(w_s, n_out - (t - lo))               # 末尾窗口截断
        if seg <= 0:
            break
        zs, zc, _, _, _ = make_sample(x_in, x_ref, x_den, msk, t, H, w_s,
                                      cfg['context_window'], cfg['context_stride'],
                                      cfg['use_covariate'])
        zs = torch.as_tensor(zs[None], device=device)
        zc = torch.as_tensor(zc[None], device=device)
        mu_s, lv_s, mu_c, lv_c = model(zs, zc)
        mu_f = 0.5 * (mu_s + mu_c)                              # 双分支融合：均值
        var_f = 0.5 * (torch.exp(lv_s) + torch.exp(lv_c))        # 与方差
        obs = torch.as_tensor(x_ref[t:t + seg][None], device=device)
        for k, mu, lv in (('fused', mu_f, torch.log(var_f)),
                          ('s', mu_s, lv_s), ('c', mu_c, lv_c)):
            sigma = torch.sqrt(torch.exp(lv)).clamp_min(1e-3)
            s = (obs - mu[:, :seg]).abs() / sigma[:, :seg]
            acc[k][t - lo:t - lo + seg] += s[0].cpu().numpy()
        cnt[t - lo:t - lo + seg] += 1.0
    return {k: v / np.maximum(cnt, 1.0) for k, v in acc.items()}


def point_adjust(pred, label):
    """point adjustment：段内任一点被命中则该段整体判为命中（Best-F1/Delay-F1 采用的协议）。"""
    out = pred.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], label, [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        if pred[s:e].any():
            out[s:e] = True
    return out


def f1_of(pred, label):
    tp = np.sum(pred & (label == 1))
    fp = np.sum(pred & (label == 0))
    fn = np.sum(~pred & (label == 1))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9), prec, rec


def best_f1(score, label, n_grid=300, adjust=False):
    """遍历阈值取最大 F1；adjust=True 时采用 point adjustment 协议。"""
    best = 0.0
    for th in np.quantile(score, np.linspace(0.50, 0.9999, n_grid)):
        pred = score > th
        if adjust:
            pred = point_adjust(pred, label)
        best = max(best, f1_of(pred, label)[0])
    return best


def f1_at(score, label, th, adjust=False):
    pred = score > th
    if adjust:
        pred = point_adjust(pred, label)
    return f1_of(pred, label)


# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    device = torch.device('cpu')
    print(f"PyWavelets available: {HAS_PYWT}")

    cfg = dict(n=3000, train_lo=40, train_hi=1600, test_lo=1800, test_hi=2800,
               seasonal_window=8, context_window=4, context_stride=2,
               n_seasonal_win=4, hidden=32, epochs=3, batch_size=32,
               lr=1.5e-2, score_stride=2, use_covariate=True)
    H = cfg['n_seasonal_win'] * cfg['seasonal_window']     # 历史长度 = m * w_s

    x, label = make_series(n=cfg['n'], period=24, train_lo=cfg['train_lo'], train_hi=cfg['train_hi'],
                           anomaly_from=cfg['test_lo'], n_point=8, n_ramp=3,
                           n_point_train=10, n_ramp_train=1, seed=SEED)
    x_hat = wavelet_denoise(x, wavelet='db4', level=3)      # 步骤 1：小波噪声分解
    msk = anomaly_mask(x, x_hat)                            # mask=1 为正常点
    tr_seg = slice(cfg['train_lo'], cfg['train_hi'])
    n_tr_inj = int((label[tr_seg] == 1).sum())
    print(f"series: n={cfg['n']}, injected anomalies: train region [{cfg['train_lo']}, {cfg['train_hi']}) "
          f"= {n_tr_inj} pts ({100.0 * label[tr_seg].mean():.2f}%), "
          f"whole series {100.0 * label.mean():.2f}%")
    print(f"denoise: ||x - x_hat||_inf={np.abs(x - x_hat).max():.3f}, "
          f"model input = denoised series x_hat")
    flag = msk == 0
    print(f"mask (kappa=3 * MAD residual): {int(flag.sum())}/{msk.size} points flagged as suspected "
          f"anomalies, recall on all injected anomalies = {(msk[label == 1] == 0).mean():.3f}")
    print(f"  train region: {int((msk[tr_seg] == 0).sum())} flagged / {n_tr_inj} injected "
          f"(recall {(msk[tr_seg][label[tr_seg] == 1] == 0).mean():.3f})")

    X_in, X_ref, X_den = x_hat, x, x_hat     # 输入用去噪序列；参考值=观测 x 与去噪 x_hat 按掩码混合
    # 预处理：用训练段统计量做 z-score 标准化（避免频域特征量纲失衡）
    m0 = X_in[cfg['train_lo']:cfg['train_hi']].mean()
    s0 = X_in[cfg['train_lo']:cfg['train_hi']].std() + 1e-8
    X_in, X_ref, X_den = (X_in - m0) / s0, (X_ref - m0) / s0, (X_den - m0) / s0
    print(f"normalize: train mean={m0:.4f}, train std={s0:.4f}")
    # 用去噪残差得到的小波噪声尺度热启动方差头（sigma^2 初值）
    ns = noise_scale(X_ref)
    var_init = float(ns ** 2)
    print(f"wavelet noise scale (standardized) = {ns:.4f} -> var_init = {var_init:.6f}")
    tr = [make_sample(X_in, X_ref, X_den, msk, t, H, cfg['seasonal_window'],
                      cfg['context_window'], cfg['context_stride'],
                      cfg['use_covariate'])
          for t in range(cfg['train_lo'], cfg['train_hi'])]
    stack = lambda k: torch.as_tensor(np.stack([s[k] for s in tr]), device=device)
    train_data = (stack(0), stack(1), stack(2), stack(3), stack(4))   # z_s, z_c, x, x_hat, mask
    print(f"train samples: {train_data[0].shape[0]}, "
          f"S feat dim={train_data[0].shape[-1]}, C feat dim={train_data[1].shape[-1]}, "
          f"history H={H}, horizon={cfg['seasonal_window']}")

    torch.manual_seed(SEED)                     # 真实掩码跑与对照跑共用同一初始化
    model = CSLSTMs(train_data[0].shape[-1], train_data[1].shape[-1],
                    hidden=cfg['hidden'], out_len=cfg['seasonal_window'],
                    var_init=var_init).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"CS-LSTMs params={n_par} (smoke scale; the paper reports ~600K)")

    # 掩码接线自检：同一批输入、同一组权重，只有掩码在变，损失必须随之改变
    with torch.no_grad():
        chk = [make_sample(X_in, X_ref, X_den, msk, t, H, cfg['seasonal_window'],
                           cfg['context_window'], cfg['context_stride'], cfg['use_covariate'])
               for t in list(range(cfg['train_lo'], cfg['train_lo'] + 300))
               + list(range(cfg['test_lo'], cfg['test_lo'] + 300))]
        ob = torch.as_tensor(np.stack([s[2] for s in chk]), device=device)
        dn = torch.as_tensor(np.stack([s[3] for s in chk]), device=device)
        mk = torch.as_tensor(np.stack([s[4] for s in chk]), device=device)
        mu_chk, lv_chk = model.s_branch(
            torch.as_tensor(np.stack([s[0] for s in chk]), device=device))
        g = torch.Generator().manual_seed(SEED)
        m_rand = (torch.rand(mk.shape, generator=g) > 0.5).to(mk.dtype)
        m_one, m_zero = torch.ones_like(mk), torch.zeros_like(mk)

        def _nll_at(m_):
            r_ = ob * m_ + dn * (1.0 - m_)                       # Eq.(6) 的掩码混合参考值
            return (lv_chk + (r_ - mu_chk) ** 2 / (torch.exp(lv_chk) + 1e-6)).mean().item()

        l_one, l_zero, l_rand = _nll_at(m_one), _nll_at(m_zero), _nll_at(m_rand)
        ref_real = ob * mk + dn * (1.0 - mk)                     # 训练实际使用的参考值
        d_mix = (ref_real - ob).norm().item()                    # 与"纯观测参考值"的差异
        d_rand = (ob * m_rand + dn * (1 - m_rand) - ob).norm().item()
        d_zero = (ob * m_zero + dn * (1 - m_zero) - ob).norm().item()
        spread = max(l_one, l_zero, l_rand) - min(l_one, l_zero, l_rand)
    print("\nmask wiring self-check (identical inputs, only mask varies):")
    print(f"  check batch = 300 train + 300 test windows, masked points in batch = {int((mk == 0).sum())}")
    print(f"  masked NLL   mask=all-1 : {l_one:.6f}")
    print(f"  masked NLL   mask=all-0 : {l_zero:.6f}")
    print(f"  masked NLL   mask=random: {l_rand:.6f}")
    print(f"  spread(max-min) = {spread:.6f}  -> mask changes the loss: {spread > 1e-6}")
    print(f"  ||mixed_target - x_obs||_2 : real-mask={d_mix:.4f}, all-0={d_zero:.4f}, "
          f"random={d_rand:.4f}  -> mixing is active: {d_mix > 0.0}")

    print("training (dual-branch masked NLL, L = L_s + L_c):")
    hist_real = train(model, train_data, cfg, H, device)

    # 对照跑：掩码强制全 1，Eq.(6) 退化为普通 NLL。数据、初始化、优化器设置完全相同，
    # 唯一变量是掩码——它给出"掩码是否真的改变了训练轨迹"的端到端证据。
    torch.manual_seed(SEED)
    model_ctl = CSLSTMs(train_data[0].shape[-1], train_data[1].shape[-1],
                        hidden=cfg['hidden'], out_len=cfg['seasonal_window'],
                        var_init=var_init).to(device)
    data_ctl = train_data[:4] + (torch.ones_like(train_data[4]),)
    print("control run (mask forced to all-1; identical data / init / optimizer):")
    hist_ctl = train(model_ctl, data_ctl, cfg, H, device, tag='control: ')

    lo, hi = cfg['test_lo'], cfg['test_hi']
    sc = anomaly_score(model, X_in, X_ref, X_den, msk, lo, hi, cfg, H, device)
    score = sc['fused']
    lab = label[lo:lo + score.size]
    tr_score = anomaly_score(model, X_in, X_ref, X_den, msk,
                             cfg['train_lo'], 800, cfg, H, device)['fused']
    tr_lab = label[cfg['train_lo']:cfg['train_lo'] + tr_score.size]
    print(f"\nscore region [{lo}, {lo + score.size}): "
          f"{int(lab.sum())} anomalous / {lab.size} points ({100.0 * lab.mean():.2f}%)")
    m_te = msk[lo:lo + score.size]
    print(f"residual mask over test region: flagged {int((m_te == 0).sum())} points "
          f"({100.0 * (1.0 - m_te.mean()):.2f}%), "
          f"recall on injected anomalies = {(m_te[lab == 1] == 0).mean():.3f}")
    print(f"calibration on train region [{cfg['train_lo']}, {cfg['train_lo'] + tr_score.size}): "
          f"mean score = {tr_score.mean():.3f} "
          f"(normal pts {tr_score[tr_lab == 0].mean():.3f}, injected pts {tr_score[tr_lab == 1].mean():.3f})")
    print(f"mean score: normal={score[lab == 0].mean():.3f}, "
          f"anomalous={score[lab == 1].mean():.3f}")
    print(f"score quantiles: p50={np.quantile(score, .5):.3f}, "
          f"p90={np.quantile(score, .9):.3f}, p99={np.quantile(score, .99):.3f}")

    best = best_f1(score, lab)
    best_pa = best_f1(score, lab, adjust=True)
    f1_3s, p3, r3 = f1_at(score, lab, 3.0)
    print(f"Best-F1 (threshold swept)        = {best:.4f}")
    print(f"Best-F1 (point adjustment)       = {best_pa:.4f}")
    print(f"F1 @ fixed |z|>3 (point-wise)    = {f1_3s:.4f}  (P={p3:.3f}, R={r3:.3f})")
    print("ablation (single branch), Best-F1: "
          f"S-LSTM only = {best_f1(sc['s'], lab):.4f}, "
          f"C-LSTM only = {best_f1(sc['c'], lab):.4f}, "
          f"fused = {best:.4f}")

    # 端到端 A/B：同一份数据、同一初始化、同一优化器，只改掩码
    sc_ctl = anomaly_score(model_ctl, X_in, X_ref, X_den, msk, lo, hi, cfg, H, device)
    best_ctl = best_f1(sc_ctl['fused'], lab)
    best_pa_ctl = best_f1(sc_ctl['fused'], lab, adjust=True)
    print("\nmask A/B (same data / init / optimizer; only the mask differs):")
    print(f"  final masked-NLL : real mask = {hist_real[-1]:+.4f}  all-1 control = {hist_ctl[-1]:+.4f}  "
          f"delta = {hist_real[-1] - hist_ctl[-1]:+.4f}")
    print(f"  Best-F1          : real mask = {best:.4f}  all-1 control = {best_ctl:.4f}  "
          f"delta = {best - best_ctl:+.4f}")
    print(f"  Best-F1 (PA)     : real mask = {best_pa:.4f}  all-1 control = {best_pa_ctl:.4f}  "
          f"delta = {best_pa - best_pa_ctl:+.4f}")
    print("  epoch-wise masked-NLL (real/control): "
          + " | ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(hist_real, hist_ctl)))
    print("sample predictions vs observations (first 8 scored points):")
    for i in range(8):
        print(f"    t={lo + i:5d}  x={X_ref[lo + i]:8.3f}  "
              f"score={score[i]:7.3f}  label={int(lab[i])}")

    # 可视化（Agg 后端，只存盘不显示）
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    o = slice(lo, lo + score.size)
    ax[0].plot(np.arange(lo, lo + score.size), X_ref[o], lw=0.8, color='black', label='observed x')
    ax[0].plot(np.arange(lo, lo + score.size), X_in[o], lw=0.8, color='tab:blue', label='denoised x_hat')
    ax[0].set_ylabel('value')
    ax[0].legend(loc='upper right', fontsize=8)
    ax[1].plot(np.arange(lo, lo + score.size), score, lw=0.8, color='tab:red', label='anomaly score')
    ax[1].axhline(3.0, ls='--', c='gray', lw=0.8, label='threshold 3')
    ax[1].set_ylabel('|x - mu| / sigma')
    ax[1].legend(loc='upper right', fontsize=8)
    ax[2].fill_between(np.arange(lo, lo + score.size), 0, lab, step='mid', color='tab:orange')
    ax[2].set_ylabel('injected label')
    ax[2].set_xlabel('time step')
    fig.tight_layout()
    fig_path = 'cs_lstms_smoke_scores.png'
    fig.savefig(fig_path, dpi=120)
    print(f"figure saved: {fig_path}")
    print(f"elapsed: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
```

### 实测运行结果

环境：Windows 11，`D:\py\Python3\python.exe`（Python 3.11.5 + torch 2.13.0+cu126，CPU 推理），PyWavelets 1.8.0，numpy 2.3.5，matplotlib 3.11.0。冒烟配置（`hidden=32`, `w=8`, `epochs=3`），脚本自报耗时 3.8~4.0 s（4 次复跑；含真实掩码跑与全 1 对照跑两次训练；进程挂钟约 6.5 s 含导入开销）。除耗时行外，各次运行的 stdout 逐字符相同（两次训练的初始化与 `randperm` 都由固定种子重置，掩码自检用固定种子的本地 `Generator`，不受全局 RNG 影响）。下面是其中一次运行的完整 stdout：

```
PyWavelets available: True
series: n=3000, injected anomalies: train region [40, 1600) = 58 pts (3.72%), whole series 5.70%
denoise: ||x - x_hat||_inf=3.383, model input = denoised series x_hat
mask (kappa=3 * MAD residual): 7/3000 points flagged as suspected anomalies, recall on all injected anomalies = 0.041
  train region: 2 flagged / 58 injected (recall 0.034)
normalize: train mean=10.7150, train std=2.2602
wavelet noise scale (standardized) = 0.1156 -> var_init = 0.013356
train samples: 1560, S feat dim=18, C feat dim=10, history H=32, horizon=8
CS-LSTMs params=13476 (smoke scale; the paper reports ~600K)

mask wiring self-check (identical inputs, only mask varies):
  check batch = 300 train + 300 test windows, masked points in batch = 32
  masked NLL   mask=all-1 : 101.024373
  masked NLL   mask=all-0 : 93.271578
  masked NLL   mask=random: 96.737386
  spread(max-min) = 7.752795  -> mask changes the loss: True
  ||mixed_target - x_obs||_2 : real-mask=6.8890, all-0=20.8722, random=14.7871  -> mixing is active: True
training (dual-branch masked NLL, L = L_s + L_c):
  epoch 1/3  masked-NLL = 16.6863  masked pts in batches = 16
  epoch 2/3  masked-NLL = -1.3827  masked pts in batches = 16
  epoch 3/3  masked-NLL = -2.3021  masked pts in batches = 16
control run (mask forced to all-1; identical data / init / optimizer):
  control: epoch 1/3  masked-NLL = 16.8209  masked pts in batches = 0
  control: epoch 2/3  masked-NLL = -1.4096  masked pts in batches = 0
  control: epoch 3/3  masked-NLL = -2.2456  masked pts in batches = 0

score region [1800, 2793): 113 anomalous / 993 points (11.38%)
residual mask over test region: flagged 5 points (0.50%), recall on injected anomalies = 0.044
calibration on train region [40, 793): mean score = 0.561 (normal pts 0.528, injected pts 1.013)
mean score: normal=1.082, anomalous=2.606
score quantiles: p50=0.852, p90=2.811, p99=5.217
Best-F1 (threshold swept)        = 0.4721
Best-F1 (point adjustment)       = 0.9456
F1 @ fixed |z|>3 (point-wise)    = 0.4352  (P=0.525, R=0.372)
ablation (single branch), Best-F1: S-LSTM only = 0.5300, C-LSTM only = 0.4013, fused = 0.4721

mask A/B (same data / init / optimizer; only the mask differs):
  final masked-NLL : real mask = -2.3021  all-1 control = -2.2456  delta = -0.0564
  Best-F1          : real mask = 0.4721  all-1 control = 0.5023  delta = -0.0302
  Best-F1 (PA)     : real mask = 0.9456  all-1 control = 0.9617  delta = -0.0161
  epoch-wise masked-NLL (real/control): 16.686/16.821 | -1.383/-1.410 | -2.302/-2.246
sample predictions vs observations (first 8 scored points):
    t= 1800  x=  -0.201  score=  0.072  label=0
    t= 1801  x=   0.192  score=  0.299  label=0
    t= 1802  x=   0.426  score=  0.192  label=0
    t= 1803  x=   0.688  score=  0.272  label=0
    t= 1804  x=   0.952  score=  0.340  label=0
    t= 1805  x=   0.862  score=  0.489  label=0
    t= 1806  x=   1.172  score=  0.131  label=0
    t= 1807  x=   1.075  score=  0.099  label=0
figure saved: cs_lstms_smoke_scores.png
elapsed: 4.0 s
```

读法：三个 epoch 的训练损失从 16.69 降到 −2.30。若残差恰好等于小波噪声尺度 $\hat{\sigma}_{\text{noise}} = 0.1156$，该损失项取 $\log\hat{\sigma}_{\text{noise}}^2 + 1 = -3.32$（噪声水平参考值，不是理论下界——$\sigma_t$ 有下界 $\sigma_{\min}$ 时损失可以继续下降）。末损失 −2.30 **高于**该参考值，说明 3 个 epoch 的预算内方差头没有走到噪声尺度：训练段正常点平均分数 0.528、注入点 1.013，测试段正常点 1.082，校准量级仍接近 1，没有虚高，但末损失不能读成"已收敛到噪声水平"。

**掩码在训练中确实被消费**，证据有两层。损失层面：同一批输入、同一组权重，只换掩码，全 1 / 全 0 / 随机给出 101.024373 / 93.271578 / 96.737386 三个互不相同的损失，混合参考值与纯观测参考值的差异范数为 6.89（真实掩码）/ 20.87（全 0）/ 14.79（随机）。训练轨迹层面：同一份数据、同一初始化、同一优化器，只把掩码强制成全 1 再训一次，末损失由 −2.3021 变到 −2.2456（Δ = −0.0564），Best-F1 由 0.4721 变到 0.5023，PA-F1 由 0.9456 变到 0.9617；每个 epoch 训练批中被掩码命中的目标点数，真实掩码下为 16，对照下为 0。差异存在且可复现，但幅度小、方向也不构成改进——**这个 A/B 证明的是掩码被真正消费，而不是掩码提升分数**。原因在掩码规则本身：训练区间注入 58 个异常（占该区间 3.72%），κ=3 的残差阈值只标出其中 2 个（召回 0.034；测试段 5/113，召回 0.044），绝大多数异常仍以观测值身份进入参考值。

检测能力方面，无 point adjustment 的 Best-F1 为 0.472，加 PA 后 0.946 —— 这个对比本身就是 point adjustment 虚高效应的演示。本条目这一跑里融合并没有胜过单分支（S-LSTM 单支 0.530 高于融合 0.472），3 个 epoch 的预算不足以让融合权重显现优势，不能据此判断论文的融合设计。**这是 smoke 规模的复现（13,476 参数、3 个 epoch、两次训练），不是论文基准成绩的复现**；论文在 Yahoo 上的 Best-F1 为 0.885（对照 FCVAE 的 0.854）。

### 论文主结果（Table 1，供对标）

| 数据集 | CS-LSTMs Best-F1 | CS-LSTMs Delay-F1 | FCVAE Best-F1 | FCVAE Delay-F1 |
|--------|------------------|-------------------|---------------|----------------|
| Yahoo | 0.885 | 0.878 | 0.854 | 0.839 |
| KPI | 0.936 | 0.879 | 0.924 | 0.851 |
| WSD | 0.910 | 0.857 | 0.805 | 0.696 |
| NAB | 0.996 | 0.918 | 0.972 | 0.899 |

相对当时最优对比方法，Best-F1 提升 +3.1% / +1.2% / +10.5% / +0.6%，Delay-F1 提升 +3.9% / +2.8% / +16.1% / +0.7%（按 Yahoo / KPI / WSD / NAB 顺序）。消融（Table 5，Best-F1）：去掉 C-LSTM 得 0.864 / 0.923 / 0.856 / 0.985，去掉 S-LSTM 得 0.717 / 0.904 / 0.762 / 0.987，去掉协变量得 0.826 / 0.925 / 0.840 / 0.986，去掉噪声分解与掩码得 0.868 / 0.913 / 0.858 / 0.972。S-LSTM 在 Yahoo 与 WSD 上的贡献最大（分别掉 0.168 与 0.148），是整套设计的承重墙。

## 参考文献

[1] ZHANG L, LI Q, YANG Y, et al. Contextual and seasonal LSTMs for time series anomaly detection[C]//The Fourteenth International Conference on Learning Representations (ICLR 2026). 2026. arXiv:2602.09690.

[2] DONOHO D L, JOHNSTONE I M. Ideal spatial adaptation by wavelet shrinkage[J]. Biometrika, 1994, 81(3): 425-455. DOI: 10.1093/biomet/81.3.425.

[3] DONOHO D L. De-noising by soft-thresholding[J]. IEEE Transactions on Information Theory, 1995, 41(3): 613-627. DOI: 10.1109/18.382009.

[4] SALINAS D, FLUNKERT V, GASTHAUS J, et al. DeepAR: probabilistic forecasting with autoregressive recurrent networks[J]. International Journal of Forecasting, 2020, 36(3): 1181-1191. DOI: 10.1016/j.ijforecast.2019.07.001.

[5] WANG Z, PEI C, MA M, et al. Revisiting VAE for unsupervised time series anomaly detection: a frequency perspective[C]//Proceedings of the ACM Web Conference 2024 (WWW '24). New York: ACM, 2024: 3096-3105. DOI: 10.1145/3589334.3645710.

[6] REN H, XU B, WANG Y, et al. Time-series anomaly detection service at Microsoft[C]//Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19). New York: ACM, 2019: 3009-3017. DOI: 10.1145/3292500.3330680.

[7] AHMAD S, LAVIN A, PURDY S, et al. Unsupervised real-time anomaly detection for streaming data[J]. Neurocomputing, 2017, 262: 134-147. DOI: 10.1016/j.neucom.2017.04.070.

[8] KIM S, CHOI K, CHOI H S, et al. Towards a rigorous evaluation of time-series anomaly detection[C]//Proceedings of the AAAI Conference on Artificial Intelligence, 2022, 36(7): 7194-7201. DOI: 10.1609/aaai.v36i7.20680.
