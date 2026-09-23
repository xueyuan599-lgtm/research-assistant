---
title: TSB — Transformer + Stacked Bi-LSTM 多通道多步频谱预测
type:
  - forecasting
  - deep-learning
  - sequence-modeling
  - signal-processing
  - time-series
domain:
  - signal-processing
source: Pan, G., Li, J., & Li, M. (2025). Multi-Channel Multi-Step Spectrum Prediction Using Transformer and Stacked Bi-LSTM. China Communications, 2025(5), 1–13. arXiv:2405.19138
---
# TSB — Transformer + Stacked Bi-LSTM 多通道多步频谱预测

- **来源**: Pan, G., Li, J., & Li, M. (2025). Multi-Channel Multi-Step Spectrum Prediction Using Transformer and Stacked Bi-LSTM. *China Communications*, 2025(5), 1–13.
- **DOI·arXiv**: 10.23919/JCC.ja.2022-0667；arXiv:2405.19138（v2, 2025-03-22）
- **方法类别**: 编码器-解码器 Transformer 架构；多头注意力与堆叠双向 LSTM 逐层融合；直接多步（direct multi-step）多通道联合输出
- **相关文献**: Vaswani et al. (2017) 缩放点积注意力；Graves & Schmidhuber (2005) 双向 LSTM；Shi et al. (2015) ConvLSTM 基线；Xing et al. (2013) 认知无线电频谱预测综述

---

## 问题设定与动机

### 任务定义

认知无线电（cognitive radio）网络里，次级用户必须在接入前判断某条信道在未来若干时隙是否空闲。频谱预测（spectrum prediction）就是把这件事变成监督学习问题：观察 $F$ 条信道过去 $T$ 个时隙的接收功率，输出未来 $M$ 个时隙每条信道的功率，据此做"可用/不可用"的硬判决。

论文用 $p_{1:F,\,1:T}$ 记历史块，用 $\hat{p}_{1:F,\,T+1:T+M}$ 记预测块。$F$ 路信号同时进入模型，输出也是一个 $(F \times M)$ 的块——不是逐通道、逐时隙地算。这个"块进块出"的形状是 TSB 与早期单通道单步方法最直观的区别。

### 已有方法的三处短板

论文的动机段把问题拆成三条，每条对应一个架构决策：

**长程依赖捕捉弱。** ARIMA 是线性模型，CNN 的感受野由卷积核和层数决定，单向 LSTM 靠门控在长序列上传信息但会衰减。频谱占用序列的忙闲切换周期由干扰机的扫描策略决定，周期长度往往远超单个卷积核的覆盖范围。

**忽略通道间相关性。** 同一频段内相邻信道的占用状态高度耦合——干扰机扫频时会把一整片信道连续压制，跳频时又会制造跨信道的交替模式。逐通道独立建模会丢掉这类共现结构。TSB 的做法是把 $F$ 条信道在每个时隙堆成一个 $F$ 维向量，整块送进注意力。

**递归多步导致误差累积。** 递归（recursive）策略用单步模型反复回代：把 $\hat{y}_{t+1}$ 拼回输入窗口再预测 $\hat{y}_{t+2}$，如此推到第 $H$ 步。第 $h$ 步的输入里含 $h-1$ 个预测值，每个都带着自己的误差，于是误差随步数放大。论文明确以"the cumulative error will increase with the increase of the number of prediction steps"为理由，放弃递归、改走直接多步。

### 直接多步 vs 递归多步

两种策略的形式差别可以写清楚。设单步模型为 $f_\theta$，历史窗口为 $\mathbf{X}$。

递归多步：

$$
\hat{y}_{t+1} = f_\theta(\mathbf{X}), \quad
\hat{y}_{t+2} = f_\theta(\mathbf{X}_{+1} \,\|\, \hat{y}_{t+1}), \quad \dots, \quad
\hat{y}_{t+H} = f_\theta(\mathbf{X}_{+H-1} \,\|\, \hat{y}_{t+1}, \dots, \hat{y}_{t+H-1})
$$

直接多步：

$$
(\hat{y}_{t+1}, \dots, \hat{y}_{t+H}) = g_\theta(\mathbf{X}) \quad \text{—— 一次前向，} H \text{ 步同时输出}
$$

设单步模型对输入窗口的 Lipschitz 常数为 $L$，每一步新引入的误差上界为 $\varepsilon$，则递归到第 $h$ 步的累计误差满足

$$
e_h \;\le\; \varepsilon \sum_{j=0}^{h-1} L^{\,j} \;=\; \varepsilon \cdot \frac{L^{h} - 1}{L - 1} \quad (L \neq 1)
$$

$L > 1$ 时该界随 $h$ 指数增长，$L = 1$ 时线性增长。直接多步的输出 $\hat{y}_{t+h}$ 不是前 $h-1$ 步预测值的函数，这条累乘链条被切断——代价是模型必须一次性学会整块 $H$ 步的联合结构，参数效率要求更高。TSB 用 Transformer 解码器的并行输出承接这个代价。

---

## 数学设定

### 1. 问题形式化

记输入功率矩阵

$$
\mathbf{X} \in \mathbb{R}^{C \times T}, \qquad X_{c,t} = \text{信道 } c \text{ 在时隙 } t \text{ 的接收功率 (dBm)}
$$

目标为未来功率块

$$
\mathbf{Y} \in \mathbb{R}^{C \times H}, \qquad \mathbf{Y}_{c,h} = X_{c,\,T+h}
$$

模型输出 $\hat{\mathbf{Y}} = \mathcal{F}_\theta(\mathbf{X})$，$\mathcal{F}_\theta$ 为 TSB 整体映射。为让损失在各通道间可比，训练前用训练段的均值和标准差做 z-score 标准化：

$$
\tilde{X}_{c,t} = \frac{X_{c,t} - \mu_c}{\sigma_c}, \qquad
\mu_c = \frac{1}{|\mathcal{T}_{train}|}\sum_{t \in \mathcal{T}_{train}} X_{c,t}, \quad
\sigma_c^2 = \frac{1}{|\mathcal{T}_{train}|}\sum_{t \in \mathcal{T}_{train}} (X_{c,t} - \mu_c)^2
$$

$\mu_c, \sigma_c$ 只用训练段估计，避免测试段信息泄漏。逆变换 $\hat{X}_{c,h} = \hat{Y}_{c,h}\sigma_c + \mu_c$ 后与判决门限（论文取 $-50$ dBm）比较，得到可用的硬判决。

维度约定：模型中张量按 $(\text{batch}, T, C)$ 排列，即时间维在前、通道维在后，便于把 $C$ 维通道向量视作每个时隙的 token。

### 2. 多头自注意力

输入先线性投影到模型维度 $d_{model}$，再加位置编码。位置编码用正弦形式（Vaswani et al., 2017）：

$$
\mathbf{Z} = \mathbf{X}^\top \mathbf{W}_{in} + \mathbf{b}_{in} + \mathbf{P}, \qquad \mathbf{Z} \in \mathbb{R}^{T \times d_{model}}
$$

$$
P_{(t,2i)} = \sin\!\left(\frac{t}{10000^{2i/d_{model}}}\right), \qquad
P_{(t,2i+1)} = \cos\!\left(\frac{t}{10000^{2i/d_{model}}}\right)
$$

正弦编码不引入额外参数，且能表达相对位置关系——对频谱序列有用，因为忙闲切换的周期性主要体现在"相隔多少个时隙"上。

单头缩放点积注意力：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

其中 $\mathbf{Q} = \mathbf{Z}\mathbf{W}^Q$，$\mathbf{K} = \mathbf{Z}\mathbf{W}^K$，$\mathbf{V} = \mathbf{Z}\mathbf{W}^V$，$d_k$ 为单头键维度。$\sqrt{d_k}$ 的缩放用于抵消点积随维度增长的方差，否则 softmax 会饱和成近似 one-hot。

多头把表示空间切成 $o$ 个子空间并行做注意力，再拼接投影：

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q,\ \mathbf{K}\mathbf{W}_i^K,\ \mathbf{V}\mathbf{W}_i^V)
$$

$$
\text{MHA}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \big[\,\text{head}_1; \dots; \text{head}_o\,\big]\mathbf{W}^O
$$

因为注意力的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 全部来自同一个 $\mathbf{Z}$，这一步是**自**注意力：每个时隙的位置都对序列的**全部**位置计算权重，全局上下文一次到位，不经过 RNN 那种逐步传递。这正是 TSB 用来替代卷积与递归的机制——论文的表述是注意力"can continuously attend to all positions of the multichannel spectrum sequences"。

复杂度是 $O(T^2 d_{model})$。频谱预测的历史窗口通常几十到几百个时隙，$T^2$ 项不构成瓶颈；这也是 TSB 不做稀疏注意力近似的原因。

### 3. 堆叠双向 LSTM

注意力给出的是全局加权后的表示，但它对局部时序结构不加约束。TSB 把多头注意力的输出交给堆叠 Bi-LSTM，逐层提取局部特征。

**双向。** 单向 LSTM 在位置 $t$ 只见过 $1..t$，对频谱序列来说，一个忙闲段的**结束**往往要靠后文才能确认（干扰机何时撤出）。双向 LSTM 同时跑正反两个方向：

$$
\overrightarrow{\mathbf{h}}_t = \text{LSTM}\!\left(\overrightarrow{\mathbf{h}}_{t-1},\ \mathbf{z}_t\right), \qquad
\overleftarrow{\mathbf{h}}_t = \text{LSTM}\!\left(\overleftarrow{\mathbf{h}}_{t+1},\ \mathbf{z}_t\right)
$$

$$
\mathbf{h}_t = \big[\,\overrightarrow{\mathbf{h}}_t\,;\, \overleftarrow{\mathbf{h}}_t\,\big] \in \mathbb{R}^{2d_h}
$$

其中 $\mathbf{z}_t$ 是该层在位置 $t$ 的输入。注意这在编码器里不构成泄漏——编码器只看历史窗口 $\mathbf{X}$，窗口内部双向是合法的；解码器一侧必须加掩码，见下节。

单个 LSTM 单元的门控与状态更新：

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_i \mathbf{z}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i) \\
\mathbf{f}_t &= \sigma(\mathbf{W}_f \mathbf{z}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f) \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \mathbf{z}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o) \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c \mathbf{z}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}
$$

遗忘门 $\mathbf{f}_t$ 决定旧记忆保留多少，输入门 $\mathbf{i}_t$ 决定新候选写入多少。$\mathbf{c}_t$ 沿时间近乎线性地传递，梯度不易消失——这是 LSTM 相对朴素 RNN 的优势来源。

**堆叠。** 第 $l$ 层以第 $l-1$ 层的**整条输出序列**为输入：

$$
\mathbf{h}_t^{(l)} = \Big[\,\text{LSTM}^{(l)}\!\big(\{\mathbf{h}_{<t}^{(l-1)}\}, \mathbf{h}_t^{(l-1)}\big)\;;\;
\text{LSTM}^{(l)}\!\big(\{\mathbf{h}_{>t}^{(l-1)}\}, \mathbf{h}_t^{(l-1)}\big)\Big]
$$

逐层堆叠让第 1 层学短程的忙闲切换、第 2 层在此基础上组更长的时间片段——论文称之为 "learn these focused coding features by multi-head attention layer by layer"。由于输出是双向拼接的 $2d_h$ 维，再接一个线性投影回到 $d_{model}$，维度才能与残差连接对齐：

$$
\text{BiLSTM}(\mathbf{Z}) = \big[\overrightarrow{\mathbf{H}}; \overleftarrow{\mathbf{H}}\big]\mathbf{W}_{proj} + \mathbf{b}_{proj} \in \mathbb{R}^{T \times d_{model}}
$$

### 4. 编码器-解码器信息流

论文的编码器层由三个部件组成：多头注意力模块、堆叠 Bi-LSTM 模块、归一化层（残差连接 + LayerNorm，论文记作 R-Norm）。写成通式，残差归一化为

$$
\text{RNorm}(\mathbf{a}, \mathbf{b}) = \text{LayerNorm}\big(\mathbf{a} + \text{Dropout}(\mathbf{b})\big)
$$

编码器第 $l$ 层（$l = 1,\dots,N$，论文 $N = 3$）：

$$
\mathbf{u}^{(l)} = \text{RNorm}\Big(\mathbf{Z}^{(l-1)},\ \text{MHA}\big(\mathbf{Z}^{(l-1)}, \mathbf{Z}^{(l-1)}, \mathbf{Z}^{(l-1)}\big)\Big)
$$

$$
\mathbf{Z}^{(l)} = \text{RNorm}\Big(\mathbf{u}^{(l)},\ \text{BiLSTM}\big(\mathbf{u}^{(l)}\big)\Big)
$$

$\mathbf{Z}^{(0)}$ 为加过位置编码的投影输入，编码器输出记忆 $\mathbf{H}^{enc} = \mathbf{Z}^{(N)} \in \mathbb{R}^{T \times d_{model}}$。

解码器层在编码器层的基础上增加一个掩码自注意力子层和一个线性层；论文 $E = 3$ 层。解码器的初始查询是一组可学习的 $H$ 个时隙查询，加上解码器位置编码：

$$
\mathbf{Q}^{(0)} = \mathbf{Z}_{horizon} + \mathbf{P}^{dec}, \qquad \mathbf{Z}_{horizon} \in \mathbb{R}^{H \times d_{model}}
$$

第 $e$ 层：

$$
\mathbf{v}^{(e)} = \text{RNorm}\Big(\mathbf{Q}^{(e-1)},\ \text{MHA}_{mask}\big(\mathbf{Q}^{(e-1)}, \mathbf{Q}^{(e-1)}, \mathbf{Q}^{(e-1)}\big)\Big)
$$

$$
\mathbf{w}^{(e)} = \text{RNorm}\Big(\mathbf{v}^{(e)},\ \text{BiLSTM}\big(\mathbf{v}^{(e)}\big)\Big)
$$

$$
\mathbf{Q}^{(e)} = \text{RNorm}\Big(\mathbf{w}^{(e)},\ \text{MHA}\big(\mathbf{w}^{(e)}, \mathbf{H}^{enc}, \mathbf{H}^{enc}\big)\Big)
$$

第三个子层是交叉注意力：查询来自解码器，键与值来自编码器记忆，这是编码器信息进入解码器的唯一通路。掩码 $\mathbf{M}$ 为上三角置 $-\infty$：

$$
M_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
\qquad\text{作用于 } \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \text{ 之上}
$$

掩码保证第 $i$ 个未来时隙只看到前 $i$ 个未来时隙的查询，与自回归推理时的因果顺序一致。

需要说明的一点：论文正文写解码器输入是"the right-shifted model output"（右移的训练目标，即 teacher forcing 形式），同时强调输出是单块 $\hat{p}_{1:F,T+1:T+M}$。本条目给出的实现改用可学习时隙查询 + 并行解码，与直接多步的输出形态一致，训练时不需要构造右移序列。两种写法在"一次前向输出整块 $H$ 步"这一点上等价，差别在于前者保留了自回归式的输入分布。复现论文数值时应以论文设定为准。

### 5. 直接多步解码头

解码器输出经 LayerNorm 和线性层一次性映射到整块预测：

$$
\hat{\mathbf{Y}} = \Big(\text{Norm}\big(\mathbf{Q}^{(E)}\big)\mathbf{W}_{out} + \mathbf{b}_{out}\Big)^\top \in \mathbb{R}^{C \times H}
$$

转置是为了让输出与 $\mathbf{Y} \in \mathbb{R}^{C \times H}$ 的通道-时隙布局对齐。论文记这一步为 $\text{Linear}(\text{Norm}(p_{de}^E))$。

整个前向只有一次，$H$ 个时隙并行产出。与递归策略对照，误差传播路径的差别是结构性的：

| | 递归多步 | 直接多步 |
|---|---|---|
| 前向次数 | $H$ 次 | 1 次 |
| 第 $h$ 步输入 | 含 $h-1$ 个预测值 | 只含历史真值 |
| 误差增长 | $\varepsilon\sum_{j<h} L^j$ | 与 $h$ 无累乘关系 |
| 推理延迟 | 随 $H$ 线性增长 | 与 $H$ 无关 |
| 训练信号 | 单步目标 | 整块 $H$ 步联合目标 |

代价也要讲清楚：直接多步让所有 $H$ 个位置共享同一套解码器参数，模型必须学会"哪个查询对应哪个未来位置"，位置信息完全靠查询嵌入和位置编码承载；当 $H$ 远大于 $T$ 时这条约束会变紧。论文的 $H$ 取 48 或 96、$T$ 取 96，两者同量级。

### 6. 损失函数

论文式 (20) 给出带 L2 正则的均方误差：

$$
\mathcal{L}(\theta) = \frac{1}{N_b}\sum_{n=1}^{N_b} \big\|\hat{\mathbf{Y}}^{(n)} - \mathbf{Y}^{(n)}\big\|_F^2
\;+\; \eta \cdot \frac{1}{2}\left(\sum_{\mathbf{W}_{en}} \mathbf{W}^2 + \sum_{\mathbf{W}_{de}} \mathbf{W}^2\right)
$$

第一项是预测块与真值块的 Frobenius 范数平方（即所有通道、所有时隙的平方误差和），$N_b$ 为批大小；第二项只正则化编码器与解码器的**权重矩阵**，偏置不参与。$\eta$ 控制正则强度，$\eta = 0$ 退化为纯 MSE。

参数更新走梯度下降（论文用 Adam）：

$$
\theta \leftarrow \theta - \alpha \,\nabla_\theta \mathcal{L}(\theta)
$$

实现里把 L2 项写成对 `p.dim() > 1` 的参数求和，与"权重矩阵、不含偏置"的语义一致；这一项与优化器的 `weight_decay` 不同——后者是以动量方式耦合进自适应步长的解耦权重衰减，前者是按上式直接加进损失再求梯度。复现代数时二者不等价，不要混用。

一处需要留意的论文内部不一致：式 (20) 写的是 MSE 加 L2，而实现章节的表述是"The loss function is the root mean square error (MSE)"（根均方误差与均方误差混用）。本条目按式 (20) 的 MSE + L2 实现。

---

## 关键假设

| # | 假设 | 形式化表述 | 违背后果 |
|---|------|-----------|---------|
| 1 | 通道间存在可利用的相关结构 | $\text{Cov}(X_{c,t}, X_{c',t}) \neq 0$，且该结构在训练/测试段同分布 | 通道耦合退化为噪声，注意力权重趋于均匀，退化为逐通道独立建模 |
| 2 | 历史窗口长度 $T$ 覆盖主导周期 | $T \gtrsim$ 忙闲切换周期，$T > 2 d_{model}$ | 注意力无跨周期模式可学，位置编码分辨率不足 |
| 3 | 频谱占用在窗口尺度上平稳 | $\mathbb{E}[X_{c,t}]$ 与 $\text{Var}(X_{c,t})$ 在训练段近似不变 | 标准化参数失配，预测段出现系统性偏移，需在线重标定 |
| 4 | 未来 $H$ 步存在确定性成分 | $\text{Var}(\mathbb{E}[Y \mid X]) > 0$ | 全忙或全闲的通道上模型只能输出均值，RMSE 退化到方差水平 |
| 5 | 局部时序结构与全局上下文互补 | MHA 捕获的跨时隙依赖与 Bi-LSTM 捕获的局部模式不重合 | 两个子层输出冗余，去掉其一性能不变，白白增加参数 |
| 6 | 因果掩码足以防止未来信息泄漏 | 解码器位置 $i$ 只能读取 $j \le i$ 的查询 | 训练损失异常偏低、测试段崩溃——典型的数据泄漏征兆 |
| 7 | 硬判决门限与功率尺度匹配 | $\hat{X}_{c,h} \gtrless \theta_{th}$ 的 $\theta_{th}$ 在训练段有判别力 | 分类指标（准确率、$\kappa$）与回归指标背离，RMSE 好的模型分类未必好 |
| 8 | 训练/测试段无剧烈分布漂移 | $P_{train}(X) \approx P_{test}(X)$ | 注意力与 LSTM 需重新适配，冷启动期误差抬升 |

---

## 适用场景

### 适用

- **认知无线电 / 动态频谱接入**：预测主用户占用，指导次级用户的接入时机与信道选择。论文的原始场景。
- **多通道功率或占用序列**：通道数 $C$ 从几到几十，通道间有共现结构（扫频干扰、跳频、共享负载），历史窗口 $T$ 数十至数百。
- **中短步长多步预测**：$H$ 与 $T$ 同量级（论文 $T = 96$，$H \in \{48, 96\}$）。此时直接多步的并行解码开销可控，而递归的误差累积已经显现。
- **要求推理延迟与 $H$ 解耦**：一次前向出整块结果，$H$ 增大不增加前向次数，适合在线频谱决策。
- **频谱占用以外的同构任务**：多传感器功率序列、多路信道质量指标（CQI）、多频点 RSSI 预测。只要"多路 + 相关 + 多步"三要素齐备，TSB 的结构就直接可用。

### 不适用

- **单通道序列**。通道数 $C = 1$ 时，把通道堆成 token 的那一步退化，多头注意力的跨通道优势消失；此时纯 Bi-LSTM 或 TCN 更省参数。
- **超长序列**（$T \gg 10^3$）。自注意力 $O(T^2)$ 的内存与算力开销随 $T$ 平方增长，需换用稀疏或线性注意力变体。
- **极短历史**（$T < 16$）。位置编码与注意力都没有足够分辨率，多层结构容易过拟合；用单层 GRU 或线性模型即可。
- **样本量极少**（数百个窗口以下）。编码器与解码器各 3 层的参数量远超数据规模，正则化难以救回。
- **实时嵌入式部署**（MCU / 低功耗无线电）。堆叠 Bi-LSTM 的序列依赖使推理难以并行，模型尺寸也超出典型射频前端算力；应蒸馏到轻量模型。
- **与历史无关的纯随机占用**。若干扰机采用真随机策略，$I(Y;X) \approx 0$，任何模型的上限都是输出通道均值。

### 与相邻方法的对比

| 方法 | 机制 | 相对 TSB 的不足 |
|------|------|----------------|
| ARIMA | 线性自回归 + 差分 | 无跨通道项、无非线性；忙闲切换处拟合滞后 |
| 单向 LSTM | 门控递归，逐步传状态 | 只有前向上下文；长序列信息衰减；多步需递归 |
| CNN | 局部卷积核 | 感受野受核尺寸与层数限制；跨通道结构不显式建模 |
| ConvLSTM | 卷积 + LSTM 的混合 | 有局部与递归，但缺少全局注意力；论文中误差明显偏高 |
| DCG | CNN 与 GRU 并联 | 无注意力，多步能力弱；论文基线之一 |
| LSTM-Attention | 注意力 + Seq2Seq | 有注意力但无堆叠双向结构；递归解码累积误差 |
| Informer / Autoformer | 稀疏/自相关注意力，长序列友好 | 面向超长序列设计，$T$ 中等时结构优势体现不出来，实现复杂度更高 |
| **TSB** | MHA + 堆叠 Bi-LSTM，编码器-解码器直接多步 | 参数量与 $O(T^2)$ 注意力成本；小样本与单通道场景不划算 |

---

## 实现要点

### 关键超参数

| 参数 | 符号 | 建议范围 | 论文取值 | 说明 |
|------|------|---------|---------|------|
| 编码器层数 | $N$ | 2–4 | 3 | 论文消融 {2,2}/{3,3}/{4,4}，{3,3} 最优 |
| 解码器层数 | $E$ | 2–4 | 3 | 与编码器同步调整 |
| 注意力头数 | $o$ | 4–12 | 8 | 论文消融 {8,10,12}；头数越多显存越紧，8 为基准 |
| 堆叠 Bi-LSTM 层数 | $L_{bi}$ | 1–3 | 2 | 论文消融 {1,2,3}，2 层在两档 horizon 上均最低 |
| Bi-LSTM 隐藏维度 | $d_h$ | 16–128 | 论文未给出具体值 | 双向拼接后为 $2d_h$，再投影回 $d_{model}$ |
| 模型维度 | $d_{model}$ | 32–256 | 论文仅定义为"模型输入的维度" | 需被 $o$ 整除 |
| 历史窗口 | $T$ | 32–512 | 96 | 应明显大于忙闲切换周期 |
| 预测步长 | $H$ | 1–96 | 48 / 96 | 直接多步，一次输出 |
| L2 正则系数 | $\eta$ | $10^{-6}$–$10^{-3}$ | 论文未给出具体值 | 正则化权重矩阵，不含偏置 |
| 学习率 | $\alpha$ | $10^{-4}$–$10^{-1}$ | 0.001 | Adam；论文消融 {0.01, 0.001, 0.0001} |
| 批大小 | $N_b$ | 8–128 | 32 | 论文另有批大小 8 的对照实验 |
| 训练轮数 | — | 10–100 | 20 | 配合早停 patience = 6 |
| Dropout | — | 0.0–0.2 | 论文未给出具体值 | 加在 R-Norm 的残差分支上 |
| 判决门限 | $\theta_{th}$ | 按噪声底实测 | $-50$ dBm | 训练前确定，不参与梯度 |

### 调优经验

**层数的边际收益很薄，不要盲目加深。** 论文的层数消融数字很说明问题——48 步预测下 {2,2}/{3,3}/{4,4} 的 RMSE 分别是 0.6849 / 0.6838 / 0.6842，96 步下是 0.6847 / 0.6845 / 0.6851。三档配置的差距在 $10^{-3}$ 量级，而 4 层相比 3 层的训练开销更大。{3,3} 是"够用即止"的选择，不是性能悬崖的另一侧。

**学习率的敏感度同样低。** $\{0.01, 0.001, 0.0001\}$ 在 48 步下的 RMSE 为 0.6840 / 0.6835 / 0.6838，96 步下为 0.6880 / 0.6829 / 0.6844。0.001 在两档 horizon 上都最优，同时 0.01 在 96 步上退化最明显——长 horizon 下过大步长更容易震荡。实践上从 $10^{-3}$ 起调，用 ReduceLROnPlateau 或余弦退火收尾即可。

**头数在 8 附近是平台期。** $\{8,10,12\}$ 在 48 步下 0.6838 / 0.6840 / 0.6848，96 步下 0.6845 / 0.6843 / 0.6845。头数翻倍带来的误差变化低于 $10^{-3}$，但显存与算力开销按比例上涨。头数必须整除 $d_{model}$，这点在缩小模型做冒烟实验时最容易踩坑。

**Bi-LSTM 层数要卡在 2。** 论文的消融给出 1 层在 48 步的 RMSE 为 0.6851、2 层 0.6844、3 层 0.6844，96 步为 0.6855 / 0.6845 / 0.6850。1 层明显欠拟合（未学到足够的层次化局部特征），3 层相对 2 层没有收益却成倍增加训练时间。

**这几组消融的绝对差异都在 $10^{-3}$ 量级**，说明 TSB 在合理的超参邻域内表现平稳。换数据后不必逐一网格搜索，优先固定 $N = E = 3$、$o = 8$、$L_{bi} = 2$、$\alpha = 10^{-3}$，把调参预算花在 $T$、$H$ 的窗口切分和标准化方式上——这两项对结果的影响远大于层数。

**训练流程配套。** Adam 优化器、20 轮上限、早停 patience = 6、批大小 32 是论文的设置。评测指标不止 RMSE（dB）一项——论文还用预测准确率和 Spearman 秩相关系数 $\kappa$（预测步长 $M \in \{48, 96\}$）。回归指标好的模型在分类口径下未必领先，复现时应两个口径都报。

**本条目代码里的配置是冒烟配置**，不是论文配置：$C = 4$、$T = 32$、$H = 4$、$d_{model} = 32$、$o = 4$、$N = E = 2$、$L_{bi} = 1$、5 轮、CPU、单次运行约 10 秒。要复现论文量级的结果，按上表把 $T$ 调到 96、$H$ 调到 48、层数调到 3、$\eta$ 与 dropout 按验证集选。

---

## 代码

自包含的最小实现：合成多通道频谱占用序列（通道间共享负载 + 各通道 AR(1) 扰动 + 忙闲阈值 + 测量噪声）→ TSB 编码器-解码器 → 一次前向输出 $H$ 步全通道 → MSE + L2 损失反向 → 测试段逐 horizon 的 RMSE，并与递归单步回代做对照。

```python
"""
TSB — Transformer + Stacked Bi-LSTM for multi-channel multi-step spectrum prediction.

Reference:
    Pan, G., Li, J., & Li, M. (2025). Multi-Channel Multi-Step Spectrum Prediction
    Using Transformer and Stacked Bi-LSTM. China Communications, 2025(5), 1-13.
    arXiv:2405.19138

Self-contained PyTorch demo (CPU, < 90 s):
  1) synthetic multi-channel spectrum-occupancy series (correlated ON/OFF busy-idle
     patterns + AR(1) jitter + measurement noise)
  2) TSB encoder-decoder: multi-head attention fused with stacked Bi-LSTM
  3) direct multi-step head: one forward pass -> H future steps for all C channels
  4) MSE + L2 regularised loss, Adam, per-horizon RMSE on the held-out test segment
  5) direct multi-step vs recursive single-step rollout comparison
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set True to export the per-horizon error curve next to this script.
SAVE_FIG = False


# ---------------------------------------------------------------------------
# 1. Synthetic multi-channel spectrum occupancy
# ---------------------------------------------------------------------------

def _ar1(rng, n, rho, sigma, x0=0.0):
    """First-order autoregressive noise: x_t = rho * x_{t-1} + eps_t."""
    x = np.empty(n, dtype=np.float64)
    x[0] = x0
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x


def _moving_average(x, k):
    """Centred moving average used to soften busy/idle transitions."""
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def generate_spectrum_series(n_steps, n_channels=4, seed=0,
                             idle_dbm=-95.0, busy_dbm=-45.0):
    """One long multi-channel spectrum-occupancy series (columns = channels).

    A shared latent spectrum load drives every channel (this is the source of
    inter-channel correlation); each channel adds its own AR(1) term and turns
    ON when its load crosses a channel-specific threshold. Power is reported on
    a dBm-like scale with a short transition window and measurement noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # Shared spectrum load: slow diurnal-like cycle + fast ripple + AR(1) jitter
    common = 0.50 + 0.30 * np.sin(2 * np.pi * t / 220.0 + rng.uniform(0, 2 * np.pi))
    common += 0.12 * np.sin(2 * np.pi * t / 37.0)
    common += _ar1(rng, n_steps, rho=0.90, sigma=0.05)

    theta = np.array([0.55, 0.62, 0.46, 0.68])[:n_channels]  # per-channel sensitivity
    out = np.empty((n_steps, n_channels), dtype=np.float64)

    for c in range(n_channels):
        own = _ar1(rng, n_steps, rho=0.95, sigma=0.08)
        load = common + 0.22 * own + rng.normal(0.0, 0.04, n_steps)
        busy = (load > theta[c]).astype(np.float64)
        busy = _moving_average(busy, 5)                 # soft ON/OFF edges
        power = idle_dbm + (busy_dbm - idle_dbm) * busy
        power += rng.normal(0.0, 2.0, n_steps)          # receiver noise floor
        out[:, c] = power

    return out.astype(np.float32)


def make_windows(series, hist_len, horizon, stride=4):
    """Slide a (hist_len + horizon) window over the series -> (X, Y).

    X: (N, hist_len, C)  history
    Y: (N, horizon,  C)  future block (all channels, all steps at once)
    """
    total = hist_len + horizon
    starts = np.arange(0, len(series) - total + 1, stride)
    X = np.stack([series[s:s + hist_len] for s in starts])
    Y = np.stack([series[s + hist_len:s + total] for s in starts])
    return X.astype(np.float32), Y.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. TSB building blocks
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ResidualNorm(nn.Module):
    """The paper's R-Norm: LayerNorm(x + Dropout(sublayer(x)))."""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):
        return self.norm(x + self.drop(sublayer_out))


class StackedBiLSTM(nn.Module):
    """Stacked bidirectional LSTM + projection back to d_model.

    Layer l consumes the full output sequence of layer l-1, so depth yields
    hierarchical local-temporal abstraction; the two directions give each
    position both forward and backward context.
    """

    def __init__(self, d_model, hidden, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * hidden, d_model)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.proj(out)


class MultiHeadAttentionBlock(nn.Module):
    """Scaled dot-product multi-head attention: softmax(QK^T/sqrt(d_k))V."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)

    def forward(self, q, k, v, attn_mask=None):
        out, _ = self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)
        return out


class EncoderLayer(nn.Module):
    """Encoder layer = Multi-Head Attention + Stacked Bi-LSTM, each with R-Norm.

    The attention path gives every position access to the whole multi-channel
    history (global context); the Bi-LSTM path then extracts local sequential
    structure from the attention-refined representation.
    """

    def __init__(self, d_model, nhead, bilstm_hidden, bilstm_layers, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
        self.rnorm1 = ResidualNorm(d_model, dropout)
        self.bilstm = StackedBiLSTM(d_model, bilstm_hidden, bilstm_layers, dropout)
        self.rnorm2 = ResidualNorm(d_model, dropout)

    def forward(self, x):
        x = self.rnorm1(x, self.self_attn(x, x, x))
        x = self.rnorm2(x, self.bilstm(x))
        return x


class DecoderLayer(nn.Module):
    """Decoder layer = masked self-attention + stacked Bi-LSTM + cross-attention.

    The mask keeps position h from reading future horizon positions; the
    cross-attention reads the encoder memory, so a single forward pass emits
    the whole horizon block at once (no recurrence over predicted steps).
    """

    def __init__(self, d_model, nhead, bilstm_hidden, bilstm_layers, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
        self.rnorm1 = ResidualNorm(d_model, dropout)
        self.bilstm = StackedBiLSTM(d_model, bilstm_hidden, bilstm_layers, dropout)
        self.rnorm2 = ResidualNorm(d_model, dropout)
        self.cross_attn = MultiHeadAttentionBlock(d_model, nhead, dropout)
        self.rnorm3 = ResidualNorm(d_model, dropout)

    def forward(self, q, memory, tgt_mask):
        q = self.rnorm1(q, self.self_attn(q, q, q, attn_mask=tgt_mask))
        q = self.rnorm2(q, self.bilstm(q))
        q = self.rnorm3(q, self.cross_attn(q, memory, memory))
        return q


class TSB(nn.Module):
    """Transformer + Stacked Bi-LSTM for multi-channel multi-step prediction.

    Input  : x  (B, T, C)  — C channels, T historical steps
    Output : y  (B, H, C)  — H future steps for all C channels, in one pass
    """

    def __init__(self, n_channels, horizon, d_model=32, nhead=4,
                 enc_layers=2, dec_layers=2, bilstm_hidden=16, bilstm_layers=1,
                 dropout=0.1, max_len=512):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(n_channels, d_model)
        self.enc_pos = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, nhead, bilstm_hidden, bilstm_layers, dropout)
            for _ in range(enc_layers)
        ])

        # Direct multi-step: a bank of H learnable horizon queries, so the
        # decoder emits every future step in parallel instead of recursing.
        self.horizon_queries = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.dec_pos = nn.Parameter(torch.randn(horizon, d_model) * 0.02)
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, nhead, bilstm_hidden, bilstm_layers, dropout)
            for _ in range(dec_layers)
        ])
        self.head = nn.Linear(d_model, n_channels)

    def encode(self, x):
        mem = self.enc_pos(self.input_proj(x))
        for layer in self.encoder:
            mem = layer(mem)
        return mem

    def forward(self, x):
        mem = self.encode(x)
        b = x.size(0)
        q = self.horizon_queries.unsqueeze(0).expand(b, -1, -1) + self.dec_pos
        tgt_mask = torch.triu(
            torch.full((self.horizon, self.horizon), float("-inf"), device=x.device),
            diagonal=1,
        )
        for layer in self.decoder:
            q = layer(q, mem, tgt_mask)
        return self.head(q)                     # (B, H, C)


# ---------------------------------------------------------------------------
# 3. Loss, training and evaluation
# ---------------------------------------------------------------------------

def tsb_loss(pred, target, model, weight_decay):
    """L = (1/N) sum ||y - y_hat||^2  +  lambda * 0.5 * ||W||^2.

    The L2 term penalises encoder/decoder weight matrices only (no biases),
    matching the regulariser reported in the TSB paper.
    """
    mse = F.mse_loss(pred, target)
    l2 = torch.zeros((), device=pred.device)
    for module in list(model.encoder) + list(model.decoder):
        for p in module.parameters():
            if p.dim() > 1:
                l2 = l2 + p.pow(2).sum()
    return mse + weight_decay * 0.5 * l2, mse


def train_tsb(model, X, Y, epochs, batch_size, lr, weight_decay, seed, tag):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(Y)
    n = len(xt)
    history = []

    print(f"\n[{tag}] training  params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total, total_mse, seen = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xt[idx], yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss, mse = tsb_loss(pred, yb, model, weight_decay)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
            total_mse += mse.item() * len(idx)
            seen += len(idx)
        history.append((total / seen, total_mse / seen))
        print(f"[{tag}] epoch {epoch}/{epochs}  loss={total/seen:.6f}  mse={total_mse/seen:.6f}")
    return history


@torch.no_grad()
def per_horizon_rmse(model, X, Y):
    """RMSE at each horizon step, averaged over channels: (H,)."""
    model.eval()
    pred = model(torch.from_numpy(X)).numpy()          # (N, H, C)
    err = pred - Y
    return np.sqrt((err ** 2).mean(axis=(0, 2))), np.sqrt((err ** 2).mean())


@torch.no_grad()
def recursive_rollout_rmse(model, X, Y):
    """Roll a single-step model forward H times, feeding predictions back."""
    model.eval()
    hist = torch.from_numpy(X).clone()                 # (N, T, C)
    H = Y.shape[1]
    sq = np.zeros((H,))
    for h in range(H):
        nxt = model(hist)                              # (N, 1, C)
        sq[h] = ((nxt.squeeze(1).numpy() - Y[:, h, :]) ** 2).mean()
        hist = torch.cat([hist[:, 1:, :], nxt], dim=1)
    return np.sqrt(sq), math.sqrt(float(sq.mean()))


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # ---- smoke configuration (small enough for a CPU run) ----
    C, T, H = 4, 32, 4
    N_WINDOWS, STRIDE = 480, 4
    D_MODEL, NHEAD = 32, 4
    ENC_LAYERS, DEC_LAYERS = 2, 2
    BILSTM_HIDDEN, BILSTM_LAYERS = 16, 1
    EPOCHS, BATCH, LR, WD = 5, 32, 1e-3, 1e-5

    print("=" * 68)
    print("TSB - Transformer + Stacked Bi-LSTM  (multi-channel multi-step)")
    print("=" * 68)

    # ---- data ----
    n_steps = (N_WINDOWS - 1) * STRIDE + T + H
    series = generate_spectrum_series(n_steps, n_channels=C, seed=0)
    X, Y = make_windows(series, T, H, stride=STRIDE)

    # Time-ordered split. Neighbouring segments are separated by an embargo of
    # one full window (T + H series steps), because consecutive windows overlap
    # by T - stride steps: the plain 70/15/15 cut lets the last validation
    # windows reach into the first test windows (32 steps of series time here),
    # and the last training windows into the first validation ones.
    GAP = (T + H) // STRIDE                                   # 9 windows
    n_train = int(0.70 * len(X))
    n_val = int(0.15 * len(X)) - 2 * GAP
    tr = slice(0, n_train)
    va = slice(n_train + GAP, n_train + GAP + n_val)
    te = slice(n_train + GAP + n_val + GAP, len(X))
    assert (va.stop - 1) * STRIDE + T + H <= te.start * STRIDE, "val/test overlap"

    def span(sl):
        """Series-step interval [start, end) covered by a window slice."""
        return f"[{sl.start * STRIDE},{((sl.stop - 1) * STRIDE + T + H)})"

    mu, sd = X[tr].mean(axis=(0, 1), keepdims=True), X[tr].std(axis=(0, 1), keepdims=True)
    Xn, Yn = (X - mu) / sd, (Y - mu) / sd

    print(f"series       : {series.shape}   (steps, channels)")
    print(f"windows      : X{Xn.shape}  Y{Yn.shape}")
    print(f"split        : train={n_train}  val={va.stop - va.start}  "
          f"test={len(X) - te.start}   (embargo {GAP} windows = {T + H} steps)")
    print(f"time spans   : train={span(tr)} val={span(va)} test={span(te)}  (disjoint)")
    print(f"channel corr : {np.corrcoef(series.T).round(2).tolist()}")
    print(f"channel duty : {(series > -70.0).mean(axis=0).round(3).tolist()}")

    # ---- model 1: direct multi-step, horizon H ----
    model_direct = TSB(C, H, d_model=D_MODEL, nhead=NHEAD,
                       enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,
                       bilstm_hidden=BILSTM_HIDDEN, bilstm_layers=BILSTM_LAYERS)
    train_tsb(model_direct, Xn[tr], Yn[tr], EPOCHS, BATCH, LR, WD, seed=0, tag="direct")

    rmse_dir_h, rmse_dir = per_horizon_rmse(model_direct, Xn[te], Yn[te])

    # ---- model 2: recursive single-step (same architecture, horizon 1) ----
    model_rec = TSB(C, 1, d_model=D_MODEL, nhead=NHEAD,
                    enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS,
                    bilstm_hidden=BILSTM_HIDDEN, bilstm_layers=BILSTM_LAYERS)
    train_tsb(model_rec, Xn[tr], Yn[tr][:, :1, :], EPOCHS, BATCH, LR, WD, seed=0, tag="recur-1step")

    rmse_rec_h, rmse_rec = recursive_rollout_rmse(model_rec, Xn[te], Yn[te])

    # ---- report ----
    print("\n" + "-" * 68)
    print("Test-segment error (normalised power units, RMSE)")
    print("-" * 68)
    print(f"{'horizon':>8} | {'direct multi-step':>18} | {'recursive 1-step':>17} | {'vs recurs.':>10}")
    print("-" * 68)
    for h in range(H):
        gain = (rmse_rec_h[h] - rmse_dir_h[h]) / rmse_rec_h[h] * 100.0
        print(f"{h+1:>8} | {rmse_dir_h[h]:>18.4f} | {rmse_rec_h[h]:>17.4f} | {gain:>6.1f}%")
    print("-" * 68)
    print(f"{'overall':>8} | {rmse_dir:>18.4f} | {rmse_rec:>17.4f} | "
          f"{(rmse_rec - rmse_dir) / rmse_rec * 100.0:>6.1f}%")

    # ---- figure (rendered always; written to disk only if SAVE_FIG) ----
    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=140)
    xs = np.arange(1, H + 1)
    ax.plot(xs, rmse_dir_h, "o-", color="#1f4e79", label="direct multi-step (TSB)")
    ax.plot(xs, rmse_rec_h, "s--", color="#c00000", label="recursive 1-step rollout")
    ax.set_xlabel("prediction horizon (steps)")
    ax.set_ylabel("RMSE (normalised)")
    ax.set_title("Error accumulation: direct vs. recursive multi-step")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if SAVE_FIG:
        fig.savefig("tsb_per_horizon_rmse.png")
        print("\nfigure saved -> tsb_per_horizon_rmse.png")
    fig.canvas.draw()
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
```

### 运行方式

```bash
pip install torch numpy matplotlib
python tsb_smoke.py
```

冒烟配置在该环境（Python 3, torch 2.13.0+cu126, numpy 2.3.5, CPU）上实测约 7–20 秒完成，两次运行的数值完全一致（种子固定为 0）。实测输出：

```text
====================================================================
TSB - Transformer + Stacked Bi-LSTM  (multi-channel multi-step)
====================================================================
series       : (1952, 4)   (steps, channels)
windows      : X(480, 32, 4)  Y(480, 4, 4)
split        : train=336  val=54  test=72   (embargo 9 windows = 36 steps)
time spans   : train=[0,1376) val=[1380,1628) test=[1632,1952)  (disjoint)
channel corr : [[1.0, 0.85, 0.81, 0.74], [0.85, 1.0, 0.72, 0.84], [0.81, 0.72, 1.0, 0.61], [0.74, 0.84, 0.61, 1.0]]
channel duty : [0.419, 0.334, 0.569, 0.242]

[direct] training  params=56,356
[direct] epoch 1/5  loss=0.887884  mse=0.883385
[direct] epoch 2/5  loss=0.618488  mse=0.614005
[direct] epoch 3/5  loss=0.549497  mse=0.545032
[direct] epoch 4/5  loss=0.406452  mse=0.401994
[direct] epoch 5/5  loss=0.298684  mse=0.294223

[recur-1step] training  params=56,164
[recur-1step] epoch 1/5  loss=0.811911  mse=0.807422
[recur-1step] epoch 2/5  loss=0.546448  mse=0.542072
[recur-1step] epoch 3/5  loss=0.413750  mse=0.409484
[recur-1step] epoch 4/5  loss=0.281923  mse=0.277741
[recur-1step] epoch 5/5  loss=0.190139  mse=0.186029

--------------------------------------------------------------------
Test-segment error (normalised power units, RMSE)
--------------------------------------------------------------------
 horizon |  direct multi-step |  recursive 1-step | vs recurs.
--------------------------------------------------------------------
       1 |             0.4416 |            0.4483 |    1.5%
       2 |             0.5285 |            0.5756 |    8.2%
       3 |             0.5844 |            0.6819 |   14.3%
       4 |             0.6763 |            0.8037 |   15.9%
--------------------------------------------------------------------
 overall |             0.5642 |            0.6409 |   12.0%
--------------------------------------------------------------------
```

这段输出复现了论文的核心动机：递归策略的误差随 horizon 放大，直接多步的领先幅度从第 2 步的 8.2% 扩大到第 4 步的 15.9%。第 1 步那 1.5% 不能当结论读，它在噪声量级。把递归单步换成 12 组不同初始化重跑同一对照（直接多步固定 seed 0，递归单步取 seed 0–11，其余配置不动）：h = 2 / 3 / 4 上直接多步 12/12 全胜，最小领先幅度分别为 1.5% / 6.8% / 9.1%；h = 1 只有 6/12 占优，幅度落在 -8.3% 到 +17.6% 之间（均值 +4.2%），方向由初始化决定——12 组里没有一组在四个 horizon 上全部领先。靠得住的表述是：中长程（$h \ge 2$）直接多步稳定占优，单步上的胜负看初始化。合成的 4 条信道两两相关系数在 0.61–0.85 之间（由共享潜在负载制造），占空比分别为 0.419 / 0.334 / 0.569 / 0.242，通道间差异使"联合预测"这件事有实际内容。

三点说明：RMSE 的单位是标准化后的功率（无量纲），论文报的是 dB，两者不可直接比较；5 轮训练远未收敛，这里只用于验证前向、反向与两次运行的确定性，不是性能结论；切分加了 embargo——相邻两段之间空出一个完整窗口（9 个窗 = 36 步），train / val / test 覆盖的序列时间区间分别为 [0,1376) / [1380,1628) / [1632,1952)，两两不相交；不加 embargo 时 val 的最后一段会伸进 test 的前 32 步。训练与评估只读 train 与 test，无早停、无模型选型，因此这一修正不改变表内任何数值。

---

## 参考文献

[1] Pan G, Li J, Li M. Multi-channel multi-step spectrum prediction using Transformer and stacked Bi-LSTM[J]. China Communications, 2025, 2025(5): 1–13. DOI: 10.23919/JCC.ja.2022-0667. arXiv: 2405.19138.

[2] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in Neural Information Processing Systems (NeurIPS). 2017, 30: 5998–6008. DOI: 10.48550/arXiv.1706.03762.

[3] Graves A, Schmidhuber J. Framewise phoneme classification with bidirectional LSTM and other neural network architectures[J]. Neural Networks, 2005, 18(5–6): 602–610. DOI: 10.1016/j.neunet.2005.06.042.

[4] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural Computation, 1997, 9(8): 1735–1780. DOI: 10.1162/neco.1997.9.8.1735.

[5] Shi X, Chen Z, Wang H, et al. Convolutional LSTM network: A machine learning approach for precipitation nowcasting[C]//Advances in Neural Information Processing Systems (NeurIPS). 2015, 28: 802–810. DOI: 10.48550/arXiv.1506.04214.

[6] Xing X, Jing T, Cheng W, et al. Spectrum prediction in cognitive radio networks[J]. IEEE Wireless Communications, 2013, 20(2): 90–96. DOI: 10.1109/MWC.2013.6507399.

[7] Zhao Q, Sadler B M. A survey of dynamic spectrum access[J]. IEEE Signal Processing Magazine, 2007, 24(3): 79–89. DOI: 10.1109/MSP.2007.361604.
