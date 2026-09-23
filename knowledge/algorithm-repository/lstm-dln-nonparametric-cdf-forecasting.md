---
title: LSTM-DLN — 深度格网网络的多步长非参数 CDF 预测
type:
  - forecasting
  - deep-learning
  - uncertainty-quantification
  - regression
  - sequence-modeling
  - time-series
domain:
  - energy-systems
  - generic-ml
source: Erdmann, N., Bentsen, L. Ø., Stenbro, R., Riise, H. N., Warakagoda, N. D., & Engelstad, P. E. (2026). Multi-Horizon Time Series Forecasting of Non-Parametric CDFs with Deep Lattice Networks. AAAI 2026, 40(1), 255–264. arXiv:2511.13756
---
# LSTM-DLN — 深度格网网络的多步长非参数 CDF 预测

- **来源**: Erdmann, N., Bentsen, L. Ø., Stenbro, R., Riise, H. N., Warakagoda, N. D., & Engelstad, P. E. (2026). Multi-Horizon Time Series Forecasting of Non-Parametric CDFs with Deep Lattice Networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(1), 255–264.
- **DOI·arXiv**: 10.1609/aaai.v40i1.36986 · arXiv:2511.13756
- **方法类别**: 概率预测 / 不确定性量化 / 单调约束神经网络 / 序列建模
- **相关文献**: Gupta et al. (2016) 格网回归与插值查找表；You et al. (2017) 深度格网网络；Sill (1998) 单调网络；Chernozhukov et al. (2010) 无交叉分位数曲线；Koenker & Bassett (1978) 分位数回归；Gneiting & Raftery (2007) CRPS。序列嵌入部分见库内条目 `lstm.md`。

## 问题设定与动机

### 分位数交叉为何破坏 CDF 合法性

概率预测的目标不是给出一个数，而是给出条件分布的完整刻画。修正式（simultaneous / implicit quantile regression, SQR）让网络以分位数水平 $\tau$ 为输入，一次前向输出该 $\tau$ 对应的分位数值，循环 $\tau$ 即得逆 CDF：

$$
Y = \bigl(f(\chi, \tau_0),\ f(\chi, \tau_1),\ \dots,\ f(\chi, \tau_q)\bigr)
$$

只要 $f$ 对 $\tau$ 没有约束，就存在 $\tau_a < \tau_b$ 而 $f(\chi,\tau_a) > f(\chi,\tau_b)$ 的可能——这就是**分位数交叉**（quantile crossover）。后果不是"精度略差"，而是产物**不合法**：

- 由 $\{\hat Q(\tau_j)\}$ 反推 CDF 时，映射 $\tau_j \mapsto \hat F^{-1}(\tau_j)$ 不再单调，无法定义一个合法的分布函数；
- 中心区间 $[\hat I_{c/2}, \hat I_{1-c/2}]$ 可能下界大于上界，PICP（区间覆盖率）与可靠性图失去意义；
- CRPS 的弹球损失求和近似要求 $\hat Q(\tau)$ 单调，否则该近似与 $\int (\hat F - \mathbb 1)^2$ 不连通。

论文实测：无约束的 LSTM-NN（每个 $\tau$ 一个独立输出头）在**整个测试集、全部分位数**上的平均交叉比例为 **3.075 %，标准差 0.0744**。一个常用的补救是往损失里加交叉惩罚项，论文明确否定这条路——惩罚项不提供任何保证，而"只有这种保证才能给出合法的逆 CDF"（原文：Only such a guarantee yields a legitimate inverse CDF）。本文条目的核心，就是把单调性做成**参数化结构**而不是训练目标。

### 递归多步 vs 直接多步

多步长预测有两条路线。递归（recursive）训练单步模型再自回归展开，误差沿步长累积，多步分布需要 Monte Carlo 传播，推理成本随 horizon 线性增长。本文取**直接多步**（direct multi-horizon）：一次前向同时给出 $h$ 个 horizon 的分布，回溯窗 $w = 96$ h（4 天），horizon $h = 36$ h。每个 horizon 的分位数输出共享同一个 LSTM 嵌入，horizon 之间的差异由未来已知外生特征（晴空指数、小时/星期的周期编码）承载。

### LSTM 与 DLN 的分工

论文把模型写成两段串联：$f_1$ 是 LSTM，负责把高维历史时序压成低维嵌入；$f_2$ 是 DLN，负责把这个嵌入与分位数水平映射成分位数值。$f_2$ 一次前向只给出完整 CDF 的一个分位数，循环 $\tau$ 得到精细逆 CDF（论文 Algorithm 1–2）。作者指出 $f_1$ 可替换为 DeepAR 或 Transformer，DLN 这一侧才是本文的适配对象。

## 数学设定

### 格网函数与插值权重

格网（lattice）是可单调约束的查找表。$D$ 维输入 $\chi \in [0,1]^D$、每维 $k$ 个关键点，对应的可训练参数为 $k^D$ 个格点值 $\theta$。格网函数的统一形式是**格点值的加权和**：

$$
f_l(\chi) \;=\; \theta^\top \psi(\chi) \;=\; \sum_{k=1}^{K} \theta_k\, \psi_k(\chi),
\qquad \psi_k(\chi) \ge 0,\quad \sum_{k=1}^{K} \psi_k(\chi) = 1
$$

$\chi$ 落在关键点之间时，$\psi_k$ 是插值权重，两个约束（非负、和为一）使它成为一组**凸坐标**。权重族有两种常见取法：单纯形（simplex）插值取 $x$ 所在单纯形的重心坐标；多线性（multilinear）插值取单位超立方体顶点 $v_k \in \{0,1\}^D$ 上的乘积权重（论文式 7）：

$$
\psi_k(\chi) \;=\; \prod_{d=0}^{D-1} \chi[d]^{\,v_k[d]}\,\bigl(1-\chi[d]\bigr)^{\,1-v_k[d]}
$$

$v_k[d]$ 取 0 或 1，故每维要么乘 $\chi[d]$ 要么乘 $1-\chi[d]$；顶点权重在多维上做双线性/多线性混合。Gupta et al. (2016) 同时测试过多线性与单纯形两种插值，本文只用多线性。

**格网函数对每个输入天然单调，原因是权重凸性 + 格点值的有序性**。固定其余维、只看第 $d$ 维：$f_l$ 是沿该维格点值序列的分段插值，插值权重非负且和为一，故输出落在相邻两个格点值的凸组合范围内。若沿该维的相邻格点值满足 $\theta_s \ge \theta_r$，则插值结果对该维单调不减；把"非减"换成"非增"即得单调递减。单调性是**格点值序关系**的性质，与输入取值无关——这正是它能给出硬保证的原因。

### 深度格网网络（DLN）

单个 $D$ 维格网需要 $k^D$ 个参数，$k=21$、$D=4$ 时已达 194 481 个，无法随特征数增长。You et al. (2017) 的 DLN 用三类可堆叠、可单独加约束的层解决这一矛盾：

1. **可约束线性嵌入层**（constrainable linear embedding）。把输入拆成单调部分 $\chi^q$ 与非单调部分 $\chi^m$，前者的权重矩阵保持非负（论文式 8）：

$$
\gamma = \begin{bmatrix} \theta^{q}\chi^{q} \\ \theta^{m}\chi^{m}\end{bmatrix} + b,
\qquad \theta^{q} \ge 0
$$

2. **校准层**（calibration）。一组并行的一维格网，每个特征单独做一次单调重映射。一维校准器是 $k$ 组 key–value 对 $(a \in \mathbb R^k,\ b \in \mathbb R^k)$，输入落在相邻两个 key 之间时在两个 value 之间线性插值；value 非减即得单调校准器。

3. **格网集成**（lattice ensemble）。一组格网 $S$，每个 $f_l : \mathbb R^{A} \to \mathbb R$ 只吃 $A$ 个特征（$A \subset D$），集成内各格网的输入子集互不相同；输出可以相加、取平均，或再送一层校准。

**为什么缓解维度灾难**：参数从单个 $k^D$ 降到约 $k^{A} \times |S|$，$A$ 通常取 2–4。论文搜索的格网输入维数为 $A \in \{2,3,4\}$、关键点 $k \in \{2,5,11,21\}$，最优组合为 $A=2$、$k=21$。作者也给出了代价边界的量化：最坏情形 $21^4 \times (128/4)$ 超过 600 万参数，DLN 参数随输入特征数膨胀，这正是它此前迟迟未进入时间序列预测领域的原因。

**单调性如何保持**：三类层都不破坏上游的单调关系——线性嵌入的单调部分权重非负、校准器 value 非减、集成内每个格网对指定维单调，因此复合函数对指定输入的单调性沿网络传递。实践上通过约束优化求解，论文用 **Dykstra 投影算法**在每次参数更新后把参数投回可行域。

### 单调约束下的同时分位数回归

本文的适配只改一处、但改在关键位置：**把分位数水平 $\tau$ 分发到集成的每一个子格网**。此前的做法只把分位数喂给其中一个格网，再指望约束向后传播；本文为每个格网保留一路分位数输入，使**每一个特征嵌入都与 CDF 建立了直接训练的关系**。于是每个子格网 $L_f$ 是二元函数 $L_f(u, \tau)$，$u$ 是特征嵌入投影到 $[0,1]$ 的标量，$\tau$ 是分位数水平，输出对 $\tau$ 单调不减。论文要求的硬约束是（式 5）：

$$
f(\chi, \tau_a) \;\le\; f(\chi, \tau_b) \qquad \forall\, \tau_a \le \tau_b
$$

把 $F$ 个子格网求和（或取平均），再过一个单调的输出校准层 $C$（复合两个非减函数仍非减）：

$$
\hat Q(\tau) \;=\; C\!\left(\sum_{f=1}^{F} L_f\bigl(u_f(\chi), \tau\bigr)\right),
\qquad \hat Q(\tau_a) \le \hat Q(\tau_b)\ \Leftrightarrow\ \tau_a \le \tau_b
$$

$\hat Q$ 单调 ⟹ 分位数区间 $[\hat Q(0.025), \hat Q(0.975)]$ 不会倒挂 ⟹ $\hat Q^{-1}$ 是一个合法的非参数 CDF，且这一性质对**任意参数取值**成立，不需要靠训练收敛来换。

与"逐 $\tau$ 独立回归"（论文中的 LSTM-QR 一行）的对比是理解收益的关键：

| 维度 | 逐 $\tau$ 独立回归（QR） | 同时/隐式回归（SQR，含 LSTM-DLN） |
|------|------------------------|--------------------------------|
| 输出结构 | 每个 $\tau$ 一个独立输出头 | 一个网络，$\tau$ 作为输入 |
| 训练信号 | 各 $\tau$ 分别优化 | $\tau \sim U(0,1)$ 采样，共享参数 |
| 交叉保证 | 无，只能事后检验 | DLN 版由参数化结构保证 |
| 单次前向 | 一个 $\tau$ | 一个 $\tau$（循环 $\tau$ 得完整逆 CDF） |
| 参数量 | 随 $\tau$ 个数线性增长 | 与 $\tau$ 个数无关 |

注意论文 Table 1 的实际结果：逐 $\tau$ 独立的 LSTM-QR 拿到了最好的 ACE（0.052），而 LSTM-DLN 以 0.061 居第二。**校准最好不等于分布合法**：LSTM-QR 是逐 $\tau$ 独立的输出头，跨 $\tau$ 没有任何约束，交叉风险并未被消除，只是在这个数据集上没触发。

### LSTM 嵌入与多步长输出

$f_1$ 把回溯窗压成嵌入向量（论文 hidden size 128、2 层、batch 64、Adam）：

$$
h_t = \mathrm{LSTM}\bigl(x_{t-L+1:t}\bigr) \in \mathbb R^{H_{\text{hid}}}
$$

对每个 horizon $h$ 与每个格网 $f$，用一个线性头把嵌入与**该 horizon 的未来已知外生特征** $e_h$（晴空指数、小时的 sin/cos 编码）映射成格网的特征输入：

$$
u_{h,f} = \sigma\bigl(\phi_{h,f}^\top [\,h_t;\ e_h\,] + b_{h,f}\bigr) \in (0,1)
$$

$$
\hat Q_h(\tau) = C\!\left(\sum_{f=1}^{F} L_f\bigl(u_{h,f},\ \tau\bigr)\right),
\qquad h = 1, \dots, H
$$

$\sigma$ 把特征压到格网定义域 $[0,1]$ 内。外生特征 $e_h$ 不可省：LSTM 只看历史，$h$ 步之后的昼夜相位必须由已知的日历/晴空信息提供。

### 弹球损失（pinball loss）

训练目标是分位数损失，逐样本形式为（论文式 2，$\gamma$ 为真值、$\gamma'$ 为预测）：

$$
l_\tau(\gamma, \gamma') =
\begin{cases}
\tau\,(\gamma - \gamma'), & \gamma - \gamma' \ge 0\\[2pt]
(1-\tau)\,(\gamma' - \gamma), & \text{其他}
\end{cases}
\;=\; \max\bigl\{\tau e,\ (\tau-1)e\bigr\},
\qquad e = \gamma - \gamma'
$$

两段写法与 $\max$ 写法完全等价：$e \ge 0$ 时 $\max$ 取 $\tau e$，$e < 0$ 时取 $(\tau-1)e = (1-\tau)|e|$。参数按 SQR 目标优化（论文式 4），$\tau$ 从 $U(0,1)$ 采样而不是只用固定的几个水平：

$$
\arg\min_{f}\ \frac{1}{N}\sum_{i=1}^{N} \mathbb E_{\tau_i \sim U(0,1)}\,\Bigl[\,l_{\tau_i}\bigl(\gamma_i,\ f(\chi_i, \tau_i)\bigr)\Bigr]
$$

论文按 minibatch 采 $\tau \sim U(0,1)$。这一步很要紧：只训练评估用的 11 个 $\tau$，中间水平拿不到梯度，逆 CDF 在 $\tau$ 之间会出现平台。

### CRPS 与评价指标

连续 CRPS 定义为预测 CDF $\hat F$ 与观测的点质量指示函数之间的 $L^2$ 距离（论文式 11）：

$$
\mathrm{CRPS}(\hat F, y) \;=\; \frac{1}{N}\sum_{i=1}^{N}\int_{-\infty}^{+\infty}\Bigl(\hat F_i(z) - \mathbb 1\{y_i \le z\}\Bigr)^{2}\,dz
$$

它与弹球损失有精确的积分等式（本条目校验过：对点质量 $F=\delta_0$、$y=1$，$2\int_0^1 \rho_\tau d\tau = 1$ 与直接积分 $\int(\cdot)^2 dz = 1$ 一致）：

$$
\mathrm{CRPS}(\hat F, y) \;=\; 2\int_0^1 l_\tau\bigl(y,\ \hat Q(\tau)\bigr)\,d\tau
$$

论文实现用有限分位数网格上的弹球损失求和近似该积分（论文式 12）。作者提示这个近似**可能偏向目标中位数附近的窄分布**——这解释了为什么论文中无约束的线性变体（LSTM-Lin、LSTM-CLin）能拿到最低 CRPS，却同时是最差的校准（ACE 0.173 / 0.190，CDF 过窄）。跨模型比较 CRPS 时要与 ACE 一起读。

区间覆盖率与校准误差：

$$
\mathrm{PICP}_c = \frac{1}{N}\sum_{i=1}^{N} s_i,
\qquad
s_i = \begin{cases}1, & \gamma_i \in \bigl[\hat I_{0.5-\frac{c}{2}},\ \hat I_{0.5+\frac{c}{2}}\bigr]\\[2pt] 0, & \text{其他}\end{cases}
$$

即名义中心区间覆盖率为 $c$ 时实际落入的比例；单水平形式下 $\mathrm{ACE}_\alpha = |\mathrm{PICP}_\alpha - (1-\alpha)|$，论文对一组中心区间取平均（式 14）：

$$
\mathrm{ACE} = \frac{1}{|C|}\sum_{c \in C}\bigl|c - \mathrm{PICP}_c\bigr|
$$

点预测指标取中位数 $\hat Q(0.5)$：$\mathrm{MAE} = \frac1N\sum_i|\gamma_i - \hat Q_i(0.5)|$，$\mathrm{RMSE} = \sqrt{\frac1N\sum_i(\gamma_i - \hat Q_i(0.5))^2}$。

论文还报告相对**智能持久性**（smart persistence）的技巧得分（式 9–10）：

$$
f_{\mathrm{sp}}(t) = \frac{x(t)}{o(t)}\cdot o(t+h),
\qquad
\mathrm{SS} = 1 - \frac{\mathrm{MSE}_f}{\mathrm{MSE}_p}
$$

$x$ 为观测辐照度、$o$ 为晴空辐照度：把当前时刻的晴空指数外推到 $t+h$ 时刻的晴空值上。$\mathrm{SS} > 0$ 表示优于该基线。

## 关键假设

- **$\tau$ 与特征解耦**：分位数水平只作为独立的格网输入通道，不与特征相乘。单调性对任意 $u$ 成立正是靠这一点——插值是格点值的凸组合，$u$ 取什么值都不影响沿 $\tau$ 的序关系。
- **输入落在 $[0,1]^D$**：格网在单位超立方体上定义，特征与 $\tau$ 都必须先归一化。论文只做 min-max 标准化；超出定义域需要额外的外推策略。
- **条件分布可由单调格网族逼近**：DLN 的表达能力来自少数低维格网的加性/集成组合，隐含"特征间交互阶数低"的假设。强高阶交互需要更多格网或更大 $A$，参数量随之指数上升。
- **训练期 $\tau$ 采样覆盖评估范围**：$\tau \sim U(0,1)$ 才能让中间水平有梯度。若只关心尾部（如 0.01/0.99），需要提高尾部采样密度，否则尾部逆 CDF 平坦、区间偏窄。
- **未来已知外生变量可用**：晴空辐照度、日历编码等必须在预测时刻可获得，否则多步长下昼夜相位无法确定。
- **同分布假设**：CRPS / PICP / ACE 的解释力依赖测试段与训练段同分布。论文刻意选 2016 年（与 2018–2020 训练段相关性最低的一年）作为测试集，验证段为 2017 年，这个切分本身就是对分布迁移的稳健性检验。

## 适用场景

- **光伏 / 风电功率与辐照度概率预测**：论文的目标任务。日前小时级太阳辐照度预测（挪威 20 个站点、2016–2020 五年小时数据、目标站为西海岸 11 号站），回溯窗 96 h、horizon 36 h。
- **需要完整预测分布而非单点的时序任务**：电力负荷、径流、电价、需求侧响应，凡下游需要分位数（备用容量、风险价值、储能充放电计划）的场景。
- **区间可靠性有硬要求的调度问题**：PICP 是可直接进约束的量，而合法的单调逆 CDF 是 PICP 有意义的前提。
- **带单调先验的低维交互回归**：任何"输出须对某输入单调"的机理型任务，可用同一套格网机制注入先验。
- **多步长同发预测**：储能调度、备用容量申报这类要求"未来 36 h 同时给出分布"的任务，直接多步比递归展开更省推理成本。

### 不适用

- **只需点预测**：单调保证的代价是参数量与推理时间。论文 Table 5 给出的相对关系：LSTM-DLN 931 375 参数、11 个分位数一次前向 2.604 s；LSTM-SMNN 448 228 参数、0.059 s；LSTM-NN 346 058 参数、0.010 s。DLN 的推理时间约为 SMNN 的 44 倍、NN 的 260 倍。点预测任务用 MLP 头即可。
- **高维稠密交互**：格网参数随输入维数指数增长（$k^A$），必须先把特征分组降维，否则参数爆炸。
- **极长序列与长记忆**：LSTM 嵌入部分在 >1000 步时会梯度衰减，应替换为 Transformer / DeepAR（论文明确说这一侧可替换），但替换后 DLN 侧的单调保证不变。
- **严格外推**：格网只在 $[0,1]^D$ 内有定义，训练集未覆盖的特征区间外没有可靠行为，极端外推场景应在特征侧先做保证覆盖的变换。
- **极小样本 + 尾部精度要求高**：$\tau$ 采样 + 格网点估计都需要样本支撑，样本过少时尾部 CDF 有偏。
- **不需要跨 $\tau$ 一致性的场合**：如果下游只消费单个 $\tau$（如只要中位数），逐 $\tau$ 独立回归参数更少、训练更快。

## 论文实验设置与结果

数据集为挪威南部 20 个站点的五年（2016–2020）小时级地表下行短波辐照度；特征共 246 个 = 12 个气象/卫星特征 × 20 站点 + 6 个时间特征（小时、星期的 sin/cos 周期编码等），来源包括 CAMS-RAD 卫星辐照度估计、NORA3 气象再分析（净下行短波通量、相对湿度、风速风向、降雪、降水、气温、云量、气压）与 CAMS McClear 晴空指数；目标为西海岸 11 号站。评价在 $[0.025, 0.975]$ 上的 11 个等距分位数上完成。

论文 Table 1（5 个随机种子均值，括号内为标准差；CRPS/MAE/RMSE 单位 W/m²，越小越好，SS 越大越好）：

| 模型 | CRPS | MAE | RMSE | ACE | SS |
|------|------|-----|------|-----|-----|
| SP（智能持久性） | — | 52.903 | 96.472 | — | — |
| LSTM-PP | — | 43.152 (0.336) | 74.933 (0.602) | — | 0.397 (0.019) |
| LSTM-QR（逐 $\tau$ 独立） | 30.506 (0.083) | 43.875 (0.226) | 74.949 (0.343) | **0.052 (0.002)** | 0.396 (0.011) |
| SQR: LSTM-Lin | **24.427 (0.247)** | 45.556 (0.360) | 77.165 (0.414) | 0.173 (0.003) | 0.360 (0.014) |
| SQR: LSTM-NN（无约束） | 27.690 (0.099) | 42.612 (0.525) | 72.930 (1.009) | 0.075 (0.003) | **0.429 (0.032)** |
| SQR: LSTM-CLin | 25.920 (0.184) | 44.619 (0.294) | 75.472 (0.397) | 0.190 (0.003) | 0.388 (0.013) |
| SQR: LSTM-SMNN | 29.333 (0.657) | 47.548 (0.974) | 81.731 (1.691) | 0.079 (0.657) | 0.282 (0.059) |
| SQR: LSTM-DLN（本文） | 29.175 (0.216) | 43.266 (0.534) | 74.520 (0.850) | 0.061 (0.003) | 0.403 (0.027) |

结论要分三层读，不能只看一列：

1. **对无约束方法**：MAE 与 RMSE 上 LSTM-DLN 与 LSTM-NN 无显著差异，但覆盖率更好（ACE 0.061 对 0.075），且**没有交叉**，而 LSTM-NN 的交叉率为 3.075 %。
2. **对同类单调方法**：LSTM-DLN 在 MAE、RMSE 上优于可扩展单调神经网络 LSTM-SMNN（43.266 对 47.548；74.520 对 81.731），CRPS 与 ACE 差异不显著；SS 0.403 对 0.282。论文的定位是"在单调网络这一支里具备竞争力"。
3. **对 CRPS 的反直觉结果**：线性变体（Lin / CLin）CRPS 最低但校准最差（ACE 0.173 / 0.190），说明它们用极窄的 CDF 换取了弹球损失指标。这正是论文提醒的评价陷阱。

PICP 在论文中只有可靠性图（Figure 3）没有数值表：曲线从约 0.2 起（对应中位数周围最小的区间），中位数附近的小区间被略微高估，两侧区间漏掉极值，$\tau$ 大致在 0.2–0.5 的分位数被低估；只有 CLin 与 Lin 明显偏离理想对角线。

## 实现要点

### 关键超参数

论文值来自其超参搜索表（Table 2–4），冒烟列为本文代码块的默认值：

| 参数 | 论文取值（搜索范围） | 冒烟默认 | 说明 |
|------|-------------------|---------|------|
| 格网输入维数 $A$ | 2（搜索 2 / 3 / 4） | 2 | 每个格网吃的特征数，参数 $\propto k^{A}$ |
| 格网关键点 $k$ | 21（搜索 2 / 5 / 11 / 21） | 5（特征轴）· 9（$\tau$ 轴） | 决定格网分辨率 |
| 格网个数 $|S|$ | —（论文未给单一最优值） | 4 | 集成规模，输出求和 |
| 校准层关键点 | 61 | 9 | 一维单调校准器 |
| 分位数校准关键点 | 11 | —（并入 $\tau$ 轴） | 论文对 $\tau$ 单设一路校准 |
| 输出校准关键点 | 61 | 9（与上共用） | 输出侧校准 |
| LSTM hidden | 128（搜索 32–512，2 倍步长） | 24 | 论文 2 层、batch 64、Adam |
| LSTM 层数 | 2（搜索 1–4） | 1 | 冒烟用单层 |
| 学习率 / epoch | 1e-3 / 10 epoch（epoch 1,2,3,4 各降 0.5） | 2e-2 / 3 epoch | 对照 NN 为 1e-3 / 30 epoch |
| 回溯窗 $w$ | 96 h（4 天） | 24 h | 自相关定窗口 |
| Horizon $h$ | 36 h | 3 | 直接多步 |
| $\tau$ 采样 | minibatch 内 $\tau \sim U(0,1)$ | 每 batch 8 个 $\tau$ | SQR 目标 |
| 评价分位数 | $[0.025,0.975]$ 上 11 个等距 | 相同 | 另用 101 点网格查交叉 |

### 调优经验

1. **$\tau$ 轴的单调参数化是全部**。用 $\theta[:,j] = \theta[:,j-1] + \mathrm{softplus}(\delta_j)$（等价于 `cumsum(softplus(.))` 再减去首列）保证每行非减。$u$ 轴**不需要**任何约束——单调性只要求沿 $\tau$ 的格点值不减，插值的凸组合性质自动把结论传给输出。约束越多越难训练，这里的约束刚好够用。
2. **$\tau$ 采样要盖满区间**。只训练评估用的 11 个 $\tau$ 会让中间分位数无梯度、逆 CDF 出现平台。论文按 minibatch 采 $\tau \sim U(0,1)$，冒烟实现每 batch 抽 8 个。
3. **校准层的输入范围要覆盖格网和的实际取值**。冒烟实现用 $[-4,4]$ 上的 9 个固定 knot 做分段线性校准并在范围外饱和；范围设窄会让高 $\tau$ 端被压平。检验方法：看 $Q(0.975) - Q(0.025)$ 的展开度分布，本文冒烟实测均值 305.36 W/m²、最小 66.41 W/m²、退化行（展开 < 1e-6）0/801。
4. **用训练集中位数初始化输出偏置**。冒烟配置只有 3 个 epoch，初始化偏置到 $\mathrm{median}(y)$ 后训练 pinball 从 0.369 降到 0.077；不初始化则前 2 个 epoch 都花在把整体水平拉到位上。
5. **未来已知外生特征不能省**。多步长下昼夜相位只能由晴空指数与日历编码提供，否则 LSTM 无法从历史窗口推断 $h$ 步之后的时刻。
6. **成本换保证**。论文的 DLN 是全场最贵的模型：931 375 参数、11 分位数一次前向 2.604 s，约为 SMNN（448 228 参数、0.059 s）的 44 倍、NN（346 058 参数、0.010 s）的 260 倍。要接受这个代价，或者用更小的 $k$、更少的格网降规模（冒烟实现 3 378 参数即可跑通全流程）。
7. **单调性必须实测验证，且要测参数扰动**。训练集上的单调只说明模型学到"没交叉"，不说明结构保证。冒烟脚本做两级检查：101 点 $\tau$ 网格上的差分，以及参数扰动后的重查。扰动口径要如实说明：20 轮里每轮都在**上一轮已扰动的参数**上再加 $N(0, 0.5^2)$ 而不复位，位移因此是累积的，等效标准差 $\sigma_{\mathrm{eq}} = 0.5\sqrt{20} \approx 2.24$——这比 20 次独立同强度扰动更狠，但**不是** 20 次独立 $\sigma = 0.5$ 的扰动。DLN 在这个累积口径下仍是 0 违反；无约束对照在随机初始化时违反 652/801 行、累积扰动下违反 801/801 行。
8. **CRPS 与 ACE 必须一起看**。低 CRPS 可以由过窄的 CDF 刷出来（论文中 Lin / CLin 即是），单调约束保证合法但不保证校准。

### 冒烟实现实测（本条目代码块，CPU，单次 4.3 s）

合成"日周期 + 云衰减 + 趋势 + 噪声"序列 1800 h，窗口 24 h、horizon 3，train/val/test = 1241/266/267，hidden 24，3 epoch：

```
LSTM-DLN : CRPS 16.7313 | MAE 33.1657 | RMSE 52.4742 | ACE 0.1318
           PICP c=0.95:0.881  c=0.76:0.815  c=0.57:0.772  c=0.38:0.577  c=0.19:0.327
LSTM-NN  : CRPS 8.7039  | MAE 24.4934 | RMSE 33.5781 | ACE 0.0779
持续法   : MAE 62.4462  | RMSE 89.1392
交叉检查 : LSTM-DLN 0/801 违反（51/101 点 τ 网格、参数扰动均然）；LSTM-NN 初始化时 652/801，扰动下 801/801
```

冒烟配置下 LSTM-DLN 在精度上劣于无约束 MLP 头（3 epoch 远未收敛，DLN 的分布族由共享格网决定、拟合更慢），这一点与论文一致——论文里 LSTM-NN 的 CRPS 也低于 LSTM-DLN。**这里的价值在结构保证，不在精度**；要精度就按论文配置把 $k$、格网数与 epoch 拉满。

## 代码

```python
"""LSTM-DLN 最小可运行实现 —— 多步长非参数 CDF 预测（冒烟验证脚本）

复现 Erdmann et al. (2026), "Multi-Horizon Time Series Forecasting of
Non-Parametric CDFs with Deep Lattice Networks", AAAI 2026, 40(1), 255-264.
DOI: 10.1609/aaai.v40i1.36986 / arXiv:2511.13756

流程:
    合成太阳辐照度序列 -> LSTM 时序嵌入 -> 分位数水平 tau 分发到格网集成
    -> 单调约束保证的逆 CDF 输出 -> 弹球损失反向传播
    -> 测试段 CRPS / PICP / ACE / MAE / RMSE
    -> 沿 tau 的单调性检查 + 参数扰动检查（分位数交叉计数，LSTM-DLN 恒为 0）

冒烟配置: hidden=24, seq_len=24, epochs=3, CPU, 单次运行 < 90 s
"""
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体（缺字时回退 DejaVu Sans）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# 冒烟配置
HIDDEN = 24
SEQ_LEN = 24
HORIZON = 3
N_LATTICES = 4          # 格网集成中的格网个数 F
N_U_BINS = 5            # 格网沿特征轴的关键点
N_TAU_BINS = 9          # 格网沿分位数轴的关键点（决定 tau 方向分辨率）
EPOCHS = 3
BATCH = 32
LR = 2e-2
N_HOURS = 1800
N_TAU_TRAIN = 8         # 每个 minibatch 采样的 tau 个数（论文 SQR 用 tau~U(0,1)）
TAUS_EVAL = torch.linspace(0.025, 0.975, 11)   # 论文：11 个等距分位数水平


# =========================================================================
# 1. 合成数据：日周期 + 云衰减 + 缓慢趋势 + 噪声（太阳辐照度式）
# =========================================================================
def make_solar_like_series(n_hours=N_HOURS, seed=SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    hour = (t % 24).astype(float)
    clear_sky = 900.0 * np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    ar = np.zeros(n_hours)
    for i in range(1, n_hours):
        ar[i] = 0.85 * ar[i - 1] + 0.15 * rng.normal(0.0, 1.0)
    cloud = np.clip(0.75 + 0.15 * np.sin(2 * np.pi * t / 72.0) + 0.10 * ar, 0.05, 1.0)
    trend = 20.0 * np.sin(2 * np.pi * t / (24.0 * 365.0))
    y = clear_sky * cloud + trend + rng.normal(0.0, 15.0, n_hours)
    return np.clip(y, 0.0, None), clear_sky


def build_windows(y, clear_sky, L, H, mu, sd):
    """切成监督样本: X (N,L,6), E (N,H,3) 未来已知外生, Y (N,H)（目标已标准化）"""
    n = len(y)
    ys, cs = (y - mu) / sd, clear_sky / sd
    hour = (np.arange(n) % 24).astype(float)
    dow = (np.arange(n) // 24 % 7).astype(float)
    feats = np.stack([ys, cs,
                      np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
                      np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0)], axis=1)
    X, E, Y = [], [], []
    for t in range(0, n - L - H + 1):
        idx = np.arange(t + L, t + L + H)
        X.append(feats[t:t + L])
        E.append(np.stack([cs[idx],
                           np.sin(2 * np.pi * (idx % 24) / 24.0),
                           np.cos(2 * np.pi * (idx % 24) / 24.0)], axis=1))
        Y.append(ys[idx])
    to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return to(X), to(E), to(Y)


# =========================================================================
# 2. 格网层：2 输入（特征 u, 分位数水平 tau）1 输出，沿 tau 单调
# =========================================================================
class MonotoneLattice2D(nn.Module):
    """格网函数 f(u, tau) = sum_k theta_k * psi_k(u, tau)

    参数 theta 为 (U, T) 查找表。沿 tau 轴用 cumsum(softplus(.)) 参数化，
    再减去首列，使每一行的 knot 值对 tau 非减；双线性插值对 knot 值取凸组合，
    保持该序关系，故插值结果对 tau 单调不减（无分位数交叉的结构性保证）。
    """

    def __init__(self, n_u_bins=N_U_BINS, n_tau_bins=N_TAU_BINS):
        super().__init__()
        self.U, self.T = n_u_bins, n_tau_bins
        raw = -1.5 + 0.2 * torch.randn(self.U, self.T)
        self.raw = nn.Parameter(raw)
        self.bias = nn.Parameter(torch.zeros(1))

    def param_grid(self):
        g = torch.cumsum(F.softplus(self.raw), dim=1)   # 沿 tau 轴累加增量
        return g - g[:, :1]                             # 首列归零，保持非减

    def forward(self, u, tau):
        """u, tau 形状可广播；返回广播后的形状"""
        g = self.param_grid()
        U, T = self.U, self.T
        ui = u.clamp(0.0, 1.0) * (U - 1)
        ti = tau.clamp(0.0, 1.0) * (T - 1)
        i0 = ui.floor().long().clamp(0, U - 1)
        j0 = ti.floor().long().clamp(0, T - 1)
        i1 = (i0 + 1).clamp(max=U - 1)
        j1 = (j0 + 1).clamp(max=T - 1)
        wu = (ui - i0.float()).clamp(0.0, 1.0)
        wt = (ti - j0.float()).clamp(0.0, 1.0)
        out = ((1 - wu) * (1 - wt) * g[i0, j0] + wu * (1 - wt) * g[i1, j0]
               + (1 - wu) * wt * g[i0, j1] + wu * wt * g[i1, j1])
        return out + self.bias


class MonotoneCalibrator1D(nn.Module):
    """校准层：一维单调分段线性映射（knot 取值由 cumsum(softplus) 保证非减）"""

    def __init__(self, n_knots=9, x_range=4.0, init_bias=0.0):
        super().__init__()
        self.K, self.x_min, self.x_max = n_knots, -x_range, x_range
        self.raw = nn.Parameter(-1.0 + 0.2 * torch.randn(n_knots))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def knot_values(self):
        v = torch.cumsum(F.softplus(self.raw), dim=0)
        return v - v[0]

    def forward(self, z):
        v = self.knot_values()
        K = self.K
        kz = (z - self.x_min) / (self.x_max - self.x_min) * (K - 1)
        kz = kz.clamp(0.0, K - 1.0)
        i0 = kz.floor().long().clamp(0, K - 2)
        w = (kz - i0.float()).clamp(0.0, 1.0)
        return (1 - w) * v[i0] + w * v[i0 + 1] + self.bias


# =========================================================================
# 3. 主模型：LSTM 嵌入 + 分位数分发到格网集成
# =========================================================================
class LSTMDLN(nn.Module):
    """每个 horizon 有 F 个特征，每个特征与 tau 组成一个 2 输入格网的输入对。

    论文的关键适配: "we reserve one input feature in every lattice in the
    ensemble for quantiles" —— 分位数水平进入**每一个**子格网，而不是只进一个。
    """

    def __init__(self, n_features=6, n_exog=3, hidden=HIDDEN, n_horizons=HORIZON,
                 n_lattices=N_LATTICES):
        super().__init__()
        self.n_h, self.n_f = n_horizons, n_lattices
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.feat = nn.Linear(hidden + n_exog, n_lattices)
        self.lattices = nn.ModuleList([MonotoneLattice2D() for _ in range(n_lattices)])
        self.calib = MonotoneCalibrator1D()

    def forward(self, x, exog, tau):
        """x (B,L,Fx), exog (B,H,E), tau (T,) -> (B,H,T) 逆 CDF 值（标准化尺度）"""
        h = self.lstm(x)[1][0][-1]                              # (B,hidden)
        h_rep = h[:, None, :].expand(-1, self.n_h, -1)          # (B,H,hidden)
        z = torch.cat([h_rep, exog], dim=-1)                    # (B,H,hidden+E)
        u = torch.sigmoid(self.feat(z))                         # (B,H,F)
        tau_in = tau.reshape(1, 1, -1)                          # 广播到 (B,H,T)
        out = 0.0
        for f, lat in enumerate(self.lattices):
            out = out + lat(u[:, :, f][:, :, None], tau_in)     # (B,H,T)
        return self.calib(out)


class UnconstrainedQuantileNet(nn.Module):
    """对照：同一 LSTM 嵌入，但 (h, exog, tau) 直接过 MLP，无单调约束"""

    def __init__(self, n_features=6, n_exog=3, hidden=HIDDEN, n_horizons=HORIZON,
                 width=32):
        super().__init__()
        self.n_h = n_horizons
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + n_exog + 1, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1))

    def forward(self, x, exog, tau):
        h = self.lstm(x)[1][0][-1]
        B, H, T = x.shape[0], self.n_h, tau.numel()
        g = torch.cat([h[:, None, :].expand(B, H, -1), exog], dim=-1)   # (B,H,D)
        g = g[:, :, None, :].expand(B, H, T, -1)
        t = tau.reshape(1, 1, T, 1).expand(B, H, T, 1)
        return self.mlp(torch.cat([g, t], dim=-1)).squeeze(-1)         # (B,H,T)


# =========================================================================
# 4. 损失与指标（均在标准化尺度上计算；报告时乘 sd 还原为 W/m^2）
# =========================================================================
def pinball_loss(y, q, tau):
    """rho_tau(e) = max(tau*e, (tau-1)*e)，e = 真值 - 预测（论文式 2）"""
    e = y[..., None] - q
    return torch.maximum(tau * e, (tau - 1.0) * e)


def picp_ace(y, q, taus):
    """PICP_c = 落入 [Q((1-c)/2), Q((1+c)/2)] 的比例；ACE = mean |c - PICP_c|"""
    lv = taus.numpy()
    rows = []
    for i in range(len(lv)):
        if lv[i] >= 0.5 - 1e-9:
            break
        j = len(lv) - 1 - i
        c = float(lv[j] - lv[i])
        p = ((y >= q[..., i]) & (y <= q[..., j])).float().mean().item()
        rows.append((c, p))
    ace = float(np.mean([abs(c - p) for c, p in rows]))
    return rows, ace


def evaluate(model, X, E, Y, taus=TAUS_EVAL):
    model.eval()
    with torch.no_grad():
        q = model(X, E, taus)
    med = q[..., len(taus) // 2]
    mae = (med - Y).abs().mean().item()
    rmse = ((med - Y) ** 2).mean().sqrt().item()
    crps = pinball_loss(Y, q, taus.reshape(1, 1, -1)).mean().item()
    rows, ace = picp_ace(Y, q, taus)
    return dict(crps=crps, mae=mae, rmse=rmse, ace=ace, picp_rows=rows), q


def monotone_violations(model, X, E, n_grid=101, tol=1e-6):
    """检验输出沿 tau 是否单调不减，返回 (总行数, 违反行数, 违反条目数, 最小增量)"""
    model.eval()
    taus = torch.linspace(0.025, 0.975, n_grid)
    with torch.no_grad():
        q = model(X, E, taus)                       # (N,H,T)
    d = q[..., 1:] - q[..., :-1]
    return (q.shape[0] * q.shape[1], int((d < -tol).any(dim=-1).sum().item()),
            int((d < -tol).sum().item()), float(d.min().item()))


def perturbation_test(model, X, E, sigma=0.5, n_trials=20, seed=0):
    """对全部参数加 N(0, sigma^2) 扰动后重查单调性。

    DLN 的单调性由参数化结构保证（对任意参数取值成立），无约束网络的单调性
    只是训练得到的经验性质。返回多轮扰动下的最大违反行数。
    """
    g = torch.Generator().manual_seed(seed)
    orig = copy.deepcopy(model.state_dict())
    worst_rows, worst_tau = 0, None
    for _ in range(n_trials):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator=g) * sigma)
        for grid in (51, 101):
            n_rows, bad_rows, _, _ = monotone_violations(model, X, E, n_grid=grid)
            if bad_rows > worst_rows:
                worst_rows, worst_tau = bad_rows, grid
    model.load_state_dict(orig)
    return worst_rows, worst_tau


# =========================================================================
# 5. 训练
# =========================================================================
def train(model, Xt, Et, Yt, Xv, Ev, Yv, epochs=EPOCHS, lr=LR, batch=BATCH,
          n_tau=N_TAU_TRAIN, tag='model'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xt.shape[0]
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            tau = torch.rand(n_tau).sort().values          # SQR: tau ~ U(0,1)
            q = model(Xt[idx], Et[idx], tau)
            loss = pinball_loss(Yt[idx], q, tau.reshape(1, 1, -1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            qv = model(Xv, Ev, TAUS_EVAL)
            vl = pinball_loss(Yv, qv, TAUS_EVAL.reshape(1, 1, -1)).mean().item()
        hist.append((tot / nb, vl))
        print(f'  [{tag}] epoch {ep + 1}/{epochs}  train pinball={tot / nb:.5f}  '
              f'val pinball={vl:.5f}')
    return hist


# =========================================================================
# 6. 主流程
# =========================================================================
if __name__ == '__main__':
    t0 = time.time()
    print('=' * 74)
    print('LSTM-DLN 冒烟验证 — 多步长非参数 CDF 预测')
    print('=' * 74)

    y, cs = make_solar_like_series()
    n_series = len(y)
    i_tr, i_va = int(0.70 * n_series), int(0.85 * n_series)
    mu = float(y[:i_tr].mean())          # 标准化常数只用训练段统计
    sd = float(y[:i_tr].std())
    X, E, Y = build_windows(y, cs, SEQ_LEN, HORIZON, mu, sd)
    n = X.shape[0]
    n_tr, n_va = int(0.70 * n), int(0.85 * n)
    Xt, Et, Yt = X[:n_tr], E[:n_tr], Y[:n_tr]
    Xv, Ev, Yv = X[n_tr:n_va], E[n_tr:n_va], Y[n_tr:n_va]
    Xs, Es, Ys = X[n_va:], E[n_va:], Y[n_va:]
    print(f'序列 {N_HOURS} h | 窗口 {SEQ_LEN} h | horizon {HORIZON} | '
          f'样本 train/val/test = {n_tr}/{n_va - n_tr}/{n - n_va}')
    print(f'辐照度 mean={mu:.1f}  sd={sd:.1f} W/m^2（训练段统计，用于标准化）')

    model = LSTMDLN()
    model.calib.bias.data.fill_(float(np.median(Yt.numpy())))
    base = UnconstrainedQuantileNet()
    n_p = lambda m: sum(p.numel() for p in m.parameters())
    print(f'LSTM-DLN 参数量 {n_p(model):,} | 无约束对照参数量 {n_p(base):,}\n')

    print('[0] 未训练时的单调性（无约束网络在随机初始化下不保证单调）')
    for name, m in (('LSTM-DLN', model), ('LSTM-NN (无约束)', base)):
        n_rows, bad_rows, bad_ent, _ = monotone_violations(m, Xs, Es)
        print(f'  {name:16s}: 违反行数 {bad_rows}/{n_rows}，违反条目数 {bad_ent}')

    print('\n[1] 训练 LSTM-DLN')
    train(model, Xt, Et, Yt, Xv, Ev, Yv, tag='DLN')
    print('[2] 训练无约束对照 (LSTM + MLP 分位数头)')
    train(base, Xt, Et, Yt, Xv, Ev, Yv, tag='NN ')

    print('\n' + '=' * 74)
    print('测试段评估（弹球损失/MAE/RMSE 由标准化尺度乘 sd 还原为 W/m^2）')
    print('=' * 74)
    results = {}
    y_true = Ys.numpy() * sd + mu
    persistence = Xs[:, -1, 0].numpy() * sd + mu      # 持续法：用窗口最后一个观测
    print(f'\n持续法基线 MAE  = {np.abs(persistence - y_true[:, 0]).mean():.4f}  '
          f'RMSE = {np.sqrt(((persistence - y_true[:, 0]) ** 2).mean()):.4f}')
    for name, m in (('LSTM-DLN', model), ('LSTM-NN (无约束)', base)):
        met, q = evaluate(m, Xs, Es, Ys)
        results[name] = (met, q)
        print(f'\n--- {name} ---')
        print(f'  CRPS  = {met["crps"] * sd:.4f}')
        print(f'  MAE   = {met["mae"] * sd:.4f}')
        print(f'  RMSE  = {met["rmse"] * sd:.4f}')
        print(f'  ACE   = {met["ace"]:.4f}')
        print('  PICP:  ' + '  '.join(f'c={c:.2f}:{p:.3f}' for c, p in met['picp_rows']))

    print('\n' + '=' * 74)
    print('分位数交叉检查：在 [0.025, 0.975] 上取 101 个 tau，检验沿 tau 单调不减')
    print('=' * 74)
    for name, m in (('LSTM-DLN', model), ('LSTM-NN (无约束)', base)):
        for grid in (51, 101):
            n_rows, bad_rows, bad_ent, min_step = monotone_violations(m, Xs, Es, n_grid=grid)
            print(f'  {name:16s} tau 网格 {grid:3d}: 违反行数 {bad_rows}/{n_rows} '
                  f'({100.0 * bad_rows / n_rows:.3f}%)  违反条目数 {bad_ent}  '
                  f'最小增量 {min_step:.3e}')

    print('\n逆 CDF 展开度检查（Q(0.975) - Q(0.025)，判断有无退化为点质量的样本）:')
    for name, m in (('LSTM-DLN', model), ('LSTM-NN (无约束)', base)):
        with torch.no_grad():
            q = m(Xs, Es, TAUS_EVAL)
        sp = (q[..., -1] - q[..., 0]) * sd
        print(f'  {name:16s}: mean={sp.mean():.2f} W/m^2  min={sp.min():.2f} W/m^2  '
              f'退化行数(展开<1e-6)={int((sp < 1e-6).sum())}/{sp.numel()}')

    print('\n参数扰动鲁棒性（对全部参数加 N(0, 0.5^2) 噪声 20 轮后重查单调性）:')
    for name, m in (('LSTM-DLN', model), ('LSTM-NN (无约束)', base)):
        worst_rows, grid = perturbation_test(m, Xs, Es, sigma=0.5, n_trials=20)
        print(f'  {name:16s}: 最大违反行数 {worst_rows}'
              + (f'（tau 网格 {grid}）' if grid else ''))

    with torch.no_grad():
        qdln = model(Xs[:1], Es[:1], TAUS_EVAL)[0, 0].numpy() * sd + mu
        qnn = base(Xs[:1], Es[:1], TAUS_EVAL)[0, 0].numpy() * sd + mu
    print('\n单样本逆 CDF 抽查（第 0 个测试窗, horizon 1, 真值 %.1f W/m^2）:' % y_true[0, 0])
    print('    tau      :' + ''.join(f'{t:>9.3f}' for t in TAUS_EVAL.numpy()))
    print('    LSTM-DLN :' + ''.join(f'{v:>9.1f}' for v in qdln))
    print('    LSTM-NN  :' + ''.join(f'{v:>9.1f}' for v in qnn))

    met_dln, q_dln = results['LSTM-DLN']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    seg = slice(0, 120)
    ax = axes[0]
    ax.plot(y_true[seg, 0], 'k-', lw=1.2, label='观测 (horizon 1)')
    ax.fill_between(np.arange(120), q_dln[seg, 0, 0].numpy() * sd + mu,
                    q_dln[seg, 0, -1].numpy() * sd + mu, alpha=0.25,
                    color='tab:blue', label='95% 分位区间')
    ax.plot(q_dln[seg, 0, 5].numpy() * sd + mu, color='tab:blue', lw=1.0, label='中位数预测')
    ax.set_xlabel('测试窗序号'); ax.set_ylabel('辐照度 (W/m²)')
    ax.set_title('LSTM-DLN 多步长概率预测（horizon 1）')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(TAUS_EVAL.numpy(), qdln, 'o-', label='LSTM-DLN（结构单调）')
    ax.plot(TAUS_EVAL.numpy(), qnn, 's--', label='LSTM-NN（无约束）')
    ax.set_xlabel('分位数水平 $\\tau$'); ax.set_ylabel('预测值 (W/m²)')
    ax.set_title('逆 CDF 抽查：格网单调约束 vs 无约束')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_dln_demo.png')
    fig.savefig(out_png, dpi=130)
    print(f'\n图已保存: {out_png}')
    print(f'总耗时 {time.time() - t0:.1f} s')
```

## 参考文献

ERDMANN N, BENTSEN L Ø, STENBRO R, et al. Multi-horizon time series forecasting of non-parametric CDFs with deep lattice networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence: Vol. 40, No. 1. 2026: 255-264. DOI: 10.1609/aaai.v40i1.36986.

GUPTA M R, COTTER A, PFEIFER J, et al. Monotonic calibrated interpolated look-up tables[J]. Journal of Machine Learning Research, 2016, 17(109): 1-47.

YOU S, DING D, CANINI K, et al. Deep lattice networks and partial monotonic functions[C]//Advances in Neural Information Processing Systems 30 (NIPS 2017). 2017: 2981-2989.

SILL J. Monotonic networks[C]//Advances in Neural Information Processing Systems 10 (NIPS 1997). 1998: 661-667.

KOENKER R, BASSETT G. Regression quantiles[J]. Econometrica, 1978, 46(1): 33-50. DOI: 10.2307/1913643.

CHERNOZHUKOV V, FERNÁNDEZ-VAL I, GALICHON A. Quantile and probability curves without crossing[J]. Econometrica, 2010, 78(3): 1093-1125. DOI: 10.3982/ECTA7880.

GNEITING T, RAFTERY A E. Strictly proper scoring rules, prediction, and estimation[J]. Journal of the American Statistical Association, 2007, 102(477): 359-378. DOI: 10.1198/016214506000001437.
