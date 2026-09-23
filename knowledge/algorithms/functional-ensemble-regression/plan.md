# 函数型数据集成回归 — 自适应门控残差校正研究（AHFE）

## Context

用户调用**算法研究系统**研究"函数型数据的集成机器学习模型"。研究配方（用户提出）：

> **在 2025-2026 前沿论文的模型基础上做创新改进，在几个特殊应用场景测试，效果提升即视为进步。**

经**三轮审阅**（执行风险审阅 → 科研课题审核 → 方法论文审稿）后重构。第三轮审阅发现关键审稿风险：**函数型门控并非空白领域**——已有正式发表的 **Functional Mixtures-of-Experts（FME）**（函数型预测变量 + scalar response + functional gating network + experts）及其可解释稀疏版本 iFME。故本轮将创新定位从"给函数型回归加门控"修正为**"自适应残差校正门控"**（adaptive gated residual correction），与 FME 明确区分。最终方案：

- **基模型**：Robust SIMPLS ensemble（参考 Alin et al. 2026, *Chemometrics & Intelligent Laboratory Systems* 269, DOI 10.1016/j.chemolab.2025.105626）——分块 B 样条平滑 + 稳健加权 SIMPLS + 潜空间惩罚回归 + 自助聚合，scalar-on-function 回归。**定位：受其启发的对照基座，非逐行复现依赖。**
- **核心方法**：**AHFE — 残差校正混合函数型集成（Adaptive Hybrid Functional Ensemble）**。方法 = **残差 GBDT 校正 + 稳健 SIMPLS 线性基座**的混合 ŷ = ŷ_A + w(x)·r̂_B，w(x)∈[0,1] 由 sigmoid-link 门控确定。**核心贡献（P4a 后诚实降格）**：非线性 SoFR 上 AHFE 相对 SIMPLS 提升 ΔR²≈+0.34（稳健巨大），线性数据门控自动关闭（w→0）不致损；**门控作为可调修正强度的灵活装置（鲁棒/自动化），不宣称逐样本优于常数 γ**——P4a 实证：adaptive ≈ fixed（含可行 oracle）。oracle gate 理论仍保留为方法学锚。区别于 FME（非多专家概率混合，而是残差校正强度门控）。
- **交付物**：知识库条目（对齐 fpca.md 格式，含理论分析节）+ 基准对比。
- **依赖**：安装 scikit-fda（tecator）+ 条目保留 from-scratch 自包含代码。

环境（已验证）：python 3.11.5 @ `D:\py\Python3\python.exe`；numpy/scipy/sklearn 1.9/xgboost 3.0/lightgbm 4.6 已装；scikit-fda 未装（批准安装）。

现有资产：`fpca.md` 的 `FunctionalData/FPCA/FLMRegressor` 基线代码；`agents/algorithm/` 6 agent；`random-forest.md`/`xgboost.md` 集成理论。

---

## 1. 基模型规格：Robust SIMPLS ensemble（Alin 2026 启发的对照基座）

**定位**：本研究的**对照基座 + 线性组件 A**。方法结构参照标准 SIMPLS/RWSIMPLS 文献与 Alin 2026 方法摘要；**不声称逐行复现 Alin 2026**，贡献声明不依赖其精确实现（审稿意见 2）。P1 直接读取 **SSRN 27 页全文预印本**逐项核对（§7.1，无需付费）。

1. **（惩罚）B 样条平滑**：函数预测子在网格上离散观测 → B-spline 设计矩阵平滑。**单函数预测子设定**（Tecator/S5 均仅一条 X(t)）下退化为**单个 B 样条块**，采用惩罚 B-spline 平滑；**框架自然推广至多函数预测子**的块对角表示（每个函数变量一个基组，块间零矩阵）——不将"分块"作为实验卖点硬保留（审稿意见 12）。P1 核实时确认 Alin 2026 分块是否另指波长区间分块，结论入 `refs_notes.md`。
2. **稳健加权 SIMPLS（RWSIMPLS）**：残差/稳健尺度（MAD）加权，抑制异常值污染。
3. **潜空间惩罚回归**：潜变量空间惩罚回归，二次型为系数函数块粗糙度惩罚 → 光滑可解释 β(t)。
4. **自助集成**：B 个 bootstrap 样本各拟合 RWSIMPLS → mean/median 聚合；OOB + .632+ 诊断。
5. **调参**：成分数 A 与惩罚 λ 经 k 折 CV。

**复现要点**：SIMPLS 标准实现（~80 行）；B-spline 基用 scipy；稳健权重 IRLS/M-估计或 MAD；mean/median 两版聚合。

---

## 2. 核心创新：AHFE — 自适应门控残差校正混合函数型集成

### 2.1 问题形式化（含 FME 先验区分）

**与现有工作的区分**：已有 **Functional Mixtures-of-Experts（FME）** 及其稀疏版本 iFME 研究函数型预测子 + scalar response + functional gating network + 多专家概率混合：ŷ(x) = Σ_k π_k(x)ŷ_k(x)。**本文不与其竞争"函数型门控"本身**。AHFE 的不同在于：**只做一个线性稳健基座 + 一个残差校正器的残差加性混合，门控控制"非线性修正强度"而非"专家权重"**——w(x) 决定"在多大程度上用 GBDT 校正 SIMPLS 的系统残差"，语义是残差校正，不是专家选择。这是与 FME 的边界（详见 §2.4 贡献表述）。

**问题设定**：PLS 潜变量是 X 的线性投影，当 Y 对 X(t) 呈**非线性功能效应**（如 Y = α + ∫Xβ dt + λ(q²−E(q²)) + ε 的交互项，q=∫Xφdt；或阈值/分段结构）时，线性潜空间集成系统性欠拟合。**单一常数加权（固定 γ 的残差提升）无法反映"非线性强度随样本变化"**——不同样本处于不同的线性/非线性区域。

### 2.2 方法设计（两组件 + 残差校正门控）

统一符号（审稿意见 2：修正此前 ŷ_B 混用两个含义的数学错误）：

1. **线性组件 A**：Robust SIMPLS ensemble（§1）→ ŷ_A。提供光滑可解释 β(t)、稳健性；低方差、高偏差（在非线性区域）。
2. **非线性组件 B = 残差预测器 r̂_B**：在残差 r = y − ŷ_A 上训练标准 GBDT（XGBoost/LightGBM/CatBoost 三选比较，用户指定考虑），记为 r̂_B(x) = GBDT(z)。特征 z 见下方"门控特征"。低偏差、高方差，捕获潜空间遗漏的非线性功能结构。
3. **残差校正门控融合（核心贡献）**：
   $$
   \hat y_{AHFE}(x) = \hat y_A(x) + w(x)\,\hat r_B(x), \qquad w(x) = \sigma\big(\alpha + z^\top\theta\big) \in [0,1]
   $$
   - **门控特征 z**（预测时可计算，审稿意见 4）：`[FPCA scores, |r̂_B(x)|, Q-residual, leverage]`。**不含真实残差幅值 |y − ŷ_A|**——预测新样本时 y 未观测，真实残差不可得；第一版不上 MLP，用 **sigmoid-link 门控回归**（无 0/1 标签、非二元交叉熵，故不称"逻辑回归"，准确术语为 sigmoid-parametrized gating regression / logistic-link gating function）。
   - **门控训练（cross-fitting，审稿意见 5）**：外层 CV 训练集中做内层 K 折 cross-fitting，每内折生成真正的 OOF：ŷ_A^OOF、r̂_B^OOF、z^OOF（特征折内拟合、折外投影）→ 在 OOF 数据上最小化 Σ_i(y_i − ŷ_{A,i}^OOF − w(z_i)r̂_{B,i}^OOF)² + λ_g‖θ‖² 训练门控 → A、B 用整个外层训练集重训 → 一次性预测外层测试。**门控全程不接触外层测试的 y**。
   - **语义**：w(x)→0 处信任 SIMPLS（线性区域，保持可解释）；w(x)→1 处启用 GBDT 校正（非线性区域）。逐样本自适应 = 从"常数 γ 残差提升"升级为"位置依赖的自适应残差校正"。
   - **与 MoE 形式的严格等价关系**（审稿意见 2 修正）：定义完整非线性专家 ŷ_B^full(x) = ŷ_A(x) + r̂_B(x)，则：
     $$
     \hat y_{AHFE} = \hat y_A + w\,\hat r_B = (1-w)\hat y_A + w\,\hat y_B^{full}
     $$
     这才与经典 MoE 混合式严格同构——**且此写法正暴露与 FME 的本质区别**：这里只有一个非线性"专家"，w 控制的是残差校正强度，不是多专家之间的概率分配。
   - **消融对照（诚实报告，不预设立场）**：AHFE-fixed（w ≡ γ，γ 由 CV 调）vs AHFE-adaptive（w(x) 学习）。**P4a 已实证：adaptive ≈ fixed（含可行 oracle）** → 消融如实报告该结果，门控的价值定位为"鲁棒/自动化装置 + oracle 理论锚 + 防劣化机制"，而非"逐样本优于常数 γ"。若确证基准（P5）在某一场景复现 adaptive > fixed 的稳健差异，则如实作为发现报告并探讨条件。

### 2.3 理论分析（审稿意见 3：升级为可证 oracle 门控）

记 e_A = Y − ŷ_A（A 的真实残差）、q = r̂_B(X)（残差模型输出）。AHFE 条件风险：

$$
R(w|z) = \mathbb{E}\big[(e_A - w q)^2 \,\big|\, z\big]
$$

对 w 求最优解，得 **oracle gate**：

$$
\boxed{\,w^*(z) = \Pi_{[0,1]}\frac{\mathbb{E}[e_A q|z]}{\mathbb{E}[q^2|z]}\,}
$$

其中 Π_[0,1] 为截断到 [0,1]。**AHFE-fixed 是其特例**（w(z) ≡ γ）：

$$
\gamma^* = \Pi_{[0,1]}\frac{\mathbb{E}[e_A q]}{\mathbb{E}[q^2]}
$$

**方法学链条**：SIMPLS → 固定残差校正（AHFE-fixed）→ 条件最优残差校正（AHFE-adaptive）。理论语义：

- 若残差模型在某类样本上确实解释了 SIMPLS 的系统残差（E[e_A q|z] 大），门控自动抬高；
- 若树组件只是在拟合噪声（E[e_A q|z] ≈ 0），w*(z) ≈ 0，AHFE 退化为 SIMPLS——**门控自带防劣化机制**；
- 系统性收益来自"以样本位置 z 为条件的偏差-方差权衡"，而非简单平均。

### 2.4 贡献表述（克制，与 FME 划界）

> "Inspired by the recent robust SIMPLS ensemble framework (Alin et al., 2026), we propose an adaptive hybrid functional ensemble (AHFE) in which a linear robust-PLS component and a residual gradient-boosting corrector are combined by a residual-correction gate over functional features. Unlike functional mixtures-of-experts, which mix multiple full experts via functional gating, AHFE's gate modulates the *strength of nonlinear residual correction* applied to an interpretable linear base, with an analytic oracle gate characterizing the optimal correction. On real and synthetic nonlinear scalar-on-function tasks, AHFE delivers large and robust gains over the base SIMPLS ensemble (ΔR² ≈ +0.34); the correction gate is a flexible, self-shrinking device that degrades gracefully on linear data, which we analyze honestly rather than claiming per-sample superiority over a fixed correction."

**对照（如实分层）**：
- **主贡献声明（headline）**：AHFE vs M3（SIMPLS ensemble 基座）——ΔR²≈+0.34 稳健巨大；AHFE vs M1/M2（景观，M2 与 AHFE-B 同特征集保证公平）——结构增益归因。
- **消融（如实发现，非预设立场）**：AHFE-adaptive vs AHFE-fixed——P4a 实证 adaptive ≈ fixed（含可行 oracle），报告为发现；门控定位为鲁棒/自动化/理论锚装置，不宣称逐样本优于常数 γ。
- **理论锚**：oracle gate 命题保留为方法学链条（SIMPLS → 固定残差校正 → 条件最优残差校正）。

真实提升（AHFE>>SIMPLS）= 贡献；自适应 vs 固定 = 诚实报告的分量。

---

## 3. 基准协议

### 核心实验（3 个，全量）
| 场景 | 数据 | 任务 | 目的 |
|------|------|------|------|
| S1 | **Tecator**（`fetch_tecator()`，215×100 NIR，单函数预测子） | 脂肪含量 SoFR | 真实数据性能 |
| S5 | **合成非线性-中心化二次项**（DGP 见 §3.1） | Y=∫Xβ dt + λ(q²−Eq²) + ε，q=∫Xφdt | 非线性优势 + 门控机制 + β̂ 恢复 |
| S6 | **污染 Tecator**（S6a/S6b，定义见 §3.1） | 同 S1 | 稳健性未被破坏 |

### 补充实验（`--quick`，可选）
- S2/S3：Tecator 水分 / 蛋白质；S4：Gasoline octane（辛烷值）。

### 方法集（4 方法 + 1 消融，克制叙事）
- **M1 FLM**（函数线性模型基线）
- **M2 Functional Features + GBDT**（非线性树基线；**与 AHFE-B 完全相同的特征库**，证明 AHFE 赢在结构而非特征工程，审稿意见 8）
- **M3 SIMPLS ensemble**（§1 基座）
- **AHFE-fixed**（消融：常数 γ 残差校正）← 对照自适应；P4a 实证 ≈ adaptive（如实报告）
- **AHFE-adaptive**（提出方法，§2）← **主贡献为 AHFE 混合整体（AHFE-fixed 亦属 AHFE 族），headline 是与 M3 的 ΔR²≈+0.34**

### 评价指标（审稿意见 7、9：改统计口径）
- 预测：**ΔRMSE = RMSE_baseline − RMSE_AHFE**（主报告）+ 95% 置信区间 + **OOF 逐样本配对损失差的 paired bootstrap**；若需正式模型比较用 repeated-CV corrected test——**不以 25 个 fold 分数当 25 个独立样本做 Wilcoxon**（训练样本重叠导致伪重复）。Holm 校正保留。
- **函数参数恢复（合成 S5 必报）**：‖β̂(t) − β(t)‖（L2 恢复误差）、∫|β̂−β| 面积。
- **门控机制验证（审稿意见 7，如实报告）**：定义逐样本真实非线性强度 NL_i = |λ(q_i² − E(q²))|，报告 **Spearman(w_i, NL_i)** 与 **NL 分位数 → 平均门控权重**图。**P4a 实证 Spearman 弱/多数不显著、adaptive≈fixed** → 如实报告该发现（含可行 oracle 上界），作为对"门控可习得非线性位置"这一直觉的诚实否定，而非作为增益证据。
- 稳健性诊断（S6）：**ΔRMSE = RMSE_contaminated − RMSE_clean**（而非仅污染后 RMSE）+ **异常样本 w_outlier 是否异常增大**（若离群样本门控大量开启 GBDT → 鲁棒性下降，是极有价值的诊断，审稿意见 11）。
- 运行时。

### 图表（→ outputs/figures/）
fig1 数据总览（tecator 光谱 + 合成曲线）；fig2 β̂ 恢复（真值 vs SIMPLS 估计）；fig3 核心场景 R²/ΔRMSE 分组柱状图（AHFE-adaptive / AHFE-fixed / M3 / M2 / M1，误差线=种子 std）；fig4 污染稳健性对比（含 ΔRMSE 与 w_outlier 诊断）；**fig5 NL 分位数 → 平均门控权重单调图（核心结果图）+ Spearman**；fig6 运行时。

---

### 3.1 关键协议锁定（P3 设计必含，4 项硬锁定后方可进 P4a）

1. **S5 数据生成过程（DGP，审稿意见 6：识别性修正）**：
   - $X(t) \sim \mathcal{GP}(0, C)$：**严格零均值**光滑高斯过程（Matern 协方差），网格 100 点，曲线有结构非白噪声
   - $q_i = \int X_i(t)\varphi(t)\,dt$，$Y_i = \alpha + \int X_i(t)\beta(t)\,dt + \lambda\big[q_i^2 - \mathbb{E}(q^2)\big] + \varepsilon_i$
   - **中心化二次项**：对中心高斯过程，线性项与中心化二次项具有良好正交性——β(t) 才真正有资格作为线性组件 ground truth（否则"β 恢复差"可能是 DGP 自身的投影偏差而非 SIMPLS 估计失败）
   - 正交约束用**协方差算子**：⟨β, Cφ⟩ = 0（而非仅 ∫βφ dt = 0）；β(t)=sin(2πt)、φ(t)=cos(2πt)，λ 控制非线性强度，SNR∈[2,5]
   - **S5b 线性阳性对照**：Y = α + ∫Xβ dt + ε，验证 SIMPLS 该设定下表现优异、AHFE 增益仅在非线性版本出现
   - 附方差/SNR 合理性检查
2. **门控训练 = cross-fitting（审稿意见 5）**：外层 CV → 内层 K 折 cross-fitting 生成真 OOF（ŷ_A^OOF、r̂_B^OOF、z^OOF）→ OOF 上训练门控（sigmoid-link 回归 + λ_g 正则）→ 外层训练集整集重训 A、B → 预测外层测试；**门控不接触外层测试 y**；z 特征折内拟合、折外投影；约束 w∈[0,1]；报告 w 校准（线性区均值趋近 0）。
3. **M2 与 AHFE-B 同特征库（审稿意见 8）**：M2 的 GBDT 与 AHFE-B 使用完全相同特征集（FPCA scores + B-spline 系数 + 一阶导数），确保 AHFE-adaptive > M2 归因于结构而非特征工程；M2 自身嵌套 CV 防泄漏（外层 5 折 → 内层 4 折 OOF，特征折内拟合）。
4. **CatBoost 回退**：P0 先装 catboost，失败立即跳过并在报告声明"因 Windows 环境兼容性未纳入，仅报告 XGBoost/LightGBM"。
5. **S6 污染定义（审稿意见 11）**：S6a = Y-outlier（10% 样本响应 +5σ）；S6b = X-函数型离群（整条光谱 level shift 或局部 spike）。报告 ΔRMSE（vs 清洁）+ w_outlier 诊断。
6. **运行模式**：核心 3 场景全量（5 种子 × 5 折）；补充场景 `--quick`（2 种子 × 3 折）；脚本内置 `--quick` + tqdm 进度。

---

## 4. 知识库条目结构（交付物）

`# 函数型数据集成回归：自适应门控残差校正（AHFE）`（`knowledge/algorithms/functional-ensemble-regression/README.md`），镜像 fpca.md 格式：
> **路径更正（2026-09-23）**：本条原写作 `knowledge/algorithm-repository/...`——该条目属自创方法，
> 已迁入自创算法库 `algorithms/`。

- **数学设定**：SoFR 模型；SIMPLS 潜变量空间；B-spline 平滑（单函数预测子 → 惩罚 B-spline；多函数预测子 → 块对角表示）；RWSIMPLS 稳健加权；bootstrap 集成；**AHFE 公式**（ŷ = ŷ_A + w·r̂_B、r̂_B=GBDT(z)、sigmoid-link 门控 w=σ(α+zᵀθ)、门控特征 z=[FPCA scores, |r̂_B|, Q-residual, leverage]）
- **理论分析节**：**oracle gate 命题**（w*(z) = Π_[0,1] E[e_A q|z]/E[q²|z]、AHFE-fixed 特例 γ*）、残差校正 vs MoE 严格等价（ŷ=(1−w)ŷ_A + w·ŷ_B^full）、方法学链条（SIMPLS → 固定残差校正 → 条件最优残差校正）、门控防劣化论证
- **关键假设**：PLS 线性潜空间假设及失效条件；门控识别性（cross-fitting，w 在 OOF 上训练、不接触外层测试 y）；稳健权重假设；与 FME/iFME 的边界声明
- **适用场景**：线性功能效应→SIMPLS 足够；非线性/交互功能效应→AHFE；异常值污染→稳健 SIMPLS/AHFE；需 β(t) 解释→线性组件；外推→树组件失效需谨慎
- **实现要点**：A 与 λ 调参、cross-fitting 门控训练流程、w∈[0,1] 约束、γ 消融设定、种子纪律、确定性模式可复现容差（§8）
- **完整 Python 代码（两小节明确分隔）**：① 自包含块（仅 numpy/scipy/sklearn/xgboost/lgbm（可选 catboost）；SIMPLS + RWSIMPLS + bootstrap 集成 + 残差 GBDT + **cross-fitting 门控融合** + 景观 + 基准，可独立运行）＝**复现实现**；② 生产用法小节（scikit-fda 加载数据、FPCA、对接）＝**生产推荐**
- **基准结果摘要**：嵌入 fig1–6 + 核心场景对比表（含 ΔRMSE+95%CI、β̂ 恢复误差、Spearman(w,NL) 列）
- **参考文献**：≥7（含 Alin 2026 SSRN + FME/iFME 竞争方法 + SIMPLS 原始文献 + oracle 门控/MoE 理论锚）

可选归档：`knowledge/algorithms/ahfe-functional-ensemble/`（README/algorithm.py/test/demo/benchmark.md/figures）。

---

## 5. 文件布局与执行编排（开发基准 / 确证基准分离，审稿意见 10）

```
outputs/functional_ensemble_regression/
├── src/  data_loader.py · simpls.py · robust_simpls.py · functional_features.py
│         methods.py(M1/M2/M3) · ahfe.py · gating.py · benchmark.py · visualize.py
├── tests/  test_functional_features.py · test_simpls.py · test_ahfe.py
├── outputs/tables/ · outputs/figures/
├── benchmark_report.md · validation.log · refs_notes.md
```

执行映射（算法研究系统，**开发基准 → 方案冻结 → 确证基准**）：

| 阶段 | 执行者 | 交付 | 门禁 |
|------|--------|------|------|
| P0 | 环境 | `pip install scikit-fda catboost`（catboost 先装，失败跳过声明回退）；验证 import + fetch_tecator | 冒烟测试；catboost 可用性判定 |
| P1 | 文献/WebFetch（formalizer 前置） | **读 SSRN 27 页全文**核对 Alin 2026（分块/稳健加权/潜空间惩罚细节）→ `refs_notes.md`；**FME/iFME 作为竞争方法文献精读并纳入对照叙事** | 基座规格锁定；FME 划界叙事锁定 |
| P2 | formalizer-agent | 形式化 AHFE 问题（线性潜空间失效 + 残差校正门控 + oracle gate 命题） | 用户确认问题定义 |
| P3 | designer-agent | AHFE 设计（组件/门控/cross-fitting/理论）+ 伪代码 + **§3.1 关键协议**（4 项硬锁定：创新定位、oracle gate、cross-fitting+删真实残差特征、S5 中心化二次项） | 用户确认设计 |
| **P4a** | coder-agent（**开发基准**） | **先只实现**：SIMPLS 基座 + AHFE-adaptive（cross-fitting 门控）+ Tecator + 合成交互 DGP；**开发种子下可自由迭代设计/debug** | **机制门禁**（非测试性能门禁）：门控 Spearman(w,NL)>0；S5b 线性区 w 均值趋 0；β̂ 恢复误差在阈值内；管线无泄漏 |
| 方案冻结 | — | DGP、超参数空间、特征库、门控结构全部 freeze → 记入 `checkpoints/冻结规格.md` | 冻结清单 |
| **P4b** | coder-agent（全量） | 补 M1/M2/M3 景观 + AHFE-fixed 消融 + SHAP + 补充数据 + **单元测试全套** | 单元测试全 PASS、核心覆盖率≥80% |
| **P5** | benchmark-agent（**确证基准**） | **用全新未触碰种子 / holdout 复现最终结论**：核心 3 场景全量 + 补充 `--quick` → 表 + 图 + benchmark_report.md（含 ΔRMSE+95%CI、paired bootstrap、Spearman(w,NL)） | 数值合理、无 NaN、结论如实报告（**失败即如实失败**，作为科学发现而非代码 bug） |
| P6 | validator-agent | 三层验证清单（§8）+ 复跑可复现 | 全部 PASS |
| P7 | 归档 | 写知识库条目 + `_index.md` 登记（70→71）+ 可选算法包 | 格式 diff 干净 |

> **防 adaptive overfitting（审稿意见 10）**：P4a 开发基准可自由迭代；一旦方案冻结（P4b 末），**不得**再看开发性能改模型；P5 确证基准用全新种子/holdout 独立复现，失败如实报告。

---

## 6. 文献清单（P1 核实带⚠）

**竞争方法文献（P1 单独精读，直接纳入对照与划界叙事）**：
1. **Functional Mixtures-of-Experts（FME）** ⚠（*Statistics and Computing*, 2023，DOI 10.1007/s11222-023-10379-0）+ **iFME**（可解释稀疏版）——**必须作为直接竞争方法写进引言/related work**，否则审稿一票 "functional gating models already exist"（Springer）

**基座与理论锚**：
2. **Alin et al. (2026)** Ensemble robust SIMPLS with block-penalized smoothing for scalar-on-function regression. *Chemometrics and Intelligent Laboratory Systems*, 269. DOI 10.1016/j.chemolab.2025.105626 ⚠ **SSRN 27 页预印本可全文免费读取**（abstract_id 5593333）→ P1 优先读 SSRN 全文逐项核对
3. de Jong (1993) SIMPLS ⚠页码；4. Hubert & Vanden Branden (2003) RSIMPLS；5. Eilers & Marx (1996) P-splines；6. Wang, Chiou & Müller (2016) FDA 综述；7. Reiss & Ogden (2007) FPC 回归/PLS；8. Breiman (2001) RF；9. Ke et al. (2017) LightGBM；10. Chen & Guestrin (2016) XGBoost；11. Wolpert (1992)/Breiman (1996) stacking；12. Möller et al. (2016) 函数型 RF（对照）；13. **Jacobs et al. (1991) Mixture-of-Experts 门控（oracle gate 理论锚）**；14. **Friedman (2001) 梯度提升（GBM 理论）**；15. Ramsay & Silverman (2005)

---

## 7. 环境/风险/回退

- scikit-fda 0.10.1 支持 NumPy 2（已核）；核心代码**自包含**，Tecator 回退 StatLib CSV，不阻塞交付
- **Alin 2026 已可免费获取全文**（SSRN 预印本），不再依赖付费墙/摘要推断（审稿意见 13）
- Gasoline octane：fda.usc 公开 CSV / skfda 内置
- 运行时：核心 3 场景全量（3 × 5 种子 × 5 折 ≈ 75 次拟合，保守 10–20 min）+ 补充 `--quick`；总计预算 30–60 min，tqdm 进度；`--quick` 兜底
- 数值守卫：成分数 A≤min(n−1,p)、PVE 下限、稳健尺度 MAD 下限、GBDT 早停、门控 w∈[0,1] 与 λ_g 正则
- **SIMPLS 对拍口径（等价性验证）**：SIMPLS（de Jong 1993）与 sklearn `PLSRegression`（NIPALS）计算路径不同，成分数 >1 时数值不完全相等、仅大样本渐进一致 → PASS 标准改为随机数据（n=50, p=10）上预测 RMSE 差 < 1e-6 或相关系数 > 0.999，注明"算法等价性验证"（详见 §8）

**风险应对速查（2026-08-07 三轮审阅合并）**：
| # | 风险 | 应对 | 落点 |
|---|------|------|------|
| 1 | SIMPLS 复现 vs sklearn NIPALS 数值不兼容 | 等价性验证（RMSE<1e-6 / corr>0.999） | §7、§8 |
| 2 | 分块 B-spline「分块」含义模糊 / Alin 复现依赖 | 单函数预测子 → 惩罚 B-spline + 多预测子块对角推广（不硬卖分块）；**贡献 Inspired by**；P1 读 SSRN 全文 | §1、P1 |
| 3 | AHFE 被质疑="PLS+残差树"工程拼接 / **与 FME 撞车** | **创新定位改为自适应残差校正门控**（非多专家 MoE 混合）+ FME/iFME 竞争文献 + oracle gate 理论 + AHFE-fixed 消融 | §2、§6、P1 |
| 4 | CatBoost Windows 安装失败 | 先装、失败立即跳过并声明回退（仅报 XGB/LGBM） | §5 P0 |
| 5 | 合成 DGP 识别性不足（∫X²β 退化 / β 受非线性项污染） | 零均值 GP + **中心化二次项** + 协方差算子正交 ⟨β,Cφ⟩=0 + S5b 阳性对照 + SNR 检查 | §3.1 |
| 6 | 运行预算失控 | 核心 3 场景全量 + 补充 `--quick` + tqdm；先 P4a 开发基准再铺全 | §5、§7 |
| 7 | M2 与 AHFE-B 特征不一致（不公平对照） | M2 与 AHFE-B **同特征库** | §3.1 |
| 8 | 自包含代码与 scikit-fda 混淆 | 知识库条目两小节明确分隔 | §4 |
| 9 | **门控泄漏/过拟合**（真实残差作特征 / 学到测试折） | 门控特征不含 y 相关量（用 \|r̂_B\|、Q-residual、leverage、FPCA scores）；**cross-fitting** OOF 训练门控；w∈[0,1]；校准报告 | §3.1 |
| 10 | **伪重复统计检验**（25 fold 当独立样本） | 主报告 ΔRMSE+95%CI + paired bootstrap / repeated-CV corrected；Holm 保留 | §3 |
| 11 | P4a"必须提升否则改"→ adaptive overfitting | **开发基准（自由迭代）/确证基准（方案冻结后全新种子）分离**，P5 失败如实报告 | §5 |
| 12 | 污染场景定义模糊 | S6a Y-outlier / S6b X-函数型离群显式定义；报告 ΔRMSE + w_outlier 诊断 | §3.1 |
| 13 | 实验矩阵庞杂、叙事散 | 核心 3 场景 + 4 方法 + 1 消融，机制验证导向 | §3、§5 |
| 14 | 门控验证标准不严谨（"S5 全体 w→1"） | 逐样本 NL_i + Spearman(w,NL)（如实报告，可能不显著）+ NL 分位数→平均 w 图；不预设 adaptive>fixed | §3、§3.1 |

---

## 7.1 付费需求（用户已授权人民币支付）

- **付费需求已基本消除（审稿意见 13）**：Alin 2026 全文已由作者在 **SSRN 放出 27 页预印本**（abstract_id 5593333），P1 优先直接读取。仅当 SSRN 链接失效或需 Elsevier 正式版排版时才需付费，届时向用户提出。
- 其余全部免费：Python 开源栈、Tecator/Gasoline 公开数据集、本地 CPU 计算。
- catboost 若需安装（PyPI 免费），P0 先装，失败立即跳过（Windows 与 NumPy 2 偶有兼容问题）。

---

## 8. 验证计划（三层分层，审稿意见 14）

### 层 1：单元测试（软件正确性，pytest ≥8，覆盖率≥80%）
维度、API、种子可复现、**无泄漏**（门控训练不接触外层测试 y）、权重范围（w∈[0,1]）、SIMPLS 数学性质（**等价性验证**：n=50, p=10 与 sklearn PLSRegression RMSE 差 < 1e-6 或 corr > 0.999，注明算法等价）、tecator 加载形状 (215,100)、特征折内拟合/折外投影、序列化。**科学假设不入单元测试。**

### 层 2：科学验证（benchmark_report，非 pytest）
AHFE 是否提升基座（ΔRMSE+95%CI+paired bootstrap，headline 为 AHFE vs M3 的 ΔR²≈+0.34）、鲁棒性是否保持（S6 ΔRMSE + w_outlier 诊断）、β̂ 恢复是否达标；**门控机制如实报告**：Spearman(w,NL) 与 NL 分位数→平均 w 图，若 adaptive≈fixed 则作为诚实发现报告（含可行 oracle 上界），**不以"adaptive 必须 > fixed"为通过标准**。用效应量与置信区间报告，**不因单次随机波动让 pytest 变红**。

### 层 3：确证基准（方案冻结后，全新种子/holdout）
复现最终结论；失败如实报告为科学发现。

**可复现容差**：XGBoost/LightGBM 多线程下 mean±std 一致 ≥1e-10 过严 → 设确定性模式后容差 **1e-6~1e-8**，或比较结果 hash / split index / 参数而非浮点逐位一致。

**其他**：全代码运行（`--quick` → 全量）无 import/语法错；数值合理性（R² 合理区间、无 NaN、β̂ 方向正确）；学术写作自检（kill list、长短句、信息密度、≥7 文献、无 AI 套话）；输出路径绝对、图表渲染、`_index.md` 计数一致。

## 待确认
- 核心实验集合（S1 + S5 + S6，默认如上；S2/S3/S4 为补充可选）
- 是否同时归档算法包至 `knowledge/algorithms/ahfe-functional-ensemble/`（默认：条目为主，算法包可选）

---

## 9. 执行方式（用户指定）

1. **计划持久化**：本文件即研究任务工作计划，随执行更新。
2. **阶段化 auto 模式**：按 P0→P1→…→P7 分阶段执行。每阶段 **auto 运行** → 完成 + **门禁核验**（表内 Gate 项）→ 状态记录到 `checkpoints/阶段状态.md` → 自动进入下一阶段。
3. **仅两类暂停**：关键决策点（P2 问题定义、P3 设计确认、方案冻结）与付费需求才暂停征询；其余自动推进。**P4a 开发基准内不设"性能不达标即回退"的硬门禁**——机制门禁为准，方案冻结后由 P5 确证基准独立裁决。
4. **范围沙箱**：所有写入限 `research-assistant/` 内，遵守 `00-scope-boundary.md`。
