# Algorithm Repository — SCI-Level Code

从顶刊提取的高质量方法实现。每条条目包含完整的数学设定和可复现代码。
**所有条目已统一为 YAML frontmatter 多标签元数据（`type` / `domain`），支持按题型谱系三层多标签联合检索。** Schema 见 [_SCHEMA.md](_SCHEMA.md)。

## 分类浏览表（快速定位）

### 经典方法

| 类别 | 条目 |
|------|------|
| 集成学习 | [Random Forest](random-forest.md), [XGBoost / LightGBM](xgboost.md) |
| 群智能优化 | [Gray Langurs Optimizer (GLO)](gray-langurs-optimizer.md), [Physarum Network Optimizer (PNO)](../algorithms/physarum-network-optimizer/README.md) |
| 因果推断 | [Differences-in-Differences (DID)](did.md), [RDD 断点回归](rdd.md) |
| 时间序列 | [ARIMA/SARIMA](arima.md), [Prophet](prophet.md), [GARCH](garch.md), [VAR/VEC](var.md), [State Space / Kalman](state-space.md) |
| 统计建模 | [Lasso / Ridge / ElasticNet](lasso-ridge-elasticnet.md), [MICE 多重插补](mice.md) |
| 机器学习 | [SVM (SVC/SVR)](svm.md), [K-Means / DBSCAN](clustering.md), [PCA / t-SNE / UMAP](dimensionality-reduction.md), [SHAP 可解释性](shap.md), [Tabular vs Deep 选型](tabular-vs-deep.md) |
| 函数型数据分析 | [FPCA / 函数型回归](fpca.md), [残差校正混合函数型集成 (AHFE)](../algorithms/functional-ensemble-regression/README.md) |
| 贝叶斯方法 | [MCMC (MH/Gibbs/HMC)](mcmc.md), [Gaussian Process](gaussian-process.md) |
| 深度学习 | [LSTM/GRU](lstm.md), [Transformer](transformer.md), [Autoencoder / VAE](autoencoder.md), [MLP 基础](mlp-basics.md) |

> **交叉引用说明（2026-09-23）**：本表含 **2 个自创方法**，实体均在自创算法库 `../algorithms/`，
> 此处只保留反向索引链接以维持标签检索命中：
>
> - **AHFE（残差校正混合函数型集成）** → `../algorithms/functional-ensemble-regression/README.md`
>   （标签 `functional` / `ensemble-learning` / `regression`）
> - **PNO-GWO（黏菌管网络-灰狼混合优化）** → `../algorithms/physarum-network-optimizer/README.md`
>   （标签 `heuristic` / `optimization` / `graph`）
>
> 两者原在本库各有一份**内容重复**的引用文档。重复已实际漂移出错误——同一张基准表里 Rastrigin 的
> 0.89× 退化被标成"持平"（两份副本都错）——故于 2026-09-23 删除本库副本，标签随算法本体迁入其
> README 的 frontmatter。本库其余 **80 个条目**均为顶刊提取的已有方法。

### 2021-2026 顶刊前沿方法

| 领域 | 条目 |
|------|------|
| **运筹优化** (8) | [Adaptive ADMM](adaptive-admm.md), [Bayesian DRO](bayesian-distributionally-robust-optimization.md), [Dynamic Optimization w/ Side Info](dynamic-optimization-side-information.md), [First-Order Penalty Bilevel Opt](first-order-penalty-bilevel-optimization.md), [ML for Spatial Branching](learning-for-spatial-branching.md), [ML-Enhanced L-Shaped Method](l-shaped-heuristics-supervised-learning.md), [MIP over ReLU Ensembles](optimizing-ensemble-neural-networks.md), [Pareto Dominance DDO](pareto-dominance-data-driven-optimization.md) |
| **机器学习** (8) | [Mamba: Selective SSM](mamba-selective-state-space-model.md), [xLSTM: Extended LSTM](xlstm-extended-lstm.md), [Flow Matching](flow-matching-generative-modeling.md), [FlashAttention](flashattention-io-aware-attention.md), [Direct Preference Optimization](direct-preference-optimization.md), [Masked Autoencoder (MAE)](masked-autoencoder-vision.md), [Fourier Neural Operator](fourier-neural-operator.md), [Segment Anything Model (SAM)](segment-anything-model.md) |
| **时序融合** (14) | [LSTM-PINN Battery SOH](lstm-pinn-battery-soh.md), [Physics-Informed LSTM](physics-informed-lstm.md), [Dual-Level PI Forecasting](dual-level-physics-informed-forecasting.md), [LSTM-PINN Seismic](lstm-pinn-seismic-response.md), [Autoencoder LSTM-PINN](autoencoder-lstm-pinn.md), [MDSTFT](mdstft.md), [TCN-Transformer-LSTM](tcn-transformer-lstm-ttl.md), [TIC-FusionNet](tic-fusionnet.md), [Fusion ConvLSTM-Net](fusion-convlstm.md), [Data-Model Hybrid Survey](data-model-hybrid-driven-survey.md), [Hybrid Physics-ML Taxonomy](hybrid-physics-ml-taxonomy.md), [CS-LSTMs 双分支异常检测](contextual-seasonal-lstms-anomaly-detection.md), [LSTM-DLN 非参数 CDF 预测](lstm-dln-nonparametric-cdf-forecasting.md), [TSB 多通道频谱预测](tsb-transformer-stacked-bilstm-spectrum-prediction.md) |
| **统计学** (7) | [Conformal Prediction Beyond Exchangeability](conformal-prediction-beyond-exchangeability.md), [Conformal Q-values for FDR](conformal-q-values-fdr-control.md), [Derandomised Knockoffs](derandomised-knockoffs.md), [Localized Conformal Prediction](localized-conformal-prediction.md), [Tensor CP Matrix Time Series](tensor-cp-decomposition-matrix-time-series.md), [Vecchia GP Approximation](vecchia-approximation-gaussian-processes.md), [Selective Inference Effect Modification](selective-inference-effect-modification-lasso.md) |
| **因果推断** (8) | [Double/Debiased ML (DML)](double-machine-learning.md), [Causal Forest (GRF)](causal-forest.md), [CATE Meta-Learners](cate-meta-learners.md), [Sensitivity Analysis (OVB)](sensitivity-analysis-omitted-variable.md), [DeepIV / Neural IV](deep-instrumental-variables.md), [Targeted Maximum Likelihood (TMLE)](targeted-maximum-likelihood-estimation.md), [Causal Representation Learning](causal-representation-learning.md), [Network Causal Inference](network-causal-inference.md) |
| **生信分析** (8) | [AlphaFold 3](alphafold3-biomolecular-structure-prediction.md), [Cell2location](cell2location-spatial-deconvolution.md), [CellRank](cellrank-trajectory-inference.md), [Enformer](enformer-gene-expression-prediction.md), [Geneformer](geneformer-single-cell-foundation-model.md), [MOFA+ / MEFISTO](mofa-multi-omics-factor-analysis.md), [ProteinMPNN](proteinmpnn-sequence-design.md), [scVI / scANVI](scvi-single-cell-deep-generative-model.md) |

---

## 🏷️ 机器可读多标签反向索引（type → 条目）

> **检索协议依据**：赛题小题按"三层题型谱系"（决策层/计算层/物理层）取标签，对本表做联合命中。
> 某层标签命中即该层相关；三层/两层联合命中者最强。
> **空档提醒**：下方标注 🅰️ 为"库内暂无专用条目"的标签——赛题如命中这些维度，需 WebSearch 兜底。

| type 标签 | 命中条目 |
|-----------|---------|
| `biomolecular` | [alphafold3](alphafold3-biomolecular-structure-prediction.md), [enformer](enformer-gene-expression-prediction.md), [geneformer](geneformer-single-cell-foundation-model.md), [proteinmpnn](proteinmpnn-sequence-design.md) |
| `causal` | [cate-meta-learners](cate-meta-learners.md), [causal-forest](causal-forest.md), [causal-representation-learning](causal-representation-learning.md), [deep-instrumental-variables](deep-instrumental-variables.md), [did](did.md), [double-machine-learning](double-machine-learning.md), [network-causal-inference](network-causal-inference.md), [rdd](rdd.md), [selective-inference](selective-inference-effect-modification-lasso.md), [sensitivity-ovb](sensitivity-analysis-omitted-variable.md), [tmle](targeted-maximum-likelihood-estimation.md) |
| `classification` | [random-forest](random-forest.md), [shap](shap.md), [svm](svm.md), [tabular-vs-deep](tabular-vs-deep.md), [xgboost](xgboost.md) |
| `clustering` | [clustering](clustering.md) |
| `computer-vision` | [masked-autoencoder-vision](masked-autoencoder-vision.md), [segment-anything-model](segment-anything-model.md) |
| `connectivity` | [geometric-percolation-connectivity](geometric-percolation-connectivity.md) |
| `deep-learning` | [alphafold3](alphafold3-biomolecular-structure-prediction.md), [autoencoder](autoencoder.md), [autoencoder-lstm-pinn](autoencoder-lstm-pinn.md), [causal-rep-learning](causal-representation-learning.md), [cell2location](cell2location-spatial-deconvolution.md), [cellrank](cellrank-trajectory-inference.md), [cs-lstms](contextual-seasonal-lstms-anomaly-detection.md), [data-model-hybrid-survey](data-model-hybrid-driven-survey.md), [deepiv](deep-instrumental-variables.md), [dim-reduction](dimensionality-reduction.md), [dpo](direct-preference-optimization.md), [dual-level-pi](dual-level-physics-informed-forecasting.md), [enformer](enformer-gene-expression-prediction.md), [flashattention](flashattention-io-aware-attention.md), [flow-matching](flow-matching-generative-modeling.md), [fno](fourier-neural-operator.md), [fusion-convlstm](fusion-convlstm.md), [geneformer](geneformer-single-cell-foundation-model.md), [hybrid-physics-ml](hybrid-physics-ml-taxonomy.md), [l-shape-spatial-branching](learning-for-spatial-branching.md), [lstm-pinn-battery](lstm-pinn-battery-soh.md), [lstm-pinn-seismic](lstm-pinn-seismic-response.md), [lstm](lstm.md), [lstm-dln](lstm-dln-nonparametric-cdf-forecasting.md), [mamba](mamba-selective-state-space-model.md), [mae](masked-autoencoder-vision.md), [mdstft](mdstft.md), [mlp-basics](mlp-basics.md), [optimizing-ensemble-nn](optimizing-ensemble-neural-networks.md), [pi-lstm](physics-informed-lstm.md), [proteinmpnn](proteinmpnn-sequence-design.md), [scvi](scvi-single-cell-deep-generative-model.md), [sam](segment-anything-model.md), [tabular-vs-deep](tabular-vs-deep.md), [tcn-transformer-lstm](tcn-transformer-lstm-ttl.md), [tsb](tsb-transformer-stacked-bilstm-spectrum-prediction.md), [tic-fusionnet](tic-fusionnet.md), [transformer](transformer.md), [xlstm](xlstm-extended-lstm.md) |
| `dimensionality-reduction` | [causal-rep-learning](causal-representation-learning.md), [dim-reduction](dimensionality-reduction.md), [fpca](fpca.md), [mofa](mofa-multi-omics-factor-analysis.md) |
| `ensemble-learning` | [causal-forest](causal-forest.md), [functional-ensemble](../algorithms/functional-ensemble-regression/README.md), [random-forest](random-forest.md), [tabular-vs-deep](tabular-vs-deep.md), [xgboost](xgboost.md) |
| `financial` | [garch](garch.md), [var](var.md) |
| `forecasting` | [arima](arima.md), [autoencoder-lstm-pinn](autoencoder-lstm-pinn.md), [cs-lstms](contextual-seasonal-lstms-anomaly-detection.md), [data-model-hybrid-survey](data-model-hybrid-driven-survey.md), [dual-level-pi](dual-level-physics-informed-forecasting.md), [fusion-convlstm](fusion-convlstm.md), [garch](garch.md), [hybrid-physics-ml](hybrid-physics-ml-taxonomy.md), [lstm-pinn-battery](lstm-pinn-battery-soh.md), [lstm-pinn-seismic](lstm-pinn-seismic-response.md), [lstm](lstm.md), [lstm-dln](lstm-dln-nonparametric-cdf-forecasting.md), [mdstft](mdstft.md), [pi-lstm](physics-informed-lstm.md), [prophet](prophet.md), [state-space](state-space.md), [tcn-transformer-lstm](tcn-transformer-lstm-ttl.md), [tsb](tsb-transformer-stacked-bilstm-spectrum-prediction.md), [tic-fusionnet](tic-fusionnet.md), [transformer](transformer.md), [xlstm](xlstm-extended-lstm.md), [var](var.md) |
| `functional` | [fpca](fpca.md), [functional-ensemble](../algorithms/functional-ensemble-regression/README.md) |
| `generative-modeling` | [alphafold3](alphafold3-biomolecular-structure-prediction.md), [autoencoder](autoencoder.md), [flow-matching](flow-matching-generative-modeling.md), [mae](masked-autoencoder-vision.md), [scvi](scvi-single-cell-deep-generative-model.md) |
| `geometry` | [geometric-percolation-connectivity](geometric-percolation-connectivity.md) |
| `graph` | [cellrank](cellrank-trajectory-inference.md), [geometric-percolation-connectivity](geometric-percolation-connectivity.md), [network-causal](network-causal-inference.md), [physarum](../algorithms/physarum-network-optimizer/README.md) |
| `heuristic` | [gray-langurs](gray-langurs-optimizer.md), [l-shaped-heuristics](l-shaped-heuristics-supervised-learning.md), [physarum](../algorithms/physarum-network-optimizer/README.md) |
| `inverse-problem` | [inverse-problem-root-finding-reliability](inverse-problem-root-finding-reliability.md) |
| `matrix-tensor` | [mofa](mofa-multi-omics-factor-analysis.md), [tensor-cp](tensor-cp-decomposition-matrix-time-series.md) |
| `monte-carlo` | [mcmc](mcmc.md), [monte-carlo-random-placement-simulation](monte-carlo-random-placement-simulation.md) |
| `multi-objective` | [pareto-dominance](pareto-dominance-data-driven-optimization.md) |
| `numerical` | [adaptive-admm](adaptive-admm.md), [first-order-penalty-bilevel](first-order-penalty-bilevel-optimization.md), [flashattention](flashattention-io-aware-attention.md), [fno](fourier-neural-operator.md), [spatial-branching](learning-for-spatial-branching.md), [vecchia-gp](vecchia-approximation-gaussian-processes.md) |
| `optimization` | [adaptive-admm](adaptive-admm.md), [bayesian-dro](bayesian-distributionally-robust-optimization.md), [dpo](direct-preference-optimization.md), [dyn-opt-side-info](dynamic-optimization-side-information.md), [first-order-penalty-bilevel](first-order-penalty-bilevel-optimization.md), [gray-langurs](gray-langurs-optimizer.md), [inverse-problem-root-finding-reliability](inverse-problem-root-finding-reliability.md), [l-shaped-heuristics](l-shaped-heuristics-supervised-learning.md), [spatial-branching](learning-for-spatial-branching.md), [optimizing-ensemble-nn](optimizing-ensemble-neural-networks.md), [pareto-dominance](pareto-dominance-data-driven-optimization.md), [physarum](../algorithms/physarum-network-optimizer/README.md) |
| `pareto` | [pareto-dominance](pareto-dominance-data-driven-optimization.md) |
| `percolation` | [geometric-percolation-connectivity](geometric-percolation-connectivity.md) |
| `physics-informed` | [autoencoder-lstm-pinn](autoencoder-lstm-pinn.md), [data-model-hybrid-survey](data-model-hybrid-driven-survey.md), [dual-level-pi](dual-level-physics-informed-forecasting.md), [fno](fourier-neural-operator.md), [hybrid-physics-ml](hybrid-physics-ml-taxonomy.md), [lstm-pinn-battery](lstm-pinn-battery-soh.md), [lstm-pinn-seismic](lstm-pinn-seismic-response.md), [pi-lstm](physics-informed-lstm.md) |
| `preprocessing` | [cs-lstms](contextual-seasonal-lstms-anomaly-detection.md), [mice](mice.md) |
| `quantile-reliability` | [inverse-problem-root-finding-reliability](inverse-problem-root-finding-reliability.md) |
| `regression` | [cate-meta-learners](cate-meta-learners.md), [causal-forest](causal-forest.md), [dml](double-machine-learning.md), [functional-ensemble](../algorithms/functional-ensemble-regression/README.md), [gaussian-process](gaussian-process.md), [lasso-ridge-elasticnet](lasso-ridge-elasticnet.md), [lstm-dln](lstm-dln-nonparametric-cdf-forecasting.md), [random-forest](random-forest.md), [selective-inference](selective-inference-effect-modification-lasso.md), [shap](shap.md), [svm](svm.md), [tabular-vs-deep](tabular-vs-deep.md), [tmle](targeted-maximum-likelihood-estimation.md), [vecchia-gp](vecchia-approximation-gaussian-processes.md), [xgboost](xgboost.md) |
| `sensitivity` | [sensitivity-ovb](sensitivity-analysis-omitted-variable.md) |
| `sequence-modeling` | [cs-lstms](contextual-seasonal-lstms-anomaly-detection.md), [enformer](enformer-gene-expression-prediction.md), [fusion-convlstm](fusion-convlstm.md), [geneformer](geneformer-single-cell-foundation-model.md), [lstm](lstm.md), [lstm-dln](lstm-dln-nonparametric-cdf-forecasting.md), [mamba](mamba-selective-state-space-model.md), [proteinmpnn](proteinmpnn-sequence-design.md), [tcn-transformer-lstm](tcn-transformer-lstm-ttl.md), [tsb](tsb-transformer-stacked-bilstm-spectrum-prediction.md), [transformer](transformer.md), [xlstm](xlstm-extended-lstm.md) |
| `signal-processing` | [cs-lstms](contextual-seasonal-lstms-anomaly-detection.md), [tic-fusionnet](tic-fusionnet.md), [tsb](tsb-transformer-stacked-bilstm-spectrum-prediction.md) |
| `simulation` | [monte-carlo-random-placement-simulation](monte-carlo-random-placement-simulation.md) |
| `spatial` | [cell2location](cell2location-spatial-deconvolution.md), [mdstft](mdstft.md), [vecchia-gp](vecchia-approximation-gaussian-processes.md) |
| `statistical-inference` | [arima](arima.md), [cell2location](cell2location-spatial-deconvolution.md), [cellrank](cellrank-trajectory-inference.md), [conformal-exchangeability](conformal-prediction-beyond-exchangeability.md), [conformal-fdr](conformal-q-values-fdr-control.md), [derandomised-knockoffs](derandomised-knockoffs.md), [did](did.md), [dml](double-machine-learning.md), [lasso-ridge-elasticnet](lasso-ridge-elasticnet.md), [localized-conformal](localized-conformal-prediction.md), [mcmc](mcmc.md), [mice](mice.md), [mofa](mofa-multi-omics-factor-analysis.md), [network-causal](network-causal-inference.md), [prophet](prophet.md), [rdd](rdd.md), [scvi](scvi-single-cell-deep-generative-model.md), [selective-inference](selective-inference-effect-modification-lasso.md), [shap](shap.md), [state-space](state-space.md), [tmle](targeted-maximum-likelihood-estimation.md), [tensor-cp](tensor-cp-decomposition-matrix-time-series.md), [var](var.md) |
| `stochastic-process` | [dyn-opt-side-info](dynamic-optimization-side-information.md), [garch](garch.md), [gaussian-process](gaussian-process.md), [l-shaped-heuristics](l-shaped-heuristics-supervised-learning.md), [mcmc](mcmc.md), [state-space](state-space.md) |
| `time-series` | [arima](arima.md), [cs-lstms](contextual-seasonal-lstms-anomaly-detection.md), [garch](garch.md), [lstm-dln](lstm-dln-nonparametric-cdf-forecasting.md), [prophet](prophet.md), [state-space](state-space.md), [tensor-cp](tensor-cp-decomposition-matrix-time-series.md), [tsb](tsb-transformer-stacked-bilstm-spectrum-prediction.md), [var](var.md) |
| `uncertainty-quantification` | [bayesian-dro](bayesian-distributionally-robust-optimization.md), [conformal-exchangeability](conformal-prediction-beyond-exchangeability.md), [conformal-fdr](conformal-q-values-fdr-control.md), [cs-lstms](contextual-seasonal-lstms-anomaly-detection.md), [dyn-opt-side-info](dynamic-optimization-side-information.md), [gaussian-process](gaussian-process.md), [inverse-problem-root-finding-reliability](inverse-problem-root-finding-reliability.md), [localized-conformal](localized-conformal-prediction.md), [lstm-dln](lstm-dln-nonparametric-cdf-forecasting.md), [monte-carlo-random-placement-simulation](monte-carlo-random-placement-simulation.md), [sensitivity-ovb](sensitivity-analysis-omitted-variable.md), [vecchia-gp](vecchia-approximation-gaussian-processes.md) |

### 🅰️ 库内暂无专用条目的高频标签（赛题命中需 WebSearch 兜底）

> 已补条目：华数 A 类缺口（几何渗透/布放仿真/反问题）已新增 3 个专用条目。
> 下表为**当前仍空缺**的标签（主要来自华数 C 类"机理"缺口）。

| 标签 | 缺口说明 |
|------|---------|
| `mechanism-wip` → `mechanism`（物理机理条目） | 储能充放电动态、电耗/碳排映射、PUE 关系等机理建模（华数 C 问题二三；库内 0 条目，需外部检索 energy storage scheduling / PUE model） |
| `bandit-rl` | 强化学习/序贯决策（多阶段调度，库内暂无） |
| `dynamic-programming` | 动态规划（调度/背包类，库内暂无） |

> `energy-systems`（领域）已从本表移除：该 domain 现有 3 条专用条目覆盖 ——
> [fusion-convlstm](fusion-convlstm.md)（多尺度 ConvLSTM 融合）、
> [lstm-pinn-battery-soh](lstm-pinn-battery-soh.md)（电池 SOH）、
> [lstm-dln](lstm-dln-nonparametric-cdf-forecasting.md)（光伏辐照度概率预测）。
> 本表是 **type** 反向索引，domain 标签不进此表；储能系统层的**机理**建模（充放电动态、
> 碳排映射、PUE）仍缺，见上表 `mechanism` 行。

> 说明：华数 A 的 `geometry`/`percolation`/`connectivity`/`simulation`/`inverse-problem`/`quantile-reliability`
> 已由新条目补齐（[geometric-percolation-connectivity](geometric-percolation-connectivity.md)、
> [monte-carlo-random-placement-simulation](monte-carlo-random-placement-simulation.md)、
> [inverse-problem-root-finding-reliability](inverse-problem-root-finding-reliability.md)）。

---

## 🔗 复合题型三层谱系 → 标签映射速查

> 单个小题常为融合型，按"决策层/计算层/物理层"三层拆解后分别取标签联合检索。
> 全词表与层定义见 [_SCHEMA.md](_SCHEMA.md)。

华数杯 2026 A 题（微构体导电介质）实证示例：

| 小问 | 决策层 | 计算层 | 物理层 | 知识库命中方向 |
|------|--------|--------|--------|--------------|
| 问题一 导通判断 | —（直接判定） | `graph`+`connectivity` | `geometry`+`percolation` | ✅ [geometric-percolation-connectivity](geometric-percolation-connectivity.md) 三层命中 |
| 问题二 导通概率 | `uncertainty-quantification` | `simulation`+`monte-carlo` | `geometry`+`percolation` | ✅ [monte-carlo-random-placement](monte-carlo-random-placement-simulation.md) 计算/物理层命中 + UQ 族 |
| 问题三 最低填充量 | `inverse-problem`+`quantile-reliability` | `simulation`+`monte-carlo` | `geometry`+`percolation` | ✅ [inverse-problem-root-finding](inverse-problem-root-finding-reliability.md) + MC + 几何 复合命中 |
| 问题四 最低成本配比 | `optimization`+`multi-objective` | `simulation`+`monte-carlo` | `geometry`+`percolation` | ✅ 优化族（pareto、DRO）+ [inverse-problem](inverse-problem-root-finding-reliability.md) 强命中 |

华数杯 2026 C 题（算电协同多目标调度）实证示例：

| 小问 | 决策层 | 计算层 | 物理层 | 知识库命中方向 |
|------|--------|--------|--------|--------------|
| 问题一 统计+预测+基础调度 | `optimization` | `forecasting`+`statistical-inference` | `time-series` | ✅ arima/garch/prophet/state-space 时序族 + 优化族双层命中 |
| 问题二 碳感知多目标调度 | `optimization`+`multi-objective` | `optimization`+`numerical` | `mechanism` | ⚠️ 决策/计算层优化族强命中；物理层 `mechanism` 空（需补机理条目） |
| 问题三 储能协同+影响 | `optimization`+`sensitivity` | `optimization`+`simulation` | `mechanism`+`energy-systems` | ⚠️ 双层优化命中；储能/机理层空缺 |
| 问题四 多区域协同+场景 | `optimization`+`multi-objective`+`uncertainty-quantification` | `optimization`+`simulation` | `mechanism`+`stochastic-process` | ✅ 三层全中：[dyn-opt-side-info](dynamic-optimization-side-information.md)、[l-shaped-heuristics](l-shaped-heuristics-supervised-learning.md) |

---

> **总计**: **80 个条目**（77 方法条目 + 3 个赛题适配条目：几何渗透/布放仿真/反问题），均含完整数学设定 + 可运行 Python 代码。
> （原 82 条中的 2 个自创方法 —— AHFE 与 PNO-GWO —— 已迁出自创算法库 `../algorithms/`，见上方交叉引用说明。）
> **元数据**: 全部条目已迁移为 `_SCHEMA.md` 统一 frontmatter（多标签 `type` / `domain`）。
> **维护**: 新条目须遵循 `_SCHEMA.md` schema（首行 frontmatter + 受控标签）才能被多标签联合检索命中。
> **最近入库**（2026-09-10，时序/序列建模 4 条）: [CS-LSTMs 双分支异常检测](contextual-seasonal-lstms-anomaly-detection.md)、[LSTM-DLN 非参数 CDF 预测](lstm-dln-nonparametric-cdf-forecasting.md)、[xLSTM](xlstm-extended-lstm.md)、[TSB 多通道频谱预测](tsb-transformer-stacked-bilstm-spectrum-prediction.md)。
