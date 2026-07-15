# Algorithm Repository — SCI-Level Code

从顶刊提取的高质量方法实现。每条条目包含完整的数学设定和可复现代码。

## 条目格式

```markdown
# [方法名称]

- **来源**: 论文标题 | 期刊 | 年份
- **DOI**: xxx
- **方法类别**: 因果推断 / 时间序列 / 机器学习 / 统计建模 / 优化 / ...

## 数学设定
- 模型框架、核心方程
- 识别策略（如适用）
- 关键假设条件

## 适用场景
- 什么数据类型、样本量、问题适合用
- 什么时候**不**该用

## 实现要点
- 参数调优经验
- 数值注意事项
- 对比 benchmark 设置

## 代码

\`\`\`python
# 完整可运行实现
\`\`\`

## 参考文献
```

## 分类索引

### 经典方法

| 类别 | 条目 |
|------|------|
| 集成学习 | [Random Forest](random-forest.md), [XGBoost / LightGBM](xgboost.md) |
| 群智能优化 | [Gray Langurs Optimizer (GLO)](gray-langurs-optimizer.md), [Physarum Network Optimizer (PNO)](physarum-network-optimizer.md) |
| 因果推断 | [Differences-in-Differences (DID)](did.md), [RDD 断点回归](rdd.md) |
| 时间序列 | [ARIMA/SARIMA](arima.md), [Prophet](prophet.md), [GARCH](garch.md), [VAR/VEC](var.md), [State Space / Kalman Filter](state-space.md) |
| 统计建模 | [Lasso / Ridge / ElasticNet](lasso-ridge-elasticnet.md) |
| 机器学习 | [SVM (SVC/SVR)](svm.md), [K-Means / DBSCAN](clustering.md), [PCA / t-SNE / UMAP](dimensionality-reduction.md) |
| 函数型数据分析 | [FPCA / 函数型回归](fpca.md) |
| 贝叶斯方法 | [MCMC (MH/Gibbs/HMC)](mcmc.md), [Gaussian Process](gaussian-process.md) |
| 深度学习 | [LSTM/GRU](lstm.md), [Transformer](transformer.md), [Autoencoder / VAE](autoencoder.md) |

### 2021-2026 顶刊前沿方法

| 领域 | 条目 | 覆盖期刊/会议 |
|------|------|-------------|
| **运筹优化** (8) | [Adaptive ADMM](adaptive-admm.md), [Bayesian DRO](bayesian-distributionally-robust-optimization.md), [Dynamic Optimization w/ Side Info](dynamic-optimization-side-information.md), [First-Order Penalty Bilevel Opt](first-order-penalty-bilevel-optimization.md), [ML for Spatial Branching](learning-for-spatial-branching.md), [ML-Enhanced L-Shaped Method](l-shaped-heuristics-supervised-learning.md), [MIP over ReLU Ensembles](optimizing-ensemble-neural-networks.md), [Pareto Dominance DDO](pareto-dominance-data-driven-optimization.md) | *SIAM J. Optim.*, *Operations Research*, *INFORMS J. Comput.*, *EJOR*, *JOTA* |
| **机器学习** (7) | [Mamba: Selective SSM](mamba-selective-state-space-model.md), [Flow Matching](flow-matching-generative-modeling.md), [FlashAttention](flashattention-io-aware-attention.md), [Direct Preference Optimization](direct-preference-optimization.md), [Masked Autoencoder (MAE)](masked-autoencoder-vision.md), [Fourier Neural Operator](fourier-neural-operator.md), [Segment Anything Model (SAM)](segment-anything-model.md) | *NeurIPS*, *ICLR*, *CVPR*, *ICCV* |
| **统计学** (7) | [Conformal Prediction Beyond Exchangeability](conformal-prediction-beyond-exchangeability.md), [Conformal Q-values for FDR](conformal-q-values-fdr-control.md), [Derandomised Knockoffs](derandomised-knockoffs.md), [Localized Conformal Prediction](localized-conformal-prediction.md), [Tensor CP Matrix Time Series](tensor-cp-decomposition-matrix-time-series.md), [Vecchia GP Approximation](vecchia-approximation-gaussian-processes.md), [Selective Inference Effect Modification](selective-inference-effect-modification-lasso.md) | *Ann. Statist.*, *JRSS-B*, *Biometrika*, *Statist. Sci.*, *JASA* |
| **因果推断** (8) | [Double/Debiased ML (DML)](double-machine-learning.md), [Causal Forest (GRF)](causal-forest.md), [CATE Meta-Learners](cate-meta-learners.md), [Sensitivity Analysis (OVB)](sensitivity-analysis-omitted-variable.md), [DeepIV / Neural IV](deep-instrumental-variables.md), [Targeted Maximum Likelihood (TMLE)](targeted-maximum-likelihood-estimation.md), [Causal Representation Learning](causal-representation-learning.md), [Network Causal Inference](network-causal-inference.md) | *Econometrica*, *JRSS-B*, *Biometrika*, *PNAS*, *NeurIPS*, *ICLR/ICML*, *Ann. Appl. Stat.* |
| **生信分析** (8) | [AlphaFold 3](alphafold3-biomolecular-structure-prediction.md), [Cell2location](cell2location-spatial-deconvolution.md), [CellRank](cellrank-trajectory-inference.md), [Enformer](enformer-gene-expression-prediction.md), [Geneformer](geneformer-single-cell-foundation-model.md), [MOFA+ / MEFISTO](mofa-multi-omics-factor-analysis.md), [ProteinMPNN](proteinmpnn-sequence-design.md), [scVI / scANVI](scvi-single-cell-deep-generative-model.md) | *Nature*, *Nature Biotech.*, *Nature Methods*, *Science*, *Genome Biology* |

> **总计**: **59 个条目**（21 经典方法 + 38 个 2021-2026 顶刊前沿），均来自顶刊/顶会。
> **语言**: 英文撰写，含完整数学设定（LaTeX）+ 可运行 Python 代码 + APA 7th 引用。
> **维护**: 如需扩充条目，请联系项目维护者或通过秘书 Agent 提交需求。
