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
