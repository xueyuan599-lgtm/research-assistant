# 工具选型（多语言）

> 由父项目同名规则于 2026-09-20 下沉而来。**本表是跨领域工具选型的唯一源**；
> 预处理专用工具见 `.claude/rules/07-data-preprocessing.md`，赛道内工具清单见对应赛道规则。

## 原则

对所有数据分析、建模、可视化任务，**优先选择最适合该任务的工具**，不局限于 skill 文档的默认语言。

## 选型指南

| 任务类型 | 首选 | 次选 |
|---------|------|------|
| 统计建模（OLS / GLM / Mixed / ARIMA） | R (`lm`, `lme4`, `forecast`) | Python (`statsmodels`), MATLAB |
| 机器学习（RF / XGBoost / SVM / 回归 / 分类 / 聚类） | Python (`scikit-learn`) | R (`caret`, `tidymodels`), MATLAB |
| 深度学习 | Python (`PyTorch`, `TF`) | — |
| 时间序列预测 | R (`forecast`, `tsibble`) | Python (`statsmodels`, `timesfm`) |
| 地理空间分析 | R (`sf`, `terra`, `raster`) | Python (`geopandas`, `rasterio`), MATLAB |
| 优化 / 运筹 | Python (`pymoo`, `scipy`) | MATLAB (`Optimization Toolbox`), R |
| 数据可视化 | Python (`matplotlib`, `seaborn`, `plotnine`) | R (`ggplot2`), MATLAB |
| 贝叶斯推断 | R (`brms`, `rstanarm`) | Python (`PyMC`), MATLAB |
| 符号数学 / 公式推导 | Python (`sympy`) | MATLAB (`Symbolic Math Toolbox`) |
| 文献计量 / 网络分析 | R (`bibliometrix`) | Python (`networkx`) |
| 数据处理（dplyr 风格） | R (`tidyverse`) | Python (`pandas`, `polars`) |
| 数据处理（大规模） | Python (`polars`, `dask`) | R (`data.table`) |

## 行为要求

- 用某个 skill 前，**先判断什么工具最适合当前任务**，不沿用 skill 的默认语言
- 选型须在方案中**写明理由**；用户明确指定工具时**服从用户选择**
- 代码须在所选工具的环境下真实可运行（验证前提见 `.claude/rules/06-cost-discipline.md` §一）
