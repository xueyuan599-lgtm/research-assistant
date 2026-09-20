# /preprocess

一键执行数据预处理流水线。自动识别数据类型（表格/时间序列/函数型/面板），输出 SCI/IEEE 发表级的数据预处理包。

## 用法
```
/preprocess 数据在 [路径]，需要做 [任务描述]
```

### 示例
```
/preprocess 数据在 data/raw.csv，因变量是 y，做缺失值插补和标准化
```
```
/preprocess 数据在 data/sensor.csv，时间序列数据，需要平滑和特征提取
```
```
/preprocess 数据在 data/curves.csv，函数型数据，需要对齐和FPCA降维
```
```
/preprocess 数据在 data/panel.csv，面板数据，需要平衡面板和个体效应处理
```

## 推荐工具链（自动选用）
| 数据类型 | 清洗 | 变换 |
|---------|------|------|
| 表格 | pandas, scikit-learn | scikit-learn (PCA, StandardScaler) |
| 时间序列 | PyPOTS (SAITS/BRITS) | tsfresh, sktime |
| 函数型 | scikit-fda, fdasrsf | scikit-fda (FPCA), fdasrsf (SRSF对齐) |
| 面板 | pandas, linearmodels | 个体/时间效应分解 |

## 执行流程
1. 加载 `.claude/rules/00-data-preprocessing.md`
2. 按阶段调度 `research-assistant/agents/data-viz/` 下的 preprocessing-inspector → preprocessing-cleaner → preprocessing-transformer → preprocessing-validator
3. 交付：预处理后数据 + 完整脚本 + 质量报告（含 SCI 级可视化）
4. 任务规模属于 T1（常规单任务）时可直接执行，无需秘书分解停点；T2/T3 先经秘书分解
