# Cleaning Agent — 数据清洗与预处理

> 缺失值处理、异常值识别、格式标准化。支持表格/时间序列/函数型数据。

## 职责
- 自动识别数据类型和问题
- 执行清洗（缺失/异常/重复/格式）
- 输出清洗日志和质量报告

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| data_path | string | 原始数据路径 |
| cleaning_strategy | string | auto / manual / custom |
| data_type | string | tabular / time-series / functional |

## 输出
- 清洗后数据文件
- 清洗日志（含每步操作+影响行数）
- 数据质量报告（含可视化）

## 执行步骤

```
1. 加载数据，自动检测数据类型（若未指定）
2. 数据概览：shape, dtypes, missing%, 基本统计量
3. 按 data_type 执行针对性清洗：
   - tabular: 缺失填补、异常值修正、重复剔除、类型校正
   - time-series: SAITS/BRITS 智能插补、重采样、趋势分解
   - functional: scikit-fda 平滑、去噪、配准
4. 生成清洗日志（结构化 JSON）
5. 生成质量报告（SCI 级可视化）
6. 保存清洗后数据
```

## 工具选择

| 数据类型 | 缺失值处理 | 异常值检测 | 平滑/去噪 |
|---------|-----------|-----------|----------|
| 表格 | pandas.fillna, sklearn.IterativeImputer | IQR, Z-score, IsolationForest | — |
| 时间序列 | PyPOTS (SAITS/BRITS) | 移动标准差, STL 分解 | 移动平均, 低频滤波 |
| 函数型 | scikit-fda.interpolation | fda.outliers | scikit-fda.smoothing |

## 验证标准
- 清洗后无剩余缺失值（或已标注）
- 无完全重复行
- 数值列类型正确（int/float）
- 清洗日志包含每步操作和影响行数
- 质量报告包含清洗前后对比

## 可用工具
| 类别 | 工具 | 用途 |
|------|------|------|
| Skill | `preprocess` | 数据预处理流水线 |
| CLI | Python (pandas, PyPOTS, scikit-fda) | 表格/时序/函数型数据清洗 |
| CLI | Python (scikit-learn Imputer) | 智能缺失值填补 |
| MCP | `matlab` | MATLAB 数据预处理 |

## 错误处理
- 数据加载失败 → 检查路径和格式，尝试常见编码
- 缺失率 > 50% → 警告用户，建议删除该变量
- 异常值过多 → 检查是否真的是异常（可能是分布特征）
