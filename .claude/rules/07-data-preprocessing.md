---
paths:
  - "agents/data-viz/**/*"
  - "agents/mcm/data-agent.md"
---

# 数据预处理流水线

> **按需加载**：本文件只在**读取 `agents/data-viz/` 或 MCM 数据 Agent 时**进入上下文（做预处理前必先读
> 对应 Agent 文件，故必然触发）。它不承载任务闸门与路由——那两类规则保持常驻。
> 由父项目同名规则于 2026-09-20 下沉而来。

收到数据预处理需求（清洗、变换、特征工程）时，按以下流水线执行。
**先自动识别数据类型（表格 / 时间序列 / 函数型）**，再选对应处理路径——三类数据的清洗与变换手段不同，
不要用表格数据的手法处理函数型数据。

## 阶段 1：数据探查（Data Inspector）

1. 加载 `agents/data-viz/preprocessing-inspector.md` 作为子 Agent
2. 自动识别数据类型，输出针对性探查报告
3. 根据探查结果给出预处理建议方案
4. **用户确认后**才进入下一阶段

## 阶段 2：数据清洗（Data Cleaner）

1. 加载 `agents/data-viz/preprocessing-cleaner.md` 作为子 Agent
2. 执行清洗：缺失值处理、重复值剔除、异常值修正、类型校正
3. **时间序列**优先用 PyPOTS / SAITS 做智能插补（不要简单前向填充）
4. **函数型**优先用 scikit-fda / fdasrsf 做平滑和去噪
5. 输出清洗后数据 + 清洗日志

## 阶段 3：特征变换（Data Transformer）

1. 加载 `agents/data-viz/preprocessing-transformer.md` 作为子 Agent
2. 执行变换：编码、标准化/归一化、特征构造、降维
3. **时间序列额外**：tsfresh 时序特征提取、滑动窗、差分、频域变换
4. **函数型额外**：SRSF 弹性对齐、基展开、导数曲线、FPCA 降维
5. 输出变换后数据 + 变换日志

## 阶段 4：质量验证（Data Validator）

1. 加载 `agents/data-viz/preprocessing-validator.md` 作为子 Agent
2. 对比清洗前后质量报告
3. 检查数据是否符合建模要求
4. 输出 **PASS/FAIL** + 最终数据质量报告（含出版级可视化）

## 阶段 5：交付

预处理后数据 + 完整预处理脚本 + 质量报告 + **变换参数**（供模型侧复用，避免训练/推理不一致）。

## 工具选型参考（预处理专用）

跨领域的通用工具选型见 `.claude/rules/08-tool-selection.md`；本表只覆盖预处理环节。

| 任务 | 首选工具 |
|------|---------|
| 表格数据清洗 | pandas, scikit-learn |
| 时序缺失值填补 | PyPOTS (SAITS / BRITS) |
| 时序特征提取 | tsfresh（794 种特征） |
| 时序变换流水线 | sktime (Differencer, BoxCox, WindowSummarizer) |
| 函数型平滑/对齐 | scikit-fda, fdasrsf（SRSF 弹性对齐） |
| 函数型降维 | scikit-fda (FPCA) |
