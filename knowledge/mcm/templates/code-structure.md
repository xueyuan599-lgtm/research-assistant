# CUMCM 代码目录结构模板

> 全国大学生数学建模竞赛代码包推荐结构。

## 目录结构

```
project/
├── data/
│   ├── raw/                    # 原始附件，只读（勿修改）
│   └── processed/              # 清洗后的中间数据
├── src/
│   ├── config.py               # 路径、随机种子、公共参数
│   ├── data_loader.py          # 文件读取与字段校验
│   ├── preprocessing.py        # 数据清洗与特征构造
│   ├── problem1.py             # 问题一模型 + 内嵌基准对照 + 适配检验
│   ├── problem2.py             # 问题二模型 + 内嵌基准对照 + 适配检验
│   ├── problem3.py             # 问题三模型 + 内嵌基准对照 + 适配检验
│   ├── problem4.py             # 问题四模型 + 内嵌基准对照（国赛通常 3-5 个小问，按需增减）
│   ├── problem5.py             # 问题五模型 + 内嵌基准对照
│   ├── sensitivity.py          # 情景与灵敏度分析
│   ├── visualization.py        # 图表生成
│   └── utils.py                # 日志、导出与通用函数
├── outputs/
│   ├── tables/                 # 结果表（CSV）
│   ├── figures/                # 图表（PNG/PDF）
│   ├── intermediate/           # 中间结果
│   └── logs/                   # 实验日志
├── tests/                      # 单元测试
├── main.py                     # 主运行入口
├── requirements.txt            # 依赖列表
└── README.md                   # 运行说明
```

## 规范要求

### 基准模型策略
- **不设独立顶层 `baseline.py`**：基准对照内嵌于各 `problemN.py`（`build_baseline()` / `compare_with_baseline()`）
- **只在关键小问**（分值最高/体现创新点/评委重点）做基准对比并突出展示
- 其余小问基准仅作内部诊断对照（判定"弱于基准→切换"），不写进论文

### 代码规范
- 中文注释
- 固定随机种子（config.py 集中配置）
- 文件读取含字段校验（缺文件/缺字段/非法数值时明确报错）
- 保存必要中间结果和运行日志
- 实验日志写入 `outputs/logs/experiment_log.csv`
- 随机算法记录种子、参数和运行时间

### 图表规范（推荐 Science/Nature 级）
- **推荐复用 `knowledge/mcm/templates/figures/`**（mcm_nature.mplstyle，中文适配）；模板为首选参考，具体图表方案由团队在 planner 阶段确认，允许定制
- **中文**：SimHei/Microsoft YaHei 字体栈 + `axes.unicode_minus=False`；所有标题/轴名/图例/刻度中文
- **格式**：矢量 PDF + 位图 PNG（300dpi, bbox tight）；尺寸：单栏 89mm / 双栏 183mm
- **配色**：Okabe-Ito 色盲安全；热图用 viridis/RdBu；无网格线；tick 朝外；轴框仅左+下
- **字号**：正文 7pt / 轴标签 8pt / 刻度 6pt / 图例 6pt（不描边）
- **选型**：按 `figures/README.md` 图表选型表（灵敏度→龙卷风图、相关→热图、轨迹→3D 等）
- **编号**：图下表上、三线表、坐标轴带单位、合理有效数字

### 运行规范
- `python main.py` 一键运行全部
- 支持逐问独立运行
- 运行时间可接受（单问 < 30 分钟）
