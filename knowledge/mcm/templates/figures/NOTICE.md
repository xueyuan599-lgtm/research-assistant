# NOTICE — 来源声明与许可

本目录为 **MCM Science / Nature 期刊级可视化库**，其中部分模板的**图型设计逻辑**改编自：

- **Academic Figure Skill** — https://github.com/TingxiYu/academic-figure-skill
- License: **Apache License 2.0**
- 获取日期：2026-07-31

## 改编来源图型

| 本库文件 | 函数 | 参考来源脚本 |
|---------|------|-------------|
| `template_radar.py` | `make_radar()` | `assets/figures/Radar/plot_comparison_radar.py`（polar 归一化逻辑） |
| `template_pca.py` | `make_pca()` | `assets/figures/PCA/plot_PCA.R`（降维 + 贡献率标注逻辑，R→Python 重写） |

## 合规说明

- 上表所列模板在文件头部 docstring 中保留了改编来源声明。
- 依据 Apache-2.0 第 4 条（Redistribution），保留版权、许可与声明文件。
- 本库其余文件（`export_figure.py`、`palette.py`、`figure_audit.py`、其余 `template_*.py` 等）为本项目独立实现，不包含上述来源代码。
- 若对外分发本库，须随附 Apache-2.0 许可证文本与本 NOTICE。
