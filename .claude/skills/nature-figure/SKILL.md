
# nature-figure

Generate Nature-journal-quality multi-panel scientific figures using matplotlib + SciencePlots.

## Trigger keywords
- "Nature figure"
- "publication plot"
- "scientific figure"
- "顶刊图"
- "nature风格"
- "出图"

## Requirements
- matplotlib
- scienceplots (`pip install SciencePlots`)
- numpy

## Default settings

```python
import scienceplots
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(['science', 'nature', 'no-latex'])
```

## Output format
- Primary: SVG (矢量, 可编辑)
- Secondary: PNG at 300 DPI
- Figure width: 7.2 inches (双栏) / 3.5 inches (单栏)

## Color palette (Okabe-Ito, colorblind-safe)
- `#0072B2` (蓝)
- `#D55E00` (橙)
- `#009E73` (绿)
- `#CC79A7` (粉)
- `#F0E442` (黄)
- `#56B4E9` (天蓝)
- `#E69F00` (金)
- `#000000` (黑)

## Chart types supported
- 折线图 (trend/line)
- 分组柱状图 (grouped bar)
- 堆叠柱状图 (stacked bar)
- 误差线图 (errorbar)
- 热图 (heatmap)
- 散点图 (scatter)
- 多面板组合图 (GridSpec multi-panel)

## Multi-panel hierarchy
Panel A: overview → Panel B: deviation → Panel C: relationship

## Rules
1. Always use colorblind-safe palette
2. No gridlines unless specified
3. Tick marks point inward
4. Font: sans-serif (Arial/Helvetica equivalent)
5. Legend within axes frame if possible
6. Axis labels with units in parentheses
