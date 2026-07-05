"""顶刊风格事件研究图 — DID Event Study"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

# ── 载入数据 ──
df = pd.read_csv("E:/wuyi/数学建模半自动/research-assistant/outputs/ftz-did/event_study_results.csv")

# ── 全局样式：Nature 风格 ──
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(6, 4))

# 数据整理
times = df["EventTime"].tolist()
coef = df["Coefficient"].tolist()
ci_low = df["CI_Lower"].tolist()
ci_high = df["CI_Upper"].tolist()
x = np.arange(len(times))

# ── 绘图 ──
ax.axhline(y=0, color="grey", linewidth=0.6, linestyle="--", zorder=1)

# 置信区间竖线
for i in range(len(x)):
    ax.plot([x[i], x[i]], [ci_low[i], ci_high[i]], color="black", linewidth=1.2, zorder=2)

# 点估计
ax.scatter(x, coef, color="steelblue", s=45, zorder=3, edgecolors="black", linewidths=0.5)

# 区分 pre / current / post 区间
ax.axvline(x=3.5, color="grey", linewidth=0.6, linestyle=":", alpha=0.5)

# x 轴标签
tick_labels = [t.replace("pre", "t-").replace("post", "t+").replace("current", "t") for t in times]
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=9)

# 标注
ax.set_ylabel("Estimated Treatment Effect", fontsize=10)
ax.set_xlabel("Event Time", fontsize=10)

# 图注
ax.text(0.02, 0.95, "Event Study (TWFE)", transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontstyle="italic")

plt.tight_layout()
plt.savefig("E:/wuyi/数学建模半自动/research-assistant/outputs/ftz-did/fig01_event_study_reproduced.png", dpi=300)
print("OK")
