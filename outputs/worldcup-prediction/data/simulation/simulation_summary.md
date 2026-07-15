# 2026 世界杯 Monte Carlo 仿真报告

## 模拟设置
- **模拟次数**: 10,000
- **预测模型**: LightGBM (Optuna 优化, CV LogLoss=0.855)
- **仿真时间**: 2026-07-09 19:23
- **当前阶段**: 四分之一决赛

## 当前赛程状态

### 已完成的比赛
- 小组赛: 48 场 ✓
- 32 强: 16 场 ✓
- 16 强: 8 场 ✓
- 8 强参赛队: Argentina, Belgium, England, France, Morocco, Norway, Spain

### 剩余比赛
- 四分之一决赛 (4 场)
- 半决赛 (2 场)
- 决赛 (1 场)

## 夺冠概率 Top 10
| 排名 | 球队 | 夺冠概率 |
|------|------|---------|
| 1 | France | 28.19% |
| 2 | Argentina | 20.76% |
| 3 | Spain | 18.24% |
| 4 | England | 14.07% |
| 5 | Morocco | 6.86% |
| 6 | Belgium | 6.68% |
| 7 | Norway | 2.69% |
| 8 | Switzerland | 2.51% |
| 9 | Brazil | 0.00% |
| 10 | Canada | 0.00% |

## 四强概率 Top 10
| 排名 | 球队 | 四强概率 |
|------|------|---------|
| 1 | Argentina | 72.60% |
| 2 | France | 72.22% |
| 3 | England | 66.29% |
| 4 | Spain | 60.92% |
| 5 | Belgium | 39.08% |
| 6 | Norway | 33.71% |
| 7 | Morocco | 27.78% |
| 8 | Switzerland | 27.40% |
| 9 | Brazil | 0.00% |
| 10 | Canada | 0.00% |

## 四分之一决赛预测

| 对阵 | 主胜 | 平局 | 客胜 |
|------|------|------|------|
| France vs Morocco | 61.3% | 22.4% | 16.3% |
| Norway vs England | 22.5% | 21.8% | 55.7% |
| Spain vs Belgium | 45.5% | 29.8% | 24.7% |
| Argentina vs Switzerland | 61.8% | 20.3% | 17.8% |

## 半决赛对阵概率

| 半区 | 对阵 | 概率 |
|------|------|------|
| 上区 | England vs France | 48.4% |
| 下区 | Argentina vs Spain | 44.3% |
| 下区 | Argentina vs Belgium | 28.2% |
| 上区 | France vs Norway | 24.6% |
| 上区 | England vs Morocco | 17.8% |
| 下区 | Spain vs Switzerland | 16.6% |
| 下区 | Belgium vs Switzerland | 10.9% |
| 上区 | Morocco vs Norway | 9.1% |

## 四强组合频率 Top 5
| 排名 | 组合 | 概率 |
|------|------|------|
| 1 | Argentina, England, France, Spain | 21.11% |
| 2 | Argentina, Belgium, England, France | 13.56% |
| 3 | Argentina, France, Norway, Spain | 10.92% |
| 4 | Argentina, England, Morocco, Spain | 8.23% |
| 5 | England, France, Spain, Switzerland | 7.92% |

## 预测冠军
**France** 是最大夺冠热门 (概率 28.19%)。

**Argentina** 是第二热门 (概率 20.76%)。

## 方法论说明

1. **模型**: 使用历史数据训练的 LightGBM 模型，包含 Elo 评分、FIFA 排名、近期状态、世界杯历史表现等 72 维特征
2. **预测**: 对每场未进行的比赛预测 主胜/平/客胜 概率
3. **模拟**: 使用 Monte Carlo 方法，每次模拟从当前实际赛程出发
4. **淘汰赛**: 平局后按概率偏向强队进行点球模拟
5. **随机种子**: 42 (保证可复现)

## 输出文件
- `champion_probability.csv` — 各队夺冠概率
- `semifinal_probability.csv` — 各队四强概率
- `champion_probability_histogram.png` — 夺冠概率直方图
- `semifinal_probability_histogram.png` — 四强概率直方图
- `tournament_bracket.png` — 淘汰赛预测对阵图
