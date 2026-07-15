# 方案B实验日志

运行时间: 2026-07-07 19:43

## CV Brier分数

### M
| 视角 | Brier |
|------|-------|
| 贝叶斯 | 0.2485 |
| 神经嵌入 | 0.2337 |
| 评分系统 | 0.2134 |
| 融合 | 0.2134 |

### W
| 视角 | Brier |
|------|-------|
| 贝叶斯 | 0.2482 |
| 神经嵌入 | 0.2302 |
| 评分系统 | 0.2102 |
| 融合 | 0.2095 |

## 融合权重

### M
```
Bayesian:        0.0000
Neural Embed:    0.0000
Rating Systems:  1.0000
(non-negative weights, sum = 1)
```

### W
```
Bayesian:        0.0462
Neural Embed:    0.1476
Rating Systems:  0.8061
(non-negative weights, sum = 1)
```

## 运行时间

总计: 2484s (41.4min)

## 2026预测统计

### M
| Stat | Value |
|------|-------|
| min | 0.0681 |
| max | 0.9267 |
| mean | 0.4904 |
| std | 0.2305 |

### W
| Stat | Value |
|------|-------|
| min | 0.0410 |
| max | 0.9595 |
| mean | 0.5022 |
| std | 0.2555 |

