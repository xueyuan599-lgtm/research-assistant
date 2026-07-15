# 方案C — 冠军追齐冲刺

## 这是啥
在方案A（GBDT集成，Brier=0.1640）基础上，根据 2026 冠军 Harrison Horan 的方案，追 Kaggle Top 100。

## 目标
Brier 0.1640 → **0.130-0.140**

## 新对话启动时需要干的事

### Step 0：下载外部数据
- [Formula Bot KenPom CSV](https://www.formulabot.com/datasets/kenpom-march-madness)（2000-2026，免费）
- 放到 `data/kenpom_historical.csv`

### Step 1：运行 solution_c.py
- 实现冠军 3 大特征（seed_diff + harry_rating + opp_qlty_pts_won）
- XGBoost Regressor 主模型
- Leave One Season Out CV
- Isotonic 校准 + 极端概率后处理
- CB + LGB 辅助模型集成

### Step 2：提交
- submission_c.csv → Kaggle

## 关键参考
- 方案详情：`方案C_冠军追齐路线.md`
- 方案A代码参考：`../solution_a.py`
- 方案A结果：M Brier=0.1796 / W Brier=0.1412 / LB=0.1640
