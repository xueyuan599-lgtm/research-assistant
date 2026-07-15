# Titanic 竞赛复盘

## 成绩
- **最佳 LB: 0.77500**（V1, CatBoost+TargetEnc+25特征, CV=0.853）
- V3 极简方案: LB 0.77272（CatBoost Top2 Hard Voting, 11特征, CV=0.834）

## 流程
1. 数据探索: 12列891行, Age缺失19.9%, Cabin缺失77.1%, 存活率38.4%
2. 特征工程: Title/Sex/Age(分组中位数)/Fare(log1p)/Embarked(S)/FamilySize/IsAlone/HasCabin
3. 11特征 → Label/OneHot编码 → 7模型 × 5seed × 10fold CV
4. CatBoost(depth=4)单模型最优 CV=0.8325, Top2 Hard Voting CV=0.8339

## 核心教训

### 1. CV与LB有6%的gap
CV 0.83 → LB 0.77。891样本太小，10-fold CV天然偏乐观。不要用CV分数骗自己。

### 2. 特征多=过拟合
V1的54特征+Target Encoding: CV 0.853 → LB 0.775。V3的11特征: CV 0.834 → LB 0.773。
更低的CV反而更高的LB。小样本上简单>复杂。

### 3. Target Encoding是小样本毒药
用训练集Survived均值编码Title/Deck，把噪声模式写进了特征列。
CV里看起来有用（+0.02），测试集上直接失效。

### 4. Titanic天花板≈0.80
891+418样本，真实生死有随机性。LB>0.82基本不靠建模（用Wikipedia查真实名单等）。
0.77-0.80是诚实建模的正常区间。

### 5. 简单方案就是最优方案
logistic regression/RF + 10个基础特征 = 最诚实。
GBDT/CatBoost略好一线但差距不大。深度神经网络在这个规模没有价值。

## 如果重新做
- 特征不超过15个
- 不用Target Encoding
- Age用分组中位数，不用RF回归填补
- 不做交互特征
- 多seed CV验证稳定性
- 先跑logistic regression看baseline，再试RF/GBDT
- 不花时间在图表和报告上

## 文件结构
```
titanic/
├── train.csv / test.csv / gender_submission.csv   # 原始数据
├── competition_overview.json                       # 竞赛规范
├── v2/
│   ├── titanic_v3.py                               # 最终脚本(极简+泛化)
│   └── submission_v3_ensemble.csv                  # 最佳提交(CV=0.834)
└── README.md                                       # 本文件
```
