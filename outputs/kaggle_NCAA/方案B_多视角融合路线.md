# 方案 B：多视角融合 — 贝叶斯 + 神经嵌入 + 评分系统

> 目标：用三种完全不同的数学视角独立建模，融合得到更鲁棒的 NCAA 概率预测
> 评价指标：Brier Score
> 数据路径：`outputs/kaggle_NCAA/data/`
> 核心理念：三个视角的数学结构不同 → 预测相关性低 → 融合增益大

---

## 总览

```
┌─ 视角1: 贝叶斯层级模型 (PyMC)
│   统计推断视角
│   y_ij ~ Bernoulli(sigmoid(θ_i - θ_j))
│   输出: P_bayes(A beats B)
│
├─ 视角2: 神经嵌入模型 (PyTorch)
│   表征学习视角
│   Team → Embedding → MLP → sigmoid
│   输出: P_nn(A beats B)
│
├─ 视角3: 评分系统集成 (纯 numpy)
│   算法博弈视角
│   50+ 评分系统加权投票
│   输出: P_rating(A beats B)
│
└─ 融合层: 二次贝叶斯回归
    logit(P_final) = α·logit(P_bayes) + β·logit(P_nn) + γ·logit(P_rating)
    权重 α,β,γ 用历史 CV 的 OOF 拟合
```

---

## 视角 1：贝叶斯层级模型

### 数学设定

核心假设：每支球队 i 有一个潜在实力参数 θ_i，比赛结果由双方实力差决定。

**模型结构：**
```
对于比赛 k（A vs B）:
  y_k ~ Bernoulli(p_k)
  logit(p_k) = θ_A - θ_B + γ·home_k

先验分布:
  θ_i ~ Normal(μ_season, σ²_team)         # 球队实力，赛季级别收缩
  μ_season ~ Normal(0, 1)                  # 赛季均值
  σ²_team ~ HalfCauchy(2.5)                # 球队间方差
  γ ~ Normal(0, 0.5)                       # 主场优势
```

**时间衰退：** 近期比赛对实力估计贡献更大
```
权重: w_k = exp(-λ · Δt_k)
λ = 0.005（半衰期约 140 天）
Δt_k = 该场比赛距当前的天数
```

### 赛季间传递

球队实力在不同赛季间有关联，不是完全独立：
```
θ_i,t ~ Normal(ρ · θ_i,t-1, σ²_carry)
ρ = 0.85  # 赛季间相关系数
```

### 实现方式

**推荐方案 A（完整 MCMC）：** 用 PyMC 采样
```python
import pymc as pm

with pm.Model():
    # 球队实力
    mu = pm.Normal("mu", 0, 1)
    sigma = pm.HalfCauchy("sigma", 2.5)
    theta = pm.Normal("theta", mu, sigma, shape=n_teams)
    
    # 主场优势
    home = pm.Normal("home", 0, 0.5)
    
    # 线性预测
    logit_p = theta[team_A_ids] - theta[team_B_ids] + home * is_home
    p = pm.math.sigmoid(logit_p)
    
    # 观测
    y = pm.Bernoulli("y", p, observed=outcomes)
    
    # 采样（4 chains, 1000 warmup, 2000 draw）
    trace = pm.sample(2000, tune=1000, chains=4, cores=2)
```

**推荐方案 B（MAP 近似/快）：** 如果 MCMC 太慢，用 scipy 做 MAP
```python
from scipy.optimize import minimize

def neg_log_likelihood(params, team_A, team_B, y, home_indicators):
    theta = params[:n_teams]
    home = params[-1]
    logit_p = theta[team_A] - theta[team_B] + home * home_indicators
    p = 1 / (1 + np.exp(-logit_p))
    # 这个 + 正则化项（先验）
    ll = -np.mean(y * np.log(p + 1e-10) + (1-y) * np.log(1-p + 1e-10))
    # L2 正则化（对应 Normal 先验）
    reg = 0.5 * np.sum(theta**2) / (n_teams * 10)
    return ll + reg

result = minimize(neg_log_likelihood, x0=np.zeros(n_teams + 1), 
                  args=(team_A_idxs, team_B_idxs, y, home_flags))
theta_map = result.x[:n_teams]
home_map = result.x[-1]
```

### 产出

- 每支球队的 θ_i 后验均值（或 MAP 估计）
- 每场比赛的预测概率：`p = sigmoid(θ_A - θ_B + γ·home)`

### 训练/预测

**训练：** 用全部历史比赛（1985-2025）估计 θ_i
**预测 2026：** 用 2026 赛季常规赛数据重新估计 θ_i（或使用赛季间传递从 2025 继承）
**对阵预测：** `P(A wins) = sigmoid(θ_A - θ_B)`

---

## 视角 2：神经嵌入模型

### 数学设定

每支球队用一个 d=32 维的嵌入向量表示。这个向量捕获了"球队风格 + 实力"的联合表征——类似 word2vec 但作用于球队。

### 模型架构

```
Team_A (整数ID) ─→ Embedding Layer (n_teams × 32) ─→ e_A
Team_B (整数ID) ─→ Embedding Layer (n_teams × 32) ─→ e_B

输入向量 = concat(e_A, e_B, e_A - e_B, e_A * e_B, |e_A - e_B|)
  维度 = 32 + 32 + 32 + 32 + 32 = 160
     │
     ▼
全连接层 1: 160 → 64, ReLU, Dropout(0.2)
全连接层 2: 64 → 32, ReLU, Dropout(0.2)
全连接层 3: 32 → 1, Sigmoid
     │
     ▼
输出: P(A beats B)
```

### 为什么嵌入有效

- **协同信号：** 如果 A 赢了 B，B 赢了 C，嵌入空间会让 A 和 C 在某种意义下接近
- **风格捕获：** 打法相似的球队（如都打快节奏/都偏防守）会在嵌入空间中聚类
- **隐性分组：** 强队自然聚集，弱队聚集，中等队按风格分散

### 实现

```python
import torch
import torch.nn as nn

class TeamEmbeddingModel(nn.Module):
    def __init__(self, n_teams, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(n_teams, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, team_a, team_b):
        e_a = self.embedding(team_a)      # (batch, 32)
        e_b = self.embedding(team_b)      # (batch, 32)
        x = torch.cat([
            e_a, e_b, e_a - e_b, 
            e_a * e_b, torch.abs(e_a - e_b)
        ], dim=1)                          # (batch, 160)
        return self.net(x).squeeze()
```

### 训练配置

```python
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

batch_size = 128
epochs = 50
early_stop_patience = 5
```

### 数据增广

每场比赛产生两个训练样本（与方案 A 相同）：
- (winner_id, loser_id) → target = 1
- (loser_id, winner_id) → target = 0

### 赛季处理

**两种策略（选一或混合）：**
1. **全局嵌入：** 所有赛季共享同一个 embedding table，好处是跨赛季信息传递，坏处是球队实力会变化
2. **赛季特定嵌入：** 每个赛季一个独立的 embedding matrix，训练时稀疏

**推荐：** 全局嵌入 + 引入 Season 作为侧输入特征（一个额外的 one-hot 或 learned embedding）

### 产出

- 训练好的嵌入矩阵 (n_teams × 32) — 可用于可视化 t-SNE
- 对任意对阵的预测概率

---

## 视角 3：评分系统集成

### 核心思想

不训练任何机器学习模型，而是构建 **多个有理论基础的评分系统**，每个系统给每支球队打一个分，然后用分数差预测胜负。

这模仿了 NCAA 委员会选择球队的方式——他们看几十种排名指标。

### 评分系统列表

#### 3.1 Elo 变体（4 个）

| 变体 | K | 主场优势 | 赛季重置 |
|------|---|---------|---------|
| Elo-A | 20 | 100 | 75% 继承 |
| Elo-B | 30 | 80 | 80% 继承 |
| Elo-C | 20 | 0 | 75% 继承 |
| Elo-D | 自适应 K（净胜分越大 K 越大） | 100 | 70% 继承 |

#### 3.2 Massey 方法

Massey 方法将评分问题转化为线性系统求解：

```
对每场比赛 (A vs B, A 赢 d 分):
  方程: r_A - r_B = d

写成矩阵形式: X·r = y
其中 X 是 (n_games × n_teams) 设计矩阵
每行: 在 A 列 = 1, B 列 = -1
y = 净胜分

约束: Σ r_i = 0（可识别性）
解: r = (X'X)⁻¹ X'y
```

```python
def massey_rating(games, n_teams):
    """
    games: DataFrame with columns (team_a, team_b, score_diff)
    """
    X = np.zeros((len(games), n_teams))
    y = np.zeros(len(games))
    
    for i, (_, row) in enumerate(games.iterrows()):
        X[i, row.team_a] = 1
        X[i, row.team_b] = -1
        y[i] = row.score_diff
    
    # 添加约束 Σ r = 0
    X = np.vstack([X, np.ones(n_teams)])
    y = np.append(y, 0)
    
    # 最小二乘解
    r, _, _, _ = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)
    return r
```

#### 3.3 Colley 方法

Colley 是 Massey 的贝叶斯版本，加入先验收缩：

```
r_i = (1 + w_i) / (2 + n_i)

其中 w_i = 球队 i 的胜场数
      n_i = 球队 i 的比赛场数
```

但实际上 Colley 也是解线性系统：
```
C·r = b
C_ii = 2 + n_i
C_ij = -n_ij（i 和 j 交手次数）
b_i = 1 + (w_i - l_i) / 2
```

```python
def colley_rating(games, n_teams):
    C = np.eye(n_teams) * 2  # 先验
    b = np.ones(n_teams)
    
    for team_a, team_b, winner in games[['team_a', 'team_b', 'winner']].values:
        if winner == team_a:
            b[team_a] += 0.5
            b[team_b] -= 0.5
        else:
            b[team_a] -= 0.5
            b[team_b] += 0.5
        C[team_a, team_b] -= 1
        C[team_b, team_a] -= 1
        C[team_a, team_a] += 1
        C[team_b, team_b] += 1
    
    return np.linalg.solve(C, b)
```

#### 3.4 PageRank 方法

构建有向图：A → B（A 赢了 B）。PageRank 在这个图上迭代：

```
PR(A) = (1-d) + d · Σ PR(B) / out_degree(B)
       B 输给了 A
```

直觉：赢过强队的球队排名高。

#### 3.5 效率评分

从 Detailed 数据计算，不依赖比赛结果：
```
Offensive Rating = 场均得分（对手强度修正）
Defensive Rating = 场均失分（对手强度修正）
Net Rating = ORtg - DRtg
```

对手强度修正：用递归方法（类似 RPI）
```
Team ORtg_adj = mean(Team ORtg - Opponent DRtg + League_avg_DRtg)
```

#### 3.6 四因素法

篮球 analytics 经典框架，四个决定比赛的因素：

| 因素 | 公式 | 权重 |
|------|------|------|
| Shooting (eFG%) | (FGM + 0.5*FGM3) / FGA | 40% |
| Turnovers | TO / Possessions | 25% |
| Rebounding | OR / (OR + DR_opp) | 20% |
| Free Throws | FTM / FTA | 15% |

综合评分 = Σ 各因素标准化得分 × 权重

#### 3.7 Massey Ordinals（仅男子）

直接从 `MMasseyOrdinals.csv` 提取排名，对每个系统取最新值：
```
用所有可用系统的排名均值作为评分
```

### 评分集成

```python
# 每个评分系统独立预测 P(A beats B)
# 通过 sigmoid(score_diff) 转换得分差为概率

def system_to_prob(score_a, score_b, scale=1.0):
    """将评分差转为概率"""
    diff = score_a - score_b
    return 1 / (1 + np.exp(-diff / scale))

# 构建所有系统的预测矩阵
# X_systems[i, j] = P_system_j(team_i's pair wins)

# 学习权重（简单逻辑回归）
from sklearn.linear_model import LogisticRegression
meta = LogisticRegression(C=1.0, penalty='l2')
meta.fit(X_systems_train, y_train)

# 权重解读：哪些评分系统最预测力强
```

**权重数量：** 约 10-15 个评分系统（不是越多越好，高性能系统会被高权重）

---

## 融合层

### 第一层：视角内融合

每个视角内部可能有多个子模型，先内部融合：

| 视角 | 内部子模型 | 融合方法 |
|------|-----------|---------|
| 贝叶斯 | 多个 MCMC chain | 后验均值 |
| 神经嵌入 | 5 个不同的随机种子 | 平均 |
| 评分系统 | 15 个评分系统 | Logistic 回归加权 |

### 第二层：跨视角融合

三个视角的输出概率相关性天然低（因为它们看到的问题结构不同）：

```
logit(P_bayes)、logit(P_nn)、logit(P_rating) 的典型相关性：
  corr(贝叶斯, 嵌入) ≈ 0.6-0.7
  corr(贝叶斯, 评分) ≈ 0.7-0.8
  corr(嵌入, 评分)   ≈ 0.5-0.6
  
→ 比 GBDT 方案中模型间相关性 (~0.95) 低得多
→ 融合增益更大
```

**融合模型：**
```python
# 用 OOF 预测训练二次贝叶斯回归
# 用 logit 转换使概率在 [-∞, +∞] 空间线性可加

logit_p_bayes = np.log(p_bayes / (1 - p_bayes + 1e-10))
logit_p_nn = np.log(p_nn / (1 - p_nn + 1e-10))
logit_p_rating = np.log(p_rating / (1 - p_rating + 1e-10))

X_fusion = np.column_stack([
    logit_p_bayes, logit_p_nn, logit_p_rating,
    # 交互项
    logit_p_bayes * logit_p_nn,
    logit_p_bayes * logit_p_rating,
    logit_p_nn * logit_p_rating,
])

# 正则化逻辑回归（防过拟合三个弱视角）
fusion_model = LogisticRegression(C=0.5, penalty='l2')
fusion_model.fit(X_fusion, y_true)

# 权重分析
# α = fusion_model.coef_[0, 0]  # 贝叶斯权重
# β = fusion_model.coef_[0, 1]  # 嵌入权重
# γ = fusion_model.coef_[0, 2]  # 评分权重
```

---

## 验证策略

与方案 A 相同的时间序列 5-fold：

```
Fold 1: train 1985-2000, val 2001-2005
Fold 2: train 1985-2005, val 2006-2010
...
Fold 5: train 1985-2020, val 2021-2025
```

**每个视角独立做 CV → 得到 OOF 预测 → 用 OOF 预测训练融合层**

### 陷阱说明

1. **贝叶斯模型：** 每次 MCMC 采样需要 ~10-30 分钟（取决于数据量），可以用 MAP 加速到 <1 分钟
2. **神经嵌入：** 在 200K 样本上训练约 5-15 分钟（CPU），如果太慢可只用 2010 年后的数据
3. **评分系统：** Massey/Colley 需要每次解线性系统 ~0.1 秒，如果包含所有赛季会稍慢

---

## 男子/女子处理

| 视角 | 男子 | 女子 |
|------|------|------|
| 贝叶斯 | 有 Massey Ordinals 作为额外先验 | 无 Ordinals，纯层级模型 |
| 嵌入 | 独立训练男/女嵌入 | 独立训练 |
| 评分 | 含 Massey Ordinals 特征 | 无 Ordinals，用其他评分系统 |

两个性别的数据完全独立建模，最后合并提交文件。

---

## 2026 预测流程

```python
for gender in ['M', 'W']:
    # 1. 用历史数据训练三个视角
    bayes_model = train_bayesian(gender)
    nn_model = train_neural_embedding(gender)
    rating_systems = train_rating_systems(gender)
    
    # 2. 5-fold CV 获取 OOF 预测
    bayes_oof, nn_oof, rating_oof = cross_val_predict(gender)
    
    # 3. 训练融合层
    fusion = train_fusion(bayes_oof, nn_oof, rating_oof, y_true)
    
    # 4. 2026 全量重训练（所有数据，不分割）
    bayes_final = retrain_bayesian_full(gender)
    nn_final = retrain_nn_full(gender)
    rating_final = retrain_rating_full(gender)
    
    # 5. 预测 2026 所有对阵
    teams = get_2026_teams(gender)
    for (a, b) in all_pairs(teams):
        p_b = bayes_final.predict(a, b)
        p_n = nn_final.predict(a, b)
        p_r = rating_final.predict(a, b)
        p_final = fusion.predict(p_b, p_n, p_r)
        sub.append({"ID": f"2026_{a}_{b}", "Pred": p_final})

combine('M', 'W') → submission_b.csv
```

---

## 输出

### 提交文件：`submission_b.csv`

与方案 A 相同格式。

### 实验日志：`solution_b_log.md`

包含：
- 三个视角各自的 CV Brier
- 融合后的 CV Brier
- 融合权重 α, β, γ
- 贝叶斯模型的后验分布可视化（球队实力排名）
- t-SNE 嵌入空间可视化（球队聚类）
- 评分系统权重排名（哪个评分系统最有效）

### 额外产出（方案创新点）

| 产出 | 格式 | 说明 |
|------|------|------|
| 球队实力排名 | markdown 表格 | 贝叶斯 θ_i 排序，含 90% 置信区间 |
| 嵌入空间 | t-SNE 散点图 | 球队按风格/实力聚类 |
| 评分系统对比 | 柱状图 | 15 个评分系统的预测力对比 |

---

## 预期性能

| 指标 | 预期值 |
|------|--------|
| 贝叶斯视角 CV Brier | ~0.12-0.14 |
| 嵌入视角 CV Brier | ~0.12-0.14 |
| 评分系统 CV Brier | ~0.12-0.13 |
| 融合后 CV Brier | **~0.105-0.115** |
| 运行时间 | 45-120 分钟 |
| 内存需求 | ~6GB |
| Python 依赖 | pandas, numpy, sklearn, scipy, torch, pymc（选装） |

**融合增益预期：** +0.010-0.020 Brier（比方案 A 更好，因为视角更多样）

---

## 代码骨架

```python
# ============================================================
# solution_b.py — 多视角融合：贝叶斯 + 神经嵌入 + 评分系统
# ============================================================

# ---------- PERSPECTIVE 1: BAYESIAN ----------
class BayesianStrengthModel:
    """基于球队潜在实力的贝叶斯层级模型"""
    
    def __init__(self, use_mcmc=False):
        self.use_mcmc = use_mcmc
        self.theta = None  # n_teams 实力向量
        self.home = 0.0
    
    def fit(self, team_a_ids, team_b_ids, home_indicators, y):
        if self.use_mcmc:
            with pm.Model() as model:
                # 定义模型
                ...
                self.trace = pm.sample(...)
        else:
            # MAP 估计
            self.theta, self.home = self._map_estimate(...)
    
    def predict(self, team_a, team_b):
        p = 1 / (1 + np.exp(-(self.theta[team_a] - self.theta[team_b])))
        return p

# ---------- PERSPECTIVE 2: NEURAL EMBEDDING ----------
class NeuralEmbeddingModel:
    """PyTorch 球队嵌入模型"""
    
    def __init__(self, n_teams, embed_dim=32):
        self.model = TeamEmbeddingModel(n_teams, embed_dim)
        self._train_data = []
    
    def fit(self, team_a_ids, team_b_ids, y, n_epochs=50):
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(team_a_ids),
            torch.LongTensor(team_b_ids),
            torch.FloatTensor(y)
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(n_epochs):
            for a, b, label in loader:
                pred = self.model(a, b)
                loss = F.binary_cross_entropy(pred, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    def predict(self, team_a, team_b):
        with torch.no_grad():
            a = torch.LongTensor([team_a])
            b = torch.LongTensor([team_b])
            return self.model(a, b).item()
    
    def get_embeddings(self):
        """返回 embedding 矩阵，可用于 t-SNE 可视化"""
        return self.model.embedding.weight.detach().numpy()

# ---------- PERSPECTIVE 3: RATING SYSTEMS ----------
class RatingSystemEnsemble:
    """多个评分系统 + 加权融合"""
    
    def __init__(self):
        self.systems = {}  # name → rating function
        self.weights = {}  # name → weight
    
    def _register_systems(self):
        self.systems['elo_A'] = compute_elo(K=20)
        self.systems['elo_B'] = compute_elo(K=30)
        self.systems['massey'] = massey_rating
        self.systems['colley'] = colley_rating
        self.systems['pagerank'] = pagerank_rating
        # ... 更多系统
    
    def fit_weights(self, team_a, team_b, y):
        """用逻辑回归学习各系统权重"""
        # 对每个系统，计算所有比赛的概率预测
        # 用这些概率做特征，训练 LogisticRegression
        pass
    
    def predict(self, team_a, team_b):
        # 加权平均各系统的预测
        pass

# ---------- FUSION ----------
class FusionModel:
    """融合三个视角"""
    
    def fit(self, bayes_oof, nn_oof, rating_oof, y):
        # logit 转换
        # 训练带交互项的 LogisticRegression
        pass
    
    def predict(self, p_b, p_n, p_r):
        logit_p_b = np.log(p_b / (1-p_b))
        # ...
        return 1 / (1 + np.exp(-fusion_pred))

# ---------- MAIN ----------
def main():
    gender_results = {}
    for gender in ['M', 'W']:
        # 1. 数据加载（同方案 A）
        reg, tourney, teams, seeds = load_data(gender)
        
        # 2. 三个视角模型
        bayes = BayesianStrengthModel(use_mcmc=False)
        nn = NeuralEmbeddingModel(n_teams=len(teams))
        ratings = RatingSystemEnsemble()
        
        # 3. CV 获取 OOF 预测
        cv_folds = time_series_cv(reg, tourney)
        bayes_oof, nn_oof, rating_oof, y = cv_predict_all(
            [bayes, nn, ratings], reg, tourney, cv_folds)
        
        # 4. 训练融合层
        fusion = FusionModel()
        fusion.fit(bayes_oof, nn_oof, rating_oof, y)
        
        # 5. 全量重训练 + 预测 2026
        p_b, p_n, p_r = retrain_and_predict_2026(
            gender, bayes, nn, ratings, teams)
        p_final = fusion.predict(p_b, p_n, p_r)
        
        gender_results[gender] = p_final
    
    # 6. 合并提交
    save_submission(gender_results, "submission_b.csv")

if __name__ == "__main__":
    main()
```

---

## 附件：嵌入可视化扩展（可选）

训练完成后，可以用 t-SNE 可视化球队嵌入空间：

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

embeddings = nn_model.get_embeddings()
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c='steelblue', alpha=0.6)
# 标注前 20 强队的队名
for i, (x, y) in enumerate(emb_2d[:20]):
    plt.annotate(team_names[i], (x, y))
plt.title("NCAA Team Embedding Space (t-SNE)")
plt.savefig("outputs/kaggle_NCAA/embedding_tsne.png")
```

强队靠近中心，同风格球队形成聚类，不同风格发散——理想的嵌入空间应该是这样。
