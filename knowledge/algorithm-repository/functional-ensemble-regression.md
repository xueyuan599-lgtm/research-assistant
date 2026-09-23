---
title: 函数型数据集成回归：残差校正混合（AHFE）
type:
  - functional
  - regression
  - ensemble-learning
domain:
  - generic-ml
---
# 函数型数据集成回归：残差校正混合（AHFE）

- **来源**: 受 Alin et al. (2026, *Chemometrics and Intelligent Laboratory Systems* 269) Ensemble robust SIMPLS 启发的方法研究（本仓库产出）；对照竞争方法 Chamroukhi et al. (2024) Functional Mixtures-of-Experts
- **DOI**: 基座 10.1016/j.chemolab.2025.105626（预印本 SSRN 5593333）
- **方法类别**: 函数型数据分析 / 集成学习

## 数学设定

### 问题：scalar-on-function 回归（SoFR）

$$
Y_i = \alpha + \int_{\mathcal{T}} X_i(t)\,\beta(t)\,dt + \varepsilon_i
$$

$X_i(t)$ 为函数预测子（离散网格观测），$Y_i$ 为标量响应。PLS 类方法（SIMPLS，de Jong 1993）把 $X$ 投影到线性潜空间估计 $\beta(t)$。当 $Y$ 对 $X$ 呈**非线性功能效应**（如中心化二次交互 $\lambda(q^2-\mathbb{E}q^2)$，$q=\int X\varphi\,dt$）时，线性潜空间系统性欠拟合。

### AHFE：残差校正混合

$$
\hat y_{AHFE}(x) = \underbrace{\hat y_A(x)}_{\text{线性稳健 SIMPLS}} \;+\; w(x)\,\underbrace{\hat r_B(x)}_{\text{残差 GBDT}}
$$

- **组件 A**：稳健加权 SIMPLS（MAD 加权 IRLS，Tukey bisquare）→ 光滑可解释 $\hat\beta(t)$，稳健抗污染，低方差高偏差。
- **组件 B**：在残差 $r = y - \hat y_A$ 上训练标准 GBDT（XGBoost/LightGBM/CatBoost），特征为函数型导出特征（FPCA 评分 + B-spline 系数 + 一阶导数统计）→ 捕获线性潜空间遗漏的非线性结构，低偏差高方差。
- **门控**：$w(x)=\sigma(\alpha+z^\top\theta)\in[0,1]$，控制**非线性修正强度**（非多专家概率混合）。

### 门控训练（cross-fitting，防泄漏）

外层 CV 训练集中做内层 K 折，生成真正 OOF（$\hat y_A^{OOF},\hat r_B^{OOF},z^{OOF}$），在 OOF 上最小化 $\sum_i(y_i-\hat y_{A,i}-w(z_i)\hat r_{B,i})^2+\lambda_g\|\theta\|^2$。门控特征 $z=[\text{FPCA scores},|\hat r_B|,Q\text{-residual}, \text{leverage}]$——**不含真实残差**（预测时 y 未观测）。A、B 用整个外层训练集重训，一次性预测测试集；门控不接触测试 y。

### Oracle gate（理论锚）

$$
w^*(z) = \Pi_{[0,1]}\frac{\mathbb{E}[e_A q|z]}{\mathbb{E}[q^2|z]}, \qquad e_A=Y-\hat y_A,\; q=\hat r_B
$$

AHFE-fixed 是特例：$\gamma^*=\Pi_{[0,1]}\mathbb{E}[e_A q]/\mathbb{E}[q^2]$。

### 与 FME 的区别

Functional Mixtures-of-Experts（FME/iFME）用函数型门控网络在**多个完整专家**间做概率混合 $\hat y=\sum_k\pi_k\hat y_k$。AHFE 只做一个线性稳健基座 + 一个残差校正器，$w$ 调制**残差校正强度**而非专家权重。

## 关键假设

- PLS 线性潜空间假设及其失效条件（非线性/交互功能效应）。
- 门控识别性：w 在 OOF 上训练，不接触外层测试 y；门控特征不含 y 相关量。
- 稳健权重假设（MAD 尺度、Tukey bisquare）。
- **诚实边界（P4a/P5 实证）**：adaptive ≈ fixed（含可行 oracle）；门控未干净追踪逐样本非线性（Spearman≈0）。贡献在"混合整体相对线性基座的增益"，不宣称逐样本门控优于常数 γ。

## 适用场景

- **线性功能效应** → SIMPLS 足够，AHFE 无增益也不致损（门控 w 不强制开启）。
- **非线性/交互功能效应** → AHFE 相对 SIMPLS 基座提升稳健巨大（合成 ΔR²≈+0.39；Tecator +0.09）。
- **异常值污染** → 稳健 SIMPLS/AHFE 抗 Y-离群（S6b 函数型离群 ΔR²≈+0.14 保持优势；S6a 强 Y-污染无增益，如实）。
- **需 $\hat\beta(t)$ 解释** → 线性组件（corr≈0.94 恢复）。
- **外推** → 树组件失效需谨慎。

## 实现要点

- 成分数 A、门控 $\lambda_g$、内层折数、特征维（FPCA/B-spline）冻结；GBDT 固定 250 树/深度 4。
- cross-fitting 是门控防泄漏的关键，勿在测试折上调 w。
- 种子纪律：开发种子与确证种子分离（本库 {100–104}），确证基准不回头看性能改模型。
- 数值守卫：A≤min(n−1,p)、MAD 尺度下限、w∈[0,1]、确定性模式容差 1e-6~1e-8。
- SIMPLS 与 sklearn PLSRegression（NIPALS）计算路径不同，等价性用预测 RMSE<1e-6/corr>0.999 验证。

## 代码

自包含复现（仅 numpy/scipy/sklearn/xgboost）：

```python
import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import Matern
from xgboost import XGBRegressor

# ---------- 组件 A：SIMPLS / 稳健 SIMPLS ----------
def simpls(X, y, A):
    X = X - X.mean(0); yc = y - y.mean(); n, p = X.shape
    A = min(A, n - 1, p); W = T = P = Q = None
    W = np.zeros((p, A)); T = np.zeros((n, A)); P = np.zeros((p, A))
    Q = np.zeros(A); V = np.zeros((p, A)); S = X.T @ yc
    for a in range(A):
        S /= np.linalg.norm(S) + 1e-14
        w = S.copy()
        if a > 0:
            w = w - V[:, :a] @ (V[:, :a].T @ w); w /= np.linalg.norm(w) + 1e-12
        t = X @ w; tt = t @ t
        c = X.T @ t / tt; Q[a] = t @ yc / tt; S = S - c * (c @ S)
        W[:, a] = w; T[:, a] = t; P[:, a] = c
        v = c - V[:, :a] @ (V[:, :a].T @ c); V[:, a] = v / (np.linalg.norm(v) + 1e-12)
    beta = W @ np.linalg.solve(P.T @ W, Q)
    return beta, float(y.mean() - X.mean(0) @ beta)

def rwsimpls(X, y, A, iters=8, c=4.685):
    n = len(y); w = np.ones(n)
    for _ in range(iters):
        beta, b0 = simpls(X * w[:, None], y * w, A)  # 简化加权版
        r = y - (X @ beta + b0); s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if s < 1e-10: break
        u = (r - np.median(r)) / s
        w = np.where(np.abs(u) <= c, (1 - (u / c) ** 2) ** 2, 0.0)
    return beta, b0

# ---------- 函数型特征（AHFE-B 与 M2 同库）----------
class FunFeat:
    def fit(self, X): self.pca = PCA(6).fit(X); return self
    def transform(self, X):
        F = self.pca.transform(X)
        dX = np.gradient(X, axis=1); d = np.sqrt((dX ** 2).mean(1))[:, None]
        return np.hstack([F, d])   # FPCA 评分 + 导数幅值（可加 B-spline 系数）

# ---------- 门控（sigmoid-link）----------
class Gate:
    def fit(self, z, y, ya, rb, lam=1e-4):
        self.zm, self.zs = z.mean(0), z.std(0); self.zs[self.zs < 1e-12] = 1
        zs = (z - self.zm) / self.zs
        def loss(t):
            w = 1 / (1 + np.exp(-np.clip(t[0] + zs @ t[1:], -30, 30)))
            return float(np.sum((y - ya - w * rb) ** 2) + lam * np.sum(t[1:] ** 2))
        self.th = minimize(loss, np.zeros(zs.shape[1] + 1), method="L-BFGS-B").x
        return self
    def predict(self, z):
        zs = (z - self.zm) / self.zs
        return 1 / (1 + np.exp(-np.clip(self.th[0] + zs @ self.th[1:], -30, 30)))

# ---------- AHFE 装配 ----------
def fit_ahfe(Xtr, ytr, Xte, mode="adaptive", A=4, inner=4, seed=0):
    X = np.asarray(Xtr, float); y = np.asarray(ytr, float)
    kf = KFold(inner, shuffle=True, random_state=seed)
    Zo, Yo, Ya, Rb = [], [], [], []
    for tr, va in kf.split(X):                      # 内层 cross-fitting 生成 OOF
        beta, b0 = rwsimpls(X[tr], y[tr], A)
        ya = X[va] @ beta + b0
        fe = FunFeat().fit(X[tr]); F = fe.transform(X[va])
        rb = XGBRegressor(n_estimators=250, max_depth=4, random_state=seed).fit(
            fe.transform(X[tr]), y[tr] - (X[tr] @ beta + b0)).predict(F)
        Zo.append(np.hstack([fe.transform(X[va]), np.abs(rb)[:, None]]))
        Yo.append(y[va]); Ya.append(ya); Rb.append(rb)
    Zo, Yo, Ya, Rb = map(np.concatenate, (Zo, Yo, Ya, Rb))
    if mode == "adaptive":
        gate = Gate().fit(Zo, Yo, Ya, Rb)
    else:
        g = float((Yo - Ya) @ Rb / (Rb @ Rb + 1e-12)); gate = None
    beta, b0 = rwsimpls(X, y, A)                    # 全训练集重训 A、B
    fe = FunFeat().fit(X)
    F = fe.transform(X)
    b_model = XGBRegressor(n_estimators=250, max_depth=4, random_state=seed).fit(F, y - (X @ beta + b0))
    Xt = np.asarray(Xte, float)
    ya_te = Xt @ beta + b0; Ft = fe.transform(Xt); rb_te = b_model.predict(Ft)
    if mode == "adaptive":
        zt = np.hstack([Ft, np.abs(rb_te)[:, None]])
        w = gate.predict(zt)
    else:
        w = np.full(len(Xt), g)
    return ya_te + w * rb_te, w, beta            # (预测, 门控权重, 线性β)

# 用法：
#   from sklearn.model_selection import train_test_split
#   X, y = load_data()          # (n, p) 曲线矩阵 + 标量响应
#   pred, w, beta = fit_ahfe(Xtr, ytr, Xte, mode="adaptive")
```

> 生产建议：数据加载用 scikit-fda（`fetch_tecator`）；FPCA/B-spline 用 `skfda.misc`；实现细节见研究包 `research-assistant/outputs/functional_ensemble_regression/`（benchmark.py / tests / figures）。

## 基准结果摘要（确证种子 {100–104}，5 折）

| 场景 | M3 基座 | AHFE-fixed | AHFE-adaptive |
|------|--------|-----------|---------------|
| S1 Tecator 脂肪 | 0.882 | **0.974** | 0.970 |
| S5 合成非线性 | 0.424 | **0.823** | 0.815 |
| S5b 线性（阳性对照）| 0.894 | 0.893 | 0.891 |
| S6b 函数型离群 | 0.833 | **0.971** | 0.970 |

- 非线性 S5：ΔRMSE=0.162，**95% CI [0.119, 0.209] 排除 0**（paired bootstrap）。
- β̂ 恢复：corr(β̂, β)=0.937。
- **诚实限制**：adaptive ≈ fixed；Spearman(w,NL)=+0.036 不显著，NL 分位数→平均 w 非单调 → 门控不追踪逐样本非线性，定位为灵活修正装置而非逐样本增益来源。

## 参考文献

1. Alin, A., et al. (2026). Ensemble robust SIMPLS with block-penalized smoothing for scalar-on-function regression. *Chemometrics and Intelligent Laboratory Systems*, 269. doi:10.1016/j.chemolab.2025.105626（SSRN 预印本 5593333）
2. Chamroukhi, F., et al. (2024). Functional mixtures-of-experts for scalar-on-function regression. *Statistics and Computing*, 34(3):98（竞争方法 FME/iFME）
3. de Jong, S. (1993). SIMPLS: An alternative approach to partial least squares regression. *Chemometrics and Intelligent Laboratory Systems*, 18(3):251–263.
4. Hubert, M., & Vanden Branden, K. (2003). Robust methods for partial least squares regression. *Journal of Chemometrics*, 17(10):537–549.
5. Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2):89–121.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
7. Ramsay, J. O., & Silverman, B. W. (2005). *Functional Data Analysis* (2nd ed.). Springer.
8. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*, 3(1):79–87（oracle 门控理论锚）。
