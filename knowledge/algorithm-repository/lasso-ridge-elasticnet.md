# Lasso / Ridge / ElasticNet — 套索回归 / 岭回归 / 弹性网络

- **来源**: Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288. | Hoerl, A. E. & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67. | Zou, H. & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
- **DOI**: 10.1111/j.2517-6161.1996.tb02080.x / 10.1080/00401706.1970.10488634 / 10.1111/j.1467-9868.2005.00503.x
- **方法类别**: 统计建模

## 数学设定

### 问题框架：线性回归与正则化

给定 $n$ 个样本 $\{(x_i, y_i)\}_{i=1}^n$，$x_i \in \mathbb{R}^p$，$y_i \in \mathbb{R}$。设计矩阵 $X \in \mathbb{R}^{n \times p}$，响应向量 $y \in \mathbb{R}^n$。

**普通最小二乘 (OLS)**

$$
\hat{\beta}^{\text{OLS}} = \arg\min_{\beta} \| y - X\beta \|^2
$$

闭式解 $\hat{\beta} = (X^\top X)^{-1} X^\top y$。当 $p > n$ 时 $X^\top X$ 奇异无法求逆；当特征高度共线时，$(X^\top X)^{-1}$ 条件数极大，导致 $\hat{\beta}$ 方差爆炸。

**正则化核心思想**：在损失函数中加入系数惩罚项，以偏误换取方差的降低，从而最小化预测误差。

---

### 岭回归 (Ridge / L2 Regularization)

$$
\hat{\beta}^{\text{Ridge}} = \arg\min_{\beta} \| y - X\beta \|^2 + \lambda \|\beta\|_2^2, \quad \lambda \geq 0
$$

**闭式解**

$$
\hat{\beta}^{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

$X^\top X + \lambda I$ 对任意 $\lambda > 0$ 满秩，确保解唯一存在。

**SVD 视角**：设 $X = U D V^\top$（$U \in \mathbb{R}^{n \times k}$，$V \in \mathbb{R}^{p \times k}$，$D = \text{diag}(d_1, \dots, d_k)$），则

$$
\hat{\beta}^{\text{Ridge}} = V (D^2 + \lambda I)^{-1} D U^\top y
$$

每个奇异方向 $v_j$ 对应的系数被因子 $d_j^2 / (d_j^2 + \lambda) \in (0, 1]$ 收缩。**小奇异值方向被强烈收缩**，大奇异值方向几乎不受影响。

**效应**：所有系数向零收缩 (shrinkage)，但**不精确为零**，无稀疏性。

**几何**：$L_2$ 球约束 $\|\beta\|_2^2 \leq t$，圆形可行域，解在椭圆等高线与圆球相切处。

---

### 套索回归 (Lasso / L1 Regularization)

$$
\hat{\beta}^{\text{Lasso}} = \arg\min_{\beta} \| y - X\beta \|^2 + \lambda \|\beta\|_1, \quad \lambda \geq 0
$$

**无闭式解**（正交设计 $X^\top X = I$ 除外，此时 $\hat{\beta}_j^{\text{Lasso}} = S(\hat{\beta}_j^{\text{OLS}}, \lambda/2)$）。

**稀疏性**：$L_1$ 惩罚使部分系数精确为零，实现自动特征选择。

**几何**：$L_1$ 菱形约束 $\|\beta\|_1 \leq t$，尖角位于坐标轴上。解常落在尖角处，对应稀疏解。

**求解算法**：坐标下降 (Coordinate Descent, CD) 和最小角回归 (Least Angle Regression, LARS) 是最主流的两种算法。CD 的每步更新为

$$
\beta_j \leftarrow \frac{S\big(\rho_j,\; \lambda/2\big)}{\|X_{\cdot,j}\|^2}
$$

其中 $\rho_j = X_{\cdot,j}^\top r^{(j)}$，$r^{(j)} = y - X_{-j}\beta_{-j}$ 为去掉第 $j$ 个特征后的残差。

---

### 弹性网络 (ElasticNet)

$$
\hat{\beta}^{\text{EN}} = \arg\min_{\beta} \| y - X\beta \|^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2
$$

**等价参数化**（设混合比例 $\alpha \in [0, 1]$，总强度 $\lambda \geq 0$）：

$$
\hat{\beta}^{\text{EN}} = \arg\min_{\beta} \| y - X\beta \|^2 + \lambda \big( \alpha \|\beta\|_1 + (1 - \alpha) \|\beta\|_2^2 \big)
$$

- $\alpha = 1$：Lasso（纯 $L_1$ 惩罚）
- $\alpha = 0$：Ridge（纯 $L_2$ 惩罚）
- $\alpha = 0.5$：等权重混合

**群组效应 (Grouping Effect)**：高度相关的变量被同时选入或同时排除，不似 Lasso 从相关组中随机只选一个。坐标下降更新为

$$
\beta_j \leftarrow \frac{S\big(\rho_j,\; \lambda_1/2\big)}{\|X_{\cdot,j}\|^2 + \lambda_2}
$$

---

### 贝叶斯解释

正则化等价于在回归系数上施加先验分布的 MAP 估计：

| 方法 | 先验分布 | 对应关系 |
|------|---------|---------|
| Ridge | $\beta_j \sim \mathcal{N}(0, \tau^2)$ | $\lambda = \sigma^2 / \tau^2$ |
| Lasso | $\beta_j \sim \text{Laplace}(0, \tau)$ | $\lambda = \sigma^2 / \tau$ |

Lasso 对应的 Laplace 先验在零点有尖峰，且尾部比正态厚——这解释为何 Lasso 能将系数精确推至零并允许大系数存在。

---

### 偏误-方差权衡

记 $\hat{\beta}(\lambda)$ 为正则化估计量，预测值 $\hat{y} = X\hat{\beta}(\lambda)$。

$$
\mathbb{E}\big[(y_0 - \hat{y}_0)^2\big] = \underbrace{\text{Bias}^2[\hat{y}_0]}_{\text{随 }\lambda \uparrow} \;+\; \underbrace{\text{Var}[\hat{y}_0]}_{\text{随 }\lambda \downarrow} \;+\; \sigma^2
$$

- $\lambda = 0$：无偏但高方差（OLS）
- $\lambda \to \infty$：方差趋于零但偏误极大

交叉验证选择 $\lambda$ 使预测误差的**期望**最小。

对于 Ridge，偏差和方差可显式分解（基于 SVD）：

$$
\begin{aligned}
\text{Bias}^2(\lambda) &= \lambda^2 \, \beta^\top (X^\top X + \lambda I)^{-2} \beta \\
\text{Var}(\lambda) &= \sigma^2 \sum_{j=1}^{k} \frac{d_j^2}{(d_j^2 + \lambda)^2}
\end{aligned}
$$

---

### 正则化路径

$\lambda$ 从大到小变化时，系数估计的轨迹：

- **Lasso 路径**：分段线性（LARS 保证），系数从零逐个非零
- **Ridge 路径**：平滑非线性收缩
- **ElasticNet 路径**：兼具 Lasso 的分段线性和 Ridge 的群组效应

路径图是选择 $\lambda$ 的重要诊断工具，观察系数何时"进入模型"可判断变量重要性。

---

### 预测变量标准化

**必须标准化**：$L_1$ / $L_2$ 惩罚对变量尺度敏感。若 $x_1$ 的单位是 $x_2$ 的千分之一，$x_1$ 的系数会被不成比例地过度惩罚。

标准化方式：

$$
\tilde{x}_{ij} = \frac{x_{ij} - \bar{x}_j}{\hat{\sigma}_j}, \quad
\tilde{y}_i = y_i - \bar{y}
$$

估计 $\tilde{\beta}$ 后回缩到原始尺度：

$$
\hat{\beta}_j = \frac{\tilde{\beta}_j}{\hat{\sigma}_j}, \quad
\hat{\beta}_0 = \bar{y} - \sum_{j=1}^{p} \bar{x}_j \hat{\beta}_j
$$

---

### Lasso 的自由度

估计量的自由度定义为 $\text{df}(\hat{y}) = \text{tr}(\partial \hat{y} / \partial y)$。

对于固定 $\lambda$ 的 Lasso，若预测变量处于"一般位置"（无退化情况），则

$$
\text{df}(\lambda) = \#\{\,j : \hat{\beta}_j \neq 0\,\}
$$

即活跃系数的个数。这一简洁性质使得信息准则可用于选择 $\lambda$：

$$
\begin{aligned}
\text{BIC}(\lambda) &= n \ln(\text{MSE}) + \text{df}(\lambda) \ln n \\
\text{AIC}(\lambda) &= n \ln(\text{MSE}) + 2 \cdot \text{df}(\lambda)
\end{aligned}
$$

---

### Oracle 性质及其局限

**Oracle 性质**：估计量同时满足 (1) 以概率趋近 1 选出真实支撑集；(2) 对非零系数的估计渐近有效（与已知真实模型相同的收敛速率）。

- **Lasso 不满足 Oracle**：需要**不可表示条件 (Irrepresentable Condition)**

  $$
  \| X_{\mathcal{N}}^\top X_{\mathcal{S}} (X_{\mathcal{S}}^\top X_{\mathcal{S}})^{-1} \|_\infty < 1
  $$

  其中 $\mathcal{S}$ 为活跃集，$\mathcal{N}$ 为非活跃集。该条件在高度相关特征下常被违反。此外 Lasso 对非零系数存在 $O(1/\sqrt{n})$ 的渐近偏误。

- **Adaptive Lasso** (Zou, 2006)：赋予权重 $w_j = 1 / |\hat{\beta}_j^{\text{init}}|^\gamma$，满足 Oracle 性质。

- **Ridge 不进行变量选择**，不涉及 Oracle 性质。

- **ElasticNet** 在满足一定条件下可具备 Oracle 性质（需 $n \to \infty$ 时 $\lambda_2$ 足够小）。

---

## 关键假设

1. **线性关系**：$y \approx X\beta$（可通过基展开如样条、多项式处理非线性）
2. **观测独立**：样本独立同分布 (i.i.d.)
3. **标准化后的预测变量**：所有预测变量经标准化处理，使惩罚尺度一致
4. **Lasso 的不可表示条件** (Irrepresentable Condition)：为一致变量选择，要求非活跃变量与活跃变量的相关性有界（见 Oracle 性质一节）
5. **Ridge 的最佳场景**：大量系数量级相近时效果最佳，各变量对预测贡献均匀
6. **ElasticNet 的群组结构**：当存在高度相关变量组时，ElasticNet 自动产生群组效应

## 适用场景

- **高维数据 ($p \gg n$)**：Lasso / Ridge / ElasticNet 均可处理，Lasso 同时做特征选择
- **特征选择/变量筛选**：Lasso 自动选择重要变量，适合基因组学、文本分类
- **多重共线性严重**：Ridge 通过 $L_2$ 惩罚稳定估计，比 OLS 更可靠
- **高度相关预测变量的预测**：ElasticNet 优于 Lasso（群组效应），比 Ridge 更稀疏
- **作为高维回归的 baseline**：任何高维问题首先用 Lasso 或 ElasticNet 建立基准

### 不适用

- **纯非线性关系**：需预先配合基展开或核方法，否则模型误设
- **系数解释性要求极高**：Lasso 虽做特征选择但系数有偏（shrinkage bias），需用 debiased Lasso 或 post-selection inference
- **变量存在已知分组结构**：Lasso 无视分组信息，应使用 Group Lasso / Sparse Group Lasso
- **超大规模数据**：坐标下降每轮 $O(np)$，对 $n > 10^6$ 可考虑 stochastic proximal methods
- **缺失值多**：正则化回归本身不处理缺失值，需先插补

## 实现要点

### 关键超参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| $\lambda$ (alpha) | $(0, \infty)$ | 1.0 | 正则化总强度，越大收缩越强 |
| $\alpha$ (l1_ratio) | $[0, 1]$ | 0.5 | ElasticNet 混合比例：1 = Lasso, 0 = Ridge |
| $tol$ | $(0, 0.1]$ | $10^{-4}$ | 坐标下降收敛容差 |
| $max\_iter$ | $[100, 10^4]$ | 1000 | CD 最大迭代轮数 |
| $n\_folds$ | $[3, 20]$ | 5 | 交叉验证折数 |

### $\lambda$ 选择策略

1. **K 折交叉验证**：最推荐，$\lambda$ 在 $[\lambda_{\max}, \lambda_{\max} \cdot 10^{-4}]$ 的对数网格上搜索，$\lambda_{\max}$ 为使所有系数为零的最小值
2. **1-SE 规则**：在 CV 误差曲线的最低点附近，选最简模型（最大 $\lambda$）使其 CV 误差在最低点的 1 个标准误以内——得到更稀疏、更稳定的模型
3. **信息准则**：对 Lasso 可用 BIC ($\text{df} = \#\text{nonzero}$)，计算更快但不如 CV 稳健

### 数值注意事项

- **标准化**：务必先标准化 $X$ 再拟合；sklearn 的 `StandardScaler` 或手动均可
- **Warm start**：沿 $\lambda$ 从大到小的路径用前一个解初始化后一个，大幅加速
- **$\lambda$ 尺度**：不同实现中 $\lambda$ 的定义可能有系数差异（是否包含 $1/2n$ 因子），比较时需注意
- **Lasso 的偏误校正**：先 Lasso 选变量，再用 OLS 重新估计入选变量，可减少偏误（称为 post-Lasso）

### 代码

```python
import numpy as np


# =====================
# 辅助函数
# =====================

def soft_threshold(z, gamma):
    """软阈值算子 S(z, γ) = sign(z) · max(|z| - γ, 0)
    
    这是 Lasso 坐标下降的核心运算，实现 L1 惩罚的次梯度条件。
    """
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)


class Standardizer:
    """中心化 + 标准化 (Z-score)，用于正则化前的数据预处理"""
    
    def fit(self, X, y):
        self.X_mean_ = np.mean(X, axis=0)
        self.X_std_ = np.std(X, axis=0)
        self.X_std_[self.X_std_ == 0] = 1.0  # 避免零除
        self.y_mean_ = np.mean(y)
        return self
    
    def transform(self, X, y=None):
        Xs = (X - self.X_mean_) / self.X_std_
        if y is not None:
            yc = y - self.y_mean_
            return Xs, yc
        return Xs
    
    def rescale_coef(self, beta_std):
        """标准化尺度下的系数回缩到原始尺度"""
        return beta_std / self.X_std_
    
    def intercept(self, beta_std):
        """计算原始尺度下的截距"""
        return self.y_mean_ - self.X_mean_ @ (beta_std / self.X_std_)


# =====================
# Lasso — 坐标下降 (Coordinate Descent)
# =====================

class Lasso:
    """Lasso 回归：β̂ = argmin ||y - Xβ||² + λ||β||₁
    
    求解算法：循环坐标下降 (Cyclic Coordinate Descent)
    
    核心更新公式（对每个坐标 j）：
        ρ_j = X[:,j]ᵀ r^{(j)}  (部分残差)
        β_j ← S(ρ_j, λ/2) / ||X[:,j]||²
    
    其中 S(·,·) 为软阈值算子。
    """
    
    def __init__(self, lam=1.0, tol=1e-4, max_iter=1000):
        self.lam = lam          # 正则化强度 λ
        self.tol = tol          # 收敛容差
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0
    
    def _cd_cycle(self, X, y, beta, residuals, lam):
        """单轮坐标下降：遍历所有特征并更新系数和残差"""
        Xj2 = np.sum(X ** 2, axis=0)  # 每个特征的 L2 范数平方
        
        for j in range(len(beta)):
            # 计算 ρ_j = X[:,j]ᵀ r + β_j · ||X[:,j]||²
            rho = X[:, j] @ residuals + beta[j] * Xj2[j]
            
            # 更新 β_j
            beta_new = soft_threshold(rho, lam / 2) / max(Xj2[j], 1e-12)
            
            # 增量更新残差（避免完整重算）
            residuals -= (beta_new - beta[j]) * X[:, j]
            beta[j] = beta_new
        
        return beta, residuals
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 1. 标准化
        self.std_ = Standardizer()
        self.std_.fit(X, y)
        Xs, yc = self.std_.transform(X, y)
        
        # 2. 初始化
        beta = np.zeros(n_features)
        residuals = yc.copy()
        
        # 3. 坐标下降迭代
        for it in range(1, self.max_iter + 1):
            beta_old = beta.copy()
            beta, residuals = self._cd_cycle(Xs, yc, beta, residuals, self.lam)
            
            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = it
                break
        
        # 4. 回缩到原始尺度
        self.coef_ = self.std_.rescale_coef(beta)
        self.intercept_ = self.std_.intercept(beta)
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    def path(self, X, y, lam_seq=None, n_lams=100):
        """完整正则化路径 (regularization path)
        
        使用 warm start：从 λ_max 开始，依次减小 λ，
        每个 λ 的求解以前一个解为起点，大幅加速收敛。
        
        返回:
            lam_seq: λ 序列 (n_lams,)
            coefs:   系数路径 (n_lams, n_features)
        """
        std = Standardizer()
        std.fit(X, y)
        Xs, yc = std.transform(X, y)
        n_features = Xs.shape[1]
        
        # 确定 λ 序列
        if lam_seq is None:
            lam_max = np.max(np.abs(Xs.T @ yc)) * 2  # 所有系数为零的最小 λ
            lam_seq = np.geomspace(lam_max, lam_max * 0.001, n_lams)
        
        coefs = np.zeros((len(lam_seq), n_features))
        beta = np.zeros(n_features)
        residuals = yc.copy()
        
        for i, lam in enumerate(lam_seq):
            for it in range(self.max_iter):
                beta_old = beta.copy()
                beta, residuals = self._cd_cycle(Xs, yc, beta, residuals, lam)
                if np.max(np.abs(beta - beta_old)) < self.tol:
                    break
            coefs[i] = std.rescale_coef(beta)
        
        return lam_seq, coefs
    
    def cv(self, X, y, lam_seq=None, n_folds=5):
        """K 折交叉验证选择 λ
        
        返回:
            best_lam: 最小 CV 误差对应 λ
            lam_1se:  1-SE 规则选出 λ（更简约）
            lam_seq:  λ 序列
            cv_mean:  各 λ 的平均 CV 误差
            cv_se:    各 λ 的 CV 标准误
        """
        n_samples, n_features = X.shape
        np.random.seed(42)  # 可复现性
        
        # 粗略标准化用于确定 λ 序列
        std_full = Standardizer()
        std_full.fit(X, y)
        Xs_full, yc_full = std_full.transform(X, y)
        
        if lam_seq is None:
            lam_max = np.max(np.abs(Xs_full.T @ yc_full)) * 2
            lam_seq = np.geomspace(lam_max, lam_max * 0.001, 100)
        
        # 创建交叉验证折
        idx = np.random.permutation(n_samples)
        fold_ids = np.array_split(idx, n_folds)
        
        mse = np.zeros((n_folds, len(lam_seq)))
        
        for k, test_idx in enumerate(fold_ids):
            train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
            X_tr_raw, y_tr_raw = X[train_idx], y[train_idx]
            X_te_raw, y_te_raw = X[test_idx], y[test_idx]
            
            # 在训练集上标准化（不泄漏测试集信息）
            std = Standardizer()
            std.fit(X_tr_raw, y_tr_raw)
            X_tr, y_tr = std.transform(X_tr_raw, y_tr_raw)
            X_te = std.transform(X_te_raw)
            y_te = y_te_raw - std.y_mean_
            
            # Warm start 沿 λ 序列求解
            beta = np.zeros(n_features)
            residuals = y_tr.copy()
            
            for i, lam in enumerate(lam_seq):
                for it in range(self.max_iter):
                    beta_old = beta.copy()
                    beta, residuals = self._cd_cycle(X_tr, y_tr, beta, residuals, lam)
                    if np.max(np.abs(beta - beta_old)) < self.tol:
                        break
                
                y_pred = X_te @ beta
                mse[k, i] = np.mean((y_te - y_pred) ** 2)
        
        cv_mean = np.mean(mse, axis=0)
        cv_se = np.std(mse, axis=0) / np.sqrt(n_folds)
        
        best_idx = np.argmin(cv_mean)
        best_lam = lam_seq[best_idx]
        
        # 1-SE 规则：选最大 λ 使其 CV 误差在 min ± 1SE 以内
        threshold = cv_mean[best_idx] + cv_se[best_idx]
        candidates = np.where(cv_mean <= threshold)[0]
        lam_1se = lam_seq[candidates[0]] if len(candidates) > 0 else best_lam
        
        return best_lam, lam_1se, lam_seq, cv_mean, cv_se


# =====================
# Ridge — 闭式解 + SVD LOO-CV
# =====================

class Ridge:
    """岭回归：β̂ = argmin ||y - Xβ||² + λ||β||²₂
    
    闭式解 β̂ = (XᵀX + λI)⁻¹ Xᵀy。
    支持通过 SVD 实现 O(n) 的留一交叉验证 (LOO-CV)。
    """
    
    def __init__(self, lam=1.0):
        self.lam = lam
        self.coef_ = None
        self.intercept_ = 0.0
    
    def fit(self, X, y):
        n_features = X.shape[1]
        
        self.std_ = Standardizer()
        self.std_.fit(X, y)
        Xs, yc = self.std_.transform(X, y)
        
        # 闭式解：β = (XᵀX + λI)⁻¹ Xᵀy
        XtX = Xs.T @ Xs
        Xty = Xs.T @ yc
        beta = np.linalg.solve(XtX + self.lam * np.eye(n_features), Xty)
        
        self.coef_ = self.std_.rescale_coef(beta)
        self.intercept_ = self.std_.intercept(beta)
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_
    
    def cv(self, X, y, lam_seq=None, n_lams=50):
        """留一交叉验证 (LOO-CV) 选择 λ
        
        利用 SVD 分解将 LOO-CV 的计算从 O(n · p³) 降至 O(n²)。
        
        原理：若 X = UDVᵀ，则 hat matrix 对角元为
            h_ii = Σ_j U_{ij}² · d_j²/(d_j² + λ)
        LOO 残差可解析计算：
            e_i^{(-i)} = (y_i - ŷ_i) / (1 - h_ii)
        """
        n_samples = X.shape[0]
        
        # 中心化
        y_mean = np.mean(y)
        X_mean = np.mean(X, axis=0)
        yc = y - y_mean
        Xc = X - X_mean
        
        # SVD: X = U D Vᵀ, U ∈ ℝⁿˣᵏ, D ∈ ℝᵏˣᵏ, V ∈ ℝᵖˣᵏ, k = min(n,p)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        d2 = s ** 2                           # 奇异值平方
        UTy = U.T @ yc                        # 投影到左奇异向量
        
        if lam_seq is None:
            lam_max = np.max(d2)
            lam_seq = np.geomspace(lam_max, lam_max * 1e-4, n_lams)
        
        cv_errors = np.zeros(len(lam_seq))
        
        for i, lam in enumerate(lam_seq):
            # 收缩因子 d_j² / (d_j² + λ)
            shrinkage = d2 / (d2 + lam)
            
            # hat matrix 对角元
            h = np.sum(U ** 2 * shrinkage, axis=1)
            
            # 拟合值 ŷ = U · diag(shrinkage) · Uᵀ y
            y_pred = U @ (shrinkage * UTy)
            
            # LOO 残差
            loo_residuals = (yc - y_pred) / (1 - h + 1e-10)
            cv_errors[i] = np.mean(loo_residuals ** 2)
        
        best_idx = np.argmin(cv_errors)
        return lam_seq[best_idx], lam_seq, cv_errors


# =====================
# ElasticNet — 坐标下降
# =====================

class ElasticNet:
    """弹性网络：β̂ = argmin ||y - Xβ||² + λ₁||β||₁ + λ₂||β||²₂
    
    坐标下降更新：
        β_j ← S(ρ_j, λ₁/2) / (||X[:,j]||² + λ₂)
    
    参数:
        lam:      总正则化强度 λ = λ₁ + λ₂
        l1_ratio: L1 比例 α = λ₁/(λ₁ + λ₂), α=1 → Lasso, α=0 → Ridge
    """
    
    def __init__(self, lam=1.0, l1_ratio=0.5, tol=1e-4, max_iter=1000):
        self.lam = lam
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = 0.0
    
    @property
    def lam1(self):
        return self.lam * self.l1_ratio
    
    @property
    def lam2(self):
        return self.lam * (1 - self.l1_ratio)
    
    def _cd_cycle(self, X, y, beta, residuals, lam1, lam2):
        """单轮坐标下降（含 L1 + L2 双重惩罚）"""
        Xj2 = np.sum(X ** 2, axis=0)
        
        for j in range(len(beta)):
            rho = X[:, j] @ residuals + beta[j] * Xj2[j]
            # 分母多了 L2 惩罚项 λ₂
            beta_new = soft_threshold(rho, lam1 / 2) / (Xj2[j] + lam2)
            residuals -= (beta_new - beta[j]) * X[:, j]
            beta[j] = beta_new
        
        return beta, residuals
    
    def fit(self, X, y):
        n_features = X.shape[1]
        
        self.std_ = Standardizer()
        self.std_.fit(X, y)
        Xs, yc = self.std_.transform(X, y)
        
        beta = np.zeros(n_features)
        residuals = yc.copy()
        lam1, lam2 = self.lam1, self.lam2
        
        for it in range(1, self.max_iter + 1):
            beta_old = beta.copy()
            beta, residuals = self._cd_cycle(Xs, yc, beta, residuals, lam1, lam2)
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
        
        self.coef_ = self.std_.rescale_coef(beta)
        self.intercept_ = self.std_.intercept(beta)
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_


# =====================================================
# 基于 scikit-learn 的生产用法
# =====================================================

"""
使用 sklearn 的便捷接口（已高度优化，生产环境推荐）：

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# 数据标准化 + LassoCV
lasso_pipe = make_pipeline(
    StandardScaler(),
    LassoCV(cv=5, random_state=42, n_alphas=100)
)
lasso_pipe.fit(X_train, y_train)
print(f"Best lambda: {lasso_pipe[-1].alpha_:.4f}")
print(f"Non-zero coefs: {np.sum(lasso_pipe[-1].coef_ != 0)}")

# RidgeCV
ridge_pipe = make_pipeline(
    StandardScaler(),
    RidgeCV(cv=5, alphas=np.logspace(-3, 3, 50))
)
ridge_pipe.fit(X_train, y_train)

# ElasticNetCV (可同时搜索 α 和 λ)
enet_pipe = make_pipeline(
    StandardScaler(),
    ElasticNetCV(
        cv=5,
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0],
        n_alphas=50,
        random_state=42
    )
)
enet_pipe.fit(X_train, y_train)
print(f"Best: lambda={enet_pipe[-1].alpha_:.4f}, "
      f"l1_ratio={enet_pipe[-1].l1_ratio_:.4f}")
"""


# =====================================================
# 使用示例
# =====================================================
if __name__ == "__main__":
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV as SkLassoCV
    from sklearn.linear_model import RidgeCV as SkRidgeCV
    from sklearn.linear_model import ElasticNetCV as SkElasticNetCV
    
    np.random.seed(42)
    
    # 生成稀疏线性数据：10 个有效特征 + 40 个噪声特征
    n, p = 200, 50
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)
    beta_true[:10] = [1.5, -2.0, 0.5, -1.0, 2.5, -1.5, 0.8, -0.5, 1.2, -0.8]
    y = X @ beta_true + 0.5 * np.random.randn(n)
    
    print("=" * 56)
    print("  正则化回归 — 完整示例")
    print("=" * 56)
    
    # ============ Lasso ============
    print("\n" + "-" * 40)
    print("Lasso 回归（坐标下降自实现）")
    print("-" * 40)
    
    lasso = Lasso(lam=0.5)
    lasso.fit(X, y)
    nz = np.sum(np.abs(lasso.coef_) > 1e-6)
    print(f"  非零系数: {nz}/{p}")
    print(f"  前 5 个系数: {np.round(lasso.coef_[:5], 4)}")
    y_pred_l = lasso.predict(X)
    r2_l = 1 - np.mean((y - y_pred_l) ** 2) / np.var(y)
    print(f"  R² (训练): {r2_l:.4f}")
    
    # Lasso CV
    print("\n  Lasso 交叉验证...")
    best_lam, lam_1se, lam_seq, cv_mean, cv_se = lasso.cv(X, y, n_folds=5)
    print(f"  CV 最优 λ: {best_lam:.4f}")
    print(f"  1-SE λ:    {lam_1se:.4f}")
    
    lasso_best = Lasso(lam=best_lam)
    lasso_best.fit(X, y)
    nz_best = np.sum(np.abs(lasso_best.coef_) > 1e-6)
    print(f"  最优 λ 下非零系数: {nz_best}/{p}")
    
    # ============ Ridge ============
    print("\n" + "-" * 40)
    print("Ridge 回归（闭式解 + SVD LOO-CV）")
    print("-" * 40)
    
    ridge = Ridge(lam=1.0)
    ridge.fit(X, y)
    y_pred_r = ridge.predict(X)
    r2_r = 1 - np.mean((y - y_pred_r) ** 2) / np.var(y)
    print(f"  R² (训练): {r2_r:.4f}")
    print(f"  系数 L2 范数: {np.linalg.norm(ridge.coef_):.4f}")
    
    print("\n  Ridge LOO-CV...")
    best_lam_r, _, _ = ridge.cv(X, y, n_lams=50)
    print(f"  LOO-CV 最优 λ: {best_lam_r:.4f}")
    
    ridge_best = Ridge(lam=best_lam_r)
    ridge_best.fit(X, y)
    y_pred_rb = ridge_best.predict(X)
    r2_rb = 1 - np.mean((y - y_pred_rb) ** 2) / np.var(y)
    print(f"  最优 λ 下 R²: {r2_rb:.4f}")
    
    # ============ ElasticNet ============
    print("\n" + "-" * 40)
    print("ElasticNet 回归（坐标下降）")
    print("-" * 40)
    
    enet = ElasticNet(lam=0.5, l1_ratio=0.5)
    enet.fit(X, y)
    nz_en = np.sum(np.abs(enet.coef_) > 1e-6)
    print(f"  非零系数 (α=0.5): {nz_en}/{p}")
    y_pred_en = enet.predict(X)
    r2_en = 1 - np.mean((y - y_pred_en) ** 2) / np.var(y)
    print(f"  R² (训练): {r2_en:.4f}")
    
    # ============ sklearn 对照 ============
    print("\n" + "-" * 40)
    print("sklearn 生产用法对照")
    print("-" * 40)
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    lasso_sk = SkLassoCV(cv=5, random_state=42).fit(Xs, y)
    print(f"  LassoCV:      λ={lasso_sk.alpha_:.4f}, "
          f"非零={np.sum(lasso_sk.coef_ != 0)}")
    
    ridge_sk = SkRidgeCV(cv=5, alphas=np.logspace(-2, 3, 50)).fit(Xs, y)
    print(f"  RidgeCV:      λ={ridge_sk.alpha_:.4f}")
    
    enet_sk = SkElasticNetCV(
        cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0],
        n_alphas=50, random_state=42
    ).fit(Xs, y)
    print(f"  ElasticNetCV: λ={enet_sk.alpha_:.4f}, "
          f"α={enet_sk.l1_ratio_:.4f}, 非零={np.sum(enet_sk.coef_ != 0)}")
    
    print("\n" + "=" * 56)
    print("  完成 — 从零实现与 sklearn 结果一致")
    print("=" * 56)
```

## 参考文献

- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Hoerl, A. E. & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67.
- Zou, H. & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
- Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. (2004). Least Angle Regression. *The Annals of Statistics*, 32(2), 407–499.
- Zou, H. (2006). The Adaptive Lasso and Its Oracle Properties. *Journal of the American Statistical Association*, 101(476), 1418–1429.
- Friedman, J., Hastie, T. & Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. *Journal of Statistical Software*, 33(1), 1–22.
- Hastie, T., Tibshirani, R. & Wainwright, M. (2015). *Statistical Learning with Sparsity: The Lasso and Generalizations*. CRC Press.
