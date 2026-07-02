# Random Forest — 随机森林

- **来源**: Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- **DOI**: 10.1023/A:1010933404324
- **方法类别**: 机器学习 / 集成学习 / 分类与回归

## 数学设定

### 模型框架
随机森林是 **Bagging + 随机子空间** 的集成方法。通过构建 $B$ 棵决策树 $\{T_b\}_{b=1}^{B}$，聚合每棵树的预测结果：

- **分类**：多数投票
  $$
  \hat{y} = \arg\max_c \sum_{b=1}^{B} \mathbb{I}[T_b(x) = c]
  $$

- **回归**：平均
  $$
  \hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
  $$

### 每棵树的构建（CART 框架）
1. **Bootstrap 抽样**：从 $N$ 个样本中有放回抽取 $N$ 个（约 $63.2\%$ 唯一样本）
2. **随机子空间**：每个分裂节点从 $p$ 个特征中随机选 $m_{try}$ 个作为候选
   - 分类默认：$m_{try} = \lfloor \sqrt{p} \rfloor$
   - 回归默认：$m_{try} = \lfloor p/3 \rfloor$
3. **最优分裂**：在 $m_{try}$ 个候选特征中选择最佳分裂点

### 分裂准则
- **分类**：基尼不纯度（CART 标准）或交叉熵
  $$
  G = 1 - \sum_{c=1}^{C} p_c^2
  $$
  分裂增益：$\Delta G = G_{parent} - \sum_{k} \frac{N_k}{N} G_{child_k}$

- **回归**：均方误差
  $$
  \Delta = SSE_{parent} - \sum_{k} SSE_{child_k}
  $$

### OOB（Out-of-Bag）误差
每个样本约 $1/e \approx 36.8\%$ 的概率不出现在某棵树的 bootstrap 样本中，这些 OOB 样本用于无偏估计泛化误差：
$$
\text{OOB Error} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}[\hat{y}_i^{\text{OOB}} \neq y_i]
$$

### 变量重要性
- **Gini 重要性**：所有树中该特征的分裂基尼增益总和
- **Permutation 重要性**：打乱该特征后 OOB 误差的增加量

### 收敛性
$$
\lim_{B \to \infty} \text{泛化误差} \leq \bar{\rho} \cdot \frac{s^2}{\theta^2}
$$

其中 $\bar{\rho}$ 为树间平均相关性，$s$ 为单棵树强度，$\theta$ 为边际效应。这意味着 RF 不会过拟合（$B$ 增大不会增加泛化误差）。

## 关键假设
- 特征含有预测信号（优于随机猜测）
- 树间相关性不能太高（否则集成增益消失）
- 对特征尺度不敏感（无需归一化）
- 能自动处理非线性、交互作用、缺失值

## 适用场景
- **中小型表格数据**（$N$ 几百到几十万，$p$ 几十到几千）
- **特征重要性筛选**：高维数据中快速筛选重要变量
- **缺失值较多的数据**：RF 内置缺失值处理（代理分裂 / 近邻填充）
- **非线性关系 + 交互效应**复杂的问题
- **作为 baseline**：几乎所有建模问题的第一道 baseline

### 不适用
- **超高维稀疏数据**（如文本分类）：线性模型 / GBDT 效果更好
- **需要外推预测**：RF 不能外推训练数据范围之外
- **可解释性要求极高**：单棵树可解释，RF 较难
- **极端不平衡分类**：需配合采样或代价敏感学习
- **在线学习 / 流数据**：RF 是批量模型

## 实现要点

### 关键超参数
| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| $n\_estimators$ | [100, 2000] | 100 | 树的数量，越大越稳定 |
| $m_{try}$ | [$\sqrt{p}$, $p/3$] | $\sqrt{p}$ / $p/3$ | 每节点候选特征数 |
| $min\_samples\_split$ | [2, 20] | 2 | 内部节点最小样本数 |
| $min\_samples\_leaf$ | [1, 10] | 1 | 叶节点最小样本数 |
| $max\_depth$ | [None, 30] | None | 树的深度控制 |
| $max\_features$ | log2, sqrt, auto, None | sqrt | 特征选择策略 |

### 调优经验
1. **先调 $m_{try}$**：对性能影响最大，通常 $\sqrt{p}$ 附近最优
2. **再调树深度**：$max\_depth$ 适度限制可提高泛化，同时降低计算
3. **最后调样本量参数**：$min\_samples\_leaf$ 增大可防止小类过拟合
4. **$n\_estimators$ 越大越好**（计算资源允许下），不愁过拟合
5. **impurity\_based 重要性有偏**（偏向连续/高基数特征），用 permutation 重要性更可靠

### 代码

```python
import numpy as np
from collections import Counter

class DecisionTreeCART:
    """CART 决策树（随机森林的基学习器）"""
    
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None, task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.task = task
        self.tree = None
        
    def _gini(self, y):
        """基尼不纯度"""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _mse(self, y):
        """均方误差"""
        return np.var(y) * len(y)
    
    def _impurity(self, y):
        return self._gini(y) if self.task == 'classification' else self._mse(y)
    
    def _leaf_value(self, y):
        if self.task == 'classification':
            return Counter(y).most_common(1)[0][0]
        return np.mean(y)
    
    def _split(self, X, y, feature, threshold):
        """按特征阈值分裂"""
        left = np.where(X[:, feature] <= threshold)[0]
        right = np.where(X[:, feature] > threshold)[0]
        return left, right
    
    def _best_split(self, X, y, feature_idx):
        """找最优分裂点"""
        best_gain = -1
        best_feat, best_thr = None, None
        parent_imp = self._impurity(y)
        n_total = len(y)
        
        for f in feature_idx:
            vals = X[:, f]
            # 只在分位点尝试，加速
            thresholds = np.percentile(vals, np.linspace(10, 90, 9))
            for thr in thresholds:
                left, right = self._split(X, y, f, thr)
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                gain = parent_imp - (len(left) / n_total) * self._impurity(y[left]) \
                                     - (len(right) / n_total) * self._impurity(y[right])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thr = thr
        return best_feat, best_thr
    
    def _build(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 终止条件
        if (self.max_depth and depth >= self.max_depth) \
           or n_samples < self.min_samples_split \
           or n_classes == 1:
            return {'value': self._leaf_value(y), 'size': n_samples}
        
        # 特征子空间
        if self.max_features:
            feat_idx = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feat_idx = np.arange(n_features)
        
        feat, thr = self._best_split(X, y, feat_idx)
        if feat is None:
            return {'value': self._leaf_value(y), 'size': n_samples}
        
        left_idx, right_idx = self._split(X, y, feat, thr)
        
        return {
            'feature': feat,
            'threshold': thr,
            'left': self._build(X[left_idx], y[left_idx], depth + 1),
            'right': self._build(X[right_idx], y[right_idx], depth + 1),
            'size': n_samples
        }
    
    def fit(self, X, y):
        n_features = X.shape[1]
        if self.max_features is None:
            if self.task == 'classification':
                self.max_features = int(np.sqrt(n_features))
            else:
                self.max_features = max(1, n_features // 3)
        self.tree = self._build(X, y)
        return self
    
    def _predict_tree(self, x, node):
        if 'value' in node:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_tree(x, node['left'])
        return self._predict_tree(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])


class RandomForest:
    """随机森林 — 分类/回归"""
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1,
                 max_features=None, task='classification', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.task = task
        self.random_state = random_state
        self.trees = []
        self.oob_score = None
        self.feature_importances_ = None
        
    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        self.trees = []
        oob_preds = np.zeros(n_samples) if self.task == 'regression' else {}
        if self.task == 'classification':
            oob_preds = {i: [] for i in range(n_samples)}
        else:
            oob_preds = np.zeros(n_samples)
            oob_counts = np.zeros(n_samples)
        
        # 特征重要性累积
        self.feature_importances_ = np.zeros(n_features)
        
        for b in range(self.n_estimators):
            # Bootstrap 抽样
            idx = np.random.choice(n_samples, n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(idx))
            
            X_boot, y_boot = X[idx], y[idx]
            X_oob, y_oob = X[oob_idx], y[oob_idx]
            
            # 训练单棵树
            tree = DecisionTreeCART(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                task=self.task
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # OOB 预测
            if len(oob_idx) > 0:
                preds = tree.predict(X_oob)
                for i_oob, i_orig in enumerate(oob_idx):
                    if self.task == 'classification':
                        oob_preds[i_orig].append(preds[i_oob])
                    else:
                        oob_preds[i_orig] += preds[i_oob]
                        oob_counts[i_orig] += 1
                        
        # OOB 评分
        if self.task == 'classification':
            y_oob_pred = np.array([
                Counter(oob_preds[i]).most_common(1)[0][0] if oob_preds[i] else -1
                for i in range(n_samples)
            ])
            valid = y_oob_pred != -1
            self.oob_score = np.mean(y_oob_pred[valid] == y[valid])
        else:
            valid = oob_counts > 0
            y_oob_pred = np.where(valid, oob_preds / oob_counts, 0)
            self.oob_score = 1 - np.mean((y_oob_pred[valid] - y[valid]) ** 2) \
                             / np.var(y[valid])
        
        return self
    
    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        if self.task == 'classification':
            # 对每个样本多数投票
            return np.array([Counter(preds[:, i]).most_common(1)[0][0] 
                           for i in range(preds.shape[1])])
        return np.mean(preds, axis=0)
    
    def predict_proba(self, X):
        """返回分类概率（仅分类任务）"""
        if self.task != 'classification':
            raise AttributeError("predict_proba only for classification")
        preds = np.array([t.predict(X) for t in self.trees])
        n_classes = len(np.unique(preds))
        proba = np.zeros((X.shape[0], n_classes))
        for i in range(X.shape[0]):
            counts = Counter(preds[:, i])
            for c, cnt in counts.items():
                proba[i, int(c)] = cnt / self.n_estimators
        return proba


# =====================
# 使用示例
# =====================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # 训练
    rf = RandomForest(n_estimators=200, max_depth=10, task='classification')
    rf.fit(X_train, y_train)
    
    # 评估
    y_pred = rf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"OOB Score: {rf.oob_score:.4f}")
```

### 基于 scikit-learn 的生产用法
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# 快速 baseline
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# 超参数搜索
param_dist = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=20, cv=5, n_jobs=-1
)
search.fit(X_train, y_train)
print(search.best_params_)
```

## 参考文献
Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
