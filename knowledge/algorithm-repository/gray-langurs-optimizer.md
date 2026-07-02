# Gray Langurs Optimizer (GLO) — 灰叶猴优化算法

- **来源**: Gray Langurs Optimizer: A Multi-group Bio-inspired Optimization Algorithm | Artificial Intelligence Review | 2026
- **DOI**: 10.1007/s10462-026-11529-2
- **作者**: Saeid Barshandeh, Nima Khodadadi, Khalid M. Mosalam
- **方法类别**: 群智能优化 / 元启发式

## 数学设定

### 灵感来源
模拟灰叶猴（*Semnopithecus*）的三种社会群体结构及其行为：

1. **一雄群（One-male Group, OMG）**：一只优势雄猴 + 多只雌猴 + 幼猴
2. **多雄群（Multi-male Group, MMG）**：多只雄猴共存，共享资源
3. **全雄群（All-male Group, AMG）**：被驱逐或主动离开的雄猴形成的单身群

此外模拟了：群体间迁移（alpha 雄猴死亡触发）、求偶行为、自由漫游。

### 核心机制

#### 1. 群体初始化
$$
X_{i,j} = L_j + r \cdot (U_j - L_j), \quad i=1,\dots,N, \; j=1,\dots,D
$$

其中 $N$ 为种群规模，$D$ 为维度，$r \sim U(0,1)$。

#### 2. 多群组并行搜索
种群分成三个子群（OMG、MMG、AMG），各自独立搜索，周期性交流：
- **OMG**（占比最大）：围绕当前最优解局部开发（exploitation）
- **MMG**（中等规模）：探索与开发平衡
- **AMG**（最小规模）：全局探索，避免早熟收敛

#### 3. 位置更新
**OMG 更新**（围绕 alpha 开发）：
$$
X_{i}^{t+1} = X_{\alpha}^{t} + \beta \cdot (X_{\alpha}^{t} - X_{i}^{t}) \cdot r_1 + \gamma \cdot (X_{r1}^{t} - X_{r2}^{t}) \cdot r_2
$$

其中 $X_{\alpha}$ 为当前最优解，$\beta$、$\gamma$ 为控制参数，$r_1, r_2 \sim U(0,1)$。

**MMG 更新**（社会互动）：
$$
X_{i}^{t+1} = X_{i}^{t} + \phi \cdot (X_{i}^{t} - X_{j}^{t}) + \psi \cdot (X_{k}^{t} - X_{l}^{t})
$$

其中 $\phi,\psi$ 为步长系数，$i,j,k,l$ 为群内不同个体。

**AMG 更新**（自由漫游 — 全局探索）：
$$
X_{i}^{t+1} = X_{i}^{t} + \delta \cdot (X_{r3}^{t} - X_{r4}^{t}) + \epsilon \cdot (L + r \cdot (U - L))
$$

$\delta$ 为探索步长，$\epsilon$ 为随机漫游系数，$r_3,r_4$ 为随机索引。

#### 4. 迁移机制
当 alpha 雄猴死亡（即最优解 $N_{eval}$ 次未更新），触发迁移：
- 该 OMG 解散，个体重新分配到 MMG 或 AMG
- 新的 OMG 从其他群组中选举产生
- 机制设计保证群组数量动态平衡

#### 5. 算法流程
```
1. 初始化种群，随机分配到 OMG/MMG/AMG
2. 评估所有个体适应度
3. 对每个群组分别执行位置更新
4. 评估新位置适应度，更新全局最优
5. 每隔 k 代执行群组间信息交换
6. 检查迁移条件（alpha 解停滞）
7. 若满足触发迁移
8. 重复 2-7 直至终止条件
9. 返回全局最优解
```

### 关键参数
| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| 种群大小 $N$ | [20, 200] | 50 | 总个体数 |
| OMG 比例 | [0.3, 0.6] | 0.5 | 一雄群占比 |
| MMG 比例 | [0.2, 0.4] | 0.3 | 多雄群占比 |
| AMG 比例 | [0.1, 0.3] | 0.2 | 全雄群占比 |
| $\beta$ | [0.5, 2.0] | 1.0 | OMG 开发步长 |
| $\gamma$ | [0.1, 1.0] | 0.5 | OMG 差异扰动 |
| 迁移阈值 | [5, 50] | 10 | alpha 停滞代数 |
| 信息交换间隔 k | [1, 10] | 3 | 群组间通信频率 |

## 适用场景
- **连续优化问题**：标准 benchmark 函数、工程约束优化
- **模型超参数调优**：SVM、BP、LSTM、CNN、Transformer 等
- **组合优化变体**：路径规划（无人机/机器人）、TSP
- **信号分解参数优化**：VMD、CEEMDAN 等
- **PID 参数整定、图像分割、故障诊断**

### 不适用
- 离散/组合优化无适配变体（需重新设计编码）
- 目标函数评估代价极高时（GLO 的多群组并行计算量较大）
- 需严格收敛保证的理论问题

## 实现要点
- 多群组结构是关键创新点，需确保群组间有足够的差异化和信息流通
- 迁移阈值不宜过小，否则群组频繁重组破坏收敛
- AMG 的随机漫游系数 $\epsilon$ 随迭代递减效果通常更好
- 建议与 PSO、GWO、DBO 等经典算法对比，突出多群组优势
- 已知开源实现：MATLAB（MathWorks File Exchange）、CSDN 含代码

## 基准测试表现
| 测试集 | 对比算法 | 结果 |
|--------|---------|------|
| 23 个经典函数（CEC2005） | ALO, AO, AOA, ASO, KH, RSA, SHO, STOA, ChOA, CCO, DE, DOA, PSO, DBO | 多数函数排名前 2 |
| 27 个 CEC2017 函数 | 同上 | 高维（50D/100D）优势明显 |
| 6 个工程设计问题 | 同上 | 综合最优 |
| 统计检验 | — | Friedman + Wilcoxon 显著差异 |

## 代码

```python
import numpy as np

class GrayLangursOptimizer:
    """Gray Langurs Optimizer (GLO) 简化实现"""
    
    def __init__(self, n_pop=50, n_dim=30, bounds=None, max_iter=500,
                 omg_ratio=0.5, mmg_ratio=0.3, amg_ratio=0.2,
                 beta=1.0, gamma=0.5, migration_threshold=10, k_exchange=3):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.bounds = bounds  # [(lb, ub), ...] per dimension
        self.max_iter = max_iter
        self.beta = beta
        self.gamma = gamma
        self.migration_threshold = migration_threshold
        self.k_exchange = k_exchange
        
        # 群组大小
        self.n_omg = int(n_pop * omg_ratio)
        self.n_mmg = int(n_pop * mmg_ratio)
        self.n_amg = n_pop - self.n_omg - self.n_mmg
        
        # 状态
        self.positions = None
        self.fitness = None
        self.alpha = None  # 全局最优位置
        self.alpha_fit = np.inf
        self.stagnation = 0  # 停滞计数器
        
    def _init_population(self):
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        self.positions = lb + np.random.rand(self.n_pop, self.n_dim) * (ub - lb)
        self.fitness = np.full(self.n_pop, np.inf)
        
    def _evaluate(self, obj_func):
        for i in range(self.n_pop):
            fit = obj_func(self.positions[i])
            self.fitness[i] = fit
            if fit < self.alpha_fit:
                self.alpha_fit = fit
                self.alpha = self.positions[i].copy()
                self.stagnation = 0
                
    def _omg_update(self, idx_group, lb, ub):
        """一雄群更新：围绕 alpha 开发"""
        for i in idx_group:
            r1, r2 = np.random.rand(2)
            r_idx = np.random.choice(idx_group)
            diff = self.alpha - self.positions[i]
            disp = self.positions[r_idx] - self.positions[np.random.choice(idx_group)]
            new_pos = self.alpha + self.beta * diff * r1 + self.gamma * disp * r2
            self.positions[i] = np.clip(new_pos, lb, ub)
            
    def _mmg_update(self, idx_group, lb, ub):
        """多雄群更新：个体间社会互动"""
        for i in idx_group:
            phi, psi = np.random.rand(2)
            j, k, l = np.random.choice(idx_group, 3, replace=False)
            diff1 = self.positions[i] - self.positions[j]
            diff2 = self.positions[k] - self.positions[l]
            new_pos = self.positions[i] + phi * diff1 + psi * diff2
            self.positions[i] = np.clip(new_pos, lb, ub)
            
    def _amg_update(self, idx_group, lb, ub):
        """全雄群更新：自由漫游（全局探索）"""
        delta = np.random.rand()
        epsilon = 0.5 * (1 - self.current_iter / self.max_iter)  # 递减
        for i in idx_group:
            r3, r4 = np.random.choice(self.n_pop, 2)
            roam = lb + np.random.rand(self.n_dim) * (ub - lb)
            new_pos = (self.positions[i] 
                       + delta * (self.positions[r3] - self.positions[r4])
                       + epsilon * (roam - self.positions[i]))
            self.positions[i] = np.clip(new_pos, lb, ub)
            
    def _migration(self):
        """迁移机制：alpha 停滞过久时重组"""
        idx_all = np.random.permutation(self.n_pop)
        self.n_omg = max(1, int(self.n_pop * 0.4))
        self.n_mmg = max(1, int(self.n_pop * 0.35))
        self.n_amg = self.n_pop - self.n_omg - self.n_mmg
        # 重新分配群组
        self.omg_idx = idx_all[:self.n_omg]
        self.mmg_idx = idx_all[self.n_omg:self.n_omg+self.n_mmg]
        self.amg_idx = idx_all[self.n_omg+self.n_mmg:]
        self.stagnation = 0
        
    def optimize(self, obj_func, verbose=True):
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        
        # 初始化
        self._init_population()
        self._evaluate(obj_func)
        
        # 初始群组分配
        idx_all = np.random.permutation(self.n_pop)
        self.omg_idx = idx_all[:self.n_omg]
        self.mmg_idx = idx_all[self.n_omg:self.n_omg+self.n_mmg]
        self.amg_idx = idx_all[self.n_omg+self.n_mmg:]
        
        self.convergence = [self.alpha_fit]
        
        for t in range(self.max_iter):
            self.current_iter = t
            
            # 各群组独立更新
            self._omg_update(self.omg_idx, lb, ub)
            self._mmg_update(self.mmg_idx, lb, ub)
            self._amg_update(self.amg_idx, lb, ub)
            
            # 评估
            self._evaluate(obj_func)
            self.convergence.append(self.alpha_fit)
            
            # 群组间信息交换（间隔 k 代）
            if t % self.k_exchange == 0:
                # 混合排序后重分配
                order = np.argsort(self.fitness)
                n1 = self.n_omg
                n2 = self.n_omg + self.n_mmg
                self.omg_idx = order[:n1]
                self.mmg_idx = order[n1:n2]
                self.amg_idx = order[n2:]
                
            # 迁移检查
            if self.stagnation >= self.migration_threshold:
                self._migration()
                
            if verbose and t % 50 == 0:
                print(f"Iter {t:4d} | Best: {self.alpha_fit:.6e} | "
                      f"OMG:{len(self.omg_idx)} MMG:{len(self.mmg_idx)} AMG:{len(self.amg_idx)}")
                
        return self.alpha, self.alpha_fit, self.convergence


# =====================
# 使用示例
# =====================
if __name__ == "__main__":
    # Rastrigin 函数（最小化）
    def rastrigin(x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    bounds = [(-5.12, 5.12)] * 30
    
    glo = GrayLangursOptimizer(
        n_pop=50, n_dim=30, bounds=bounds, max_iter=500,
        beta=1.0, gamma=0.5, migration_threshold=10
    )
    best_x, best_f, conv = glo.optimize(rastrigin)
    print(f"\n最优适应度: {best_f:.6e}")
```

## 参考文献
Barshandeh, S., Khodadadi, N., & Mosalam, K. M. (2026). Gray Langurs Optimizer: A Multi-group Bio-inspired Optimization Algorithm. *Artificial Intelligence Review*. DOI: 10.1007/s10462-026-11529-2
