---
name: bio-inspired-algo
description: Design novel bio-inspired optimization algorithms. Takes a biological observation and transforms it into a mathematically rigorous, benchmark-validated metaheuristic algorithm ready for publication.
---

# Bio-Inspired Algorithm Design

将生物灵感转化为可发表的新型元启发式算法。覆盖从现象观察到数学建模、实现、验证、发表的完整 pipeline。

## When to use

- 你想提出一个**新的群智能优化算法**
- 你观察到某种生物行为，想判断它能否做成算法
- 你需要**系统化的方法论**而非碰运气式地"取个名字凑公式"
- 你想了解现有算法有哪些生物机制已被"用完"，哪些还有空白
- 你需要完整的**benchmark 验证方案**来支撑论文

## Trigger keywords
- "新算法"
- "生物启发"
- "群智能"
- "元启发式"
- "优化算法"
- "仿生算法"
- "bio-inspired"
- "metaheuristic"
- "swarm intelligence"

## The Bio-to-Algorithm Pipeline

```
生物观察
   ↓
机制抽象（哪些行为可数学化？）
   ↓
数学建模（算子设计 + 参数控制）
   ↓
算法框架（伪代码 + 流程图）
   ↓
代码实现
   ↓
标准测试（CEC benchmark 套件）
   ↓
统计检验（Friedman / Wilcoxon）
   ↓
应用验证（工程设计问题）
   ↓
论文撰写
```

---

## Phase 1: 生物机制评估

不是所有生物行为都适合做算法。评估一个生物灵感时需要回答：

### 可行性筛查 checklist

- [ ] 该行为是否包含 **探索（exploration）** 与 **开发（exploitation）** 的天然平衡？
- [ ] 是否有 **明确的个体-群体交互**（信息传递/竞争/合作）？
- [ ] 是否有 **环境反馈机制**（个体根据环境状态调整行为）？
- [ ] 行为是否可以参数化？（连续可微、离散、混合）
- [ ] 该机制是否与现有算法有本质区别？（非 PSO/GWO/DE 的简单变体）
- [ ] 是否有足够的行为学/生物学文献支撑机制设计？

### 常见的"可算法化"生物机制

| 机制 | 算法对应 | 经典案例 |
|------|---------|---------|
| 群体觅食 / 搜索 | 探索-开发平衡 | PSO（粒子群）, ABC（人工蜂群） |
| 社会等级 / 领导力 | 最优解引导 | GWO（灰狼）, ChOA（黑猩猩） |
| 繁殖 / 遗传 / 变异 | 全局搜索多样性 | GA（遗传算法）, DE（差分进化） |
| 共生 / 寄生 / 捕食 | 多群组竞争协作 | SOS（共生搜索）, CSO（竞争群） |
| 免疫记忆 | 局部搜索与记忆 | AIS（人工免疫系统） |
| 迁徙 / 扩散 | 全局探索跳出局部 | MBO（君主蝴蝶）, GLO（灰叶猴） |
| 筑巢 / 建筑 | 基于位置/轨迹的搜索 | PGO（鸽群）, BBO（生物地理学） |
| 化学通讯（信息素） | 间接信息共享 | ACO（蚁群）, BCO（蜜蜂） |
| 赌博 / 风险决策 | 随机搜索策略 | 较新的风险导向算法 |

---

## Phase 2: 数学建模模板

### 2.1 种群初始化
标准方式（实数编码）：
$$
X_{i,j} = L_j + r_{i,j} \cdot (U_j - L_j), \quad i=1,\dots,N, \; j=1,\dots,D
$$

如果生物机制包含特殊初始状态（如年龄结构、等级分布），修改初始化为对应分布。

### 2.2 核心算子设计模式

大多数生物启发算法的算子可以映射到以下 4 种基本操作：

**A. 向最优解靠拢（Attraction / Leadership）**
$$
X_i^{t+1} = X_i^t + \alpha \cdot (X_{best}^t - X_i^t) \cdot r
$$
对应生物行为：追随首领、趋向食物源、群体中心聚集。

**B. 随机探索（Random Walk / Perturbation）**
$$
X_i^{t+1} = X_i^t + \beta \cdot \text{Lévy}(\lambda) \oplus (X_{best}^t - X_i^t)
$$
对应生物行为：随机漫游、Levy 飞行（果蝇/信天翁）、布朗运动。

**C. 个体间差异化（Mutation / Competition）**
$$
X_i^{t+1} = X_{r1}^t + F \cdot (X_{r2}^t - X_{r3}^t)
$$
对应生物行为：遗传变异、竞争、淘汰。

**D. 环境反馈（Conditional / Switching）**
$$
X_i^{t+1} = 
\begin{cases}
\text{策略A}, & \text{if } p < \tau \\
\text{策略B}, & \text{otherwise}
\end{cases}
$$
对应生物行为：环境刺激响应、应激行为、昼夜节律切换。

### 2.3 创新点设计策略

| 策略 | 说明 | 论文案例 |
|------|------|---------|
| **新生物机制** | 从未被算法化的生物行为 | GLO 的三种社会群体结构 |
| **旧机制 + 新数学** | 已知生物行为，但用不同的数学建模 | 从 Levy 飞行到自适应步长 |
| **多机制融合** | 组合 2-3 种生物行为 | 觅食+迁徙+社会等级 |
| **自适应参数** | 现有算法的参数让算法自己调 | 自适应的 exploration/exploitation 比 |
| **问题结构利用** | 利用问题的数学结构增强搜索 | 将梯度信息引入群智能 |

### 2.4 参数控制策略

```
                探索（Exploration）                   开发（Exploitation）
                    |                                      |
初始阶段 ──────────→ ●──────────────────────────────────────→
                    |  ← 参数 A 大, 参数 B 小              |
                    |                                      |
中间阶段 ──────────→ ────●──────────────────────────────────→
                    |    ← A 递减, B 递增                  |
                    |                                      |
末期阶段 ──────────→ ───────────────●──────────────────────→
                    |               ← A 小, B 大            |
```

常见的参数衰减函数：
- 线性：$a(t) = a_{max} - (a_{max} - a_{min}) \cdot t/T$
- 指数：$a(t) = a_{max} \cdot \exp(-\beta \cdot t/T)$
- 余弦：$a(t) = a_{min} + 0.5(a_{max} - a_{min})(1 + \cos(\pi t/T))$

---

## Phase 3: 标准测试协议

### 3.1 Benchmark 函数套件

| 测试集 | 函数数 | 特征 | 用途 |
|--------|--------|------|------|
| CEC2005 (23 经典) | 23 | 单峰(7)+多峰(6)+固定维度(10) | 基础测试 |
| CEC2014 | 30 | 旋转+位移+混合 | 中等难度 |
| CEC2017 | 29 | 混合+组合函数 | 主流 benchmark |
| CEC2020 | 10 | 精选难函数 | 前沿对比 |
| CEC2022 | 12 | 最新基准 | 最前沿 |

### 3.2 实验设计

```
维度设置: 10D, 30D, 50D, 100D
独立运行: 30 次（每次不同随机种子）
终止条件: 最大函数评价次数（如 D*10000）
对比算法: 8-15 个（含经典 + 近年 + 最先进）
```

### 3.3 统计验证

**必须做的 3 个检验：**

1. **Wilcoxon 秩和检验** — 两两对比（算法 A vs 算法 B）
   - H0: 两个算法的中位数性能无差异
   - 显著性水平 $\alpha = 0.05$

2. **Friedman 检验** — 多算法整体排名
   - 给出所有算法在所有函数上的平均排名
   - 排名越低越好

3. **Holm 后验检验** — Friedman 拒绝后的多重比较
   - 控制 family-wise error rate

**汇报表格模板：**

| Function | Proposed | PSO | GWO | ... | Wilcoxon p (PSO) | Wilcoxon p (GWO) |
|----------|----------|-----|-----|-----|------------------|------------------|
| F1 | **1.2e-30** | 4.5e-15 | 2.1e-10 | ... | 1.2e-6 (†) | 1.2e-6 (†) |
| F2 | **3.4e-16** | 7.8e-10 | 1.3e-10 | ... | 1.2e-6 (†) | 0.032 (†) |
| ... |

- Bold: 最优结果
- †: 在 0.05 水平显著优于对比算法

### 3.4 应用验证

选 3-6 个真实的工程设计问题（约束优化）：
- **焊接梁设计（Welded Beam）**
- **拉伸/压缩弹簧设计（Tension/Compression Spring）**
- **减速器设计（Speed Reducer）**
- **压力容器设计（Pressure Vessel）**
- 其他领域：路径规划、PID 调优、图像分割、特征选择

使用**罚函数法**或**ε-约束处理**处理约束条件。

---

## Phase 4: 代码模板

```python
import numpy as np
from typing import Callable, Tuple, List

class BioInspiredOptimizer:
    """新型生物启发优化算法的标准模板"""
    
    def __init__(
        self,
        n_pop: int = 50,
        n_dim: int = 30,
        bounds: List[Tuple[float, float]] = None,
        max_fes: int = 300000,
        name: str = "Bio-Algorithm"
    ):
        self.n_pop = n_pop
        self.n_dim = n_dim
        self.bounds = bounds
        self.max_fes = max_fes  # 统一用 FES 而非迭代次数
        self.name = name
        self.n_fes = 0
        self.convergence_curve = []
        
    def init_population(self) -> np.ndarray:
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        return lb + np.random.rand(self.n_pop, self.n_dim) * (ub - lb)
    
    def boundary_check(self, X: np.ndarray) -> np.ndarray:
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        return np.clip(X, lb, ub)
    
    def evaluate(self, X: np.ndarray, obj_func: Callable) -> float:
        self.n_fes += 1
        return obj_func(X)
    
    # =========== 用户需要实现的核心方法 ===========
    
    def exploration_operator(self, X: np.ndarray, best: np.ndarray, t: int, T: int) -> np.ndarray:
        """探索算子：对应生物行为的全局搜索阶段"""
        raise NotImplementedError
    
    def exploitation_operator(self, X: np.ndarray, best: np.ndarray, t: int, T: int) -> np.ndarray:
        """开发算子：对应生物行为的局部精化阶段"""
        raise NotImplementedError
    
    def switch_condition(self, t: int, T: int) -> bool:
        """探索/开发切换条件"""
        return np.random.rand() < (1 - t / T)
    
    def post_update(self, X: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """后处理（可选）：对应特殊生物机制如迁移、死亡、繁殖"""
        return X
    
    # =========== 优化流程（通常不需要修改） ===========
    
    def optimize(self, obj_func: Callable, verbose: bool = True) -> Tuple[np.ndarray, float, List]:
        self.n_fes = 0
        self.convergence_curve = []
        
        # 1. 初始化
        X = self.init_population()
        fitness = np.array([self.evaluate(X[i], obj_func) for i in range(self.n_pop)])
        idx_best = np.argmin(fitness)
        best = X[idx_best].copy()
        best_fit = fitness[idx_best]
        self.convergence_curve.append(best_fit)
        
        T = self.max_fes // self.n_pop  # 约等价迭代次数
        
        for t in range(1, T + 1):
            # 2. 探索 vs 开发切换
            if self.switch_condition(t, T):
                X_new = self.exploration_operator(X, best, t, T)
            else:
                X_new = self.exploitation_operator(X, best, t, T)
            
            # 3. 边界检查
            X_new = self.boundary_check(X_new)
            
            # 4. 评估
            for i in range(self.n_pop):
                fit_new = self.evaluate(X_new[i], obj_func)
                if fit_new < fitness[i]:
                    X[i] = X_new[i]
                    fitness[i] = fit_new
                    if fit_new < best_fit:
                        best = X[i].copy()
                        best_fit = fit_new
            
            # 5. 后处理
            X = self.post_update(X, fitness)
            
            self.convergence_curve.append(best_fit)
            
            if verbose and t % (T // 10) == 0:
                print(f"FES: {self.n_fes:6d} | Best: {best_fit:.6e}")
                
            if self.n_fes >= self.max_fes:
                break
                
        return best, best_fit, self.convergence_curve
```

---

## Phase 5: 论文写作指南

### 论文结构

| 章节 | 内容要点 |
|------|---------|
| **1. Introduction** | 研究动机 + 现有方法不足 + 本文贡献（3-4 条 bullet） |
| **2. Related Work** | 生物启发算法分类 + 代表性方法回顾 + 现存 gap |
| **3. Biological Inspiration** | 目标生物的生态/行为描述 + 为什么适合做优化算法 |
| **4. Proposed Algorithm** | 数学建模 + 伪代码 + 流程图 + 参数说明 + 计算复杂度 |
| **5. Experimental Results** | 实验设置 + 对比结果 + 统计检验 + 收敛图 + 参数分析 |
| **6. Engineering Applications** | 3-6 个应用 + 对比结果 |
| **7. Discussion** | 算法优势 + 局限性 + 参数敏感性 |
| **8. Conclusion** | 总结 + 未来工作 |

### 常见被拒原因（审稿人关注点）

1. **没有真正的生物连接**：公式和生物行为对不上，挂羊头卖狗肉
2. **对比算法过弱**：只对比经典算法（PSO/GA/GWO），缺少近年先进算法
3. **统计检验缺失**：均值排名就下结论，没有 Wilcoxon/Friedman
4. **没有参数分析**：算法参数怎么取、敏感性如何
5. **"Yet Another Algorithm"**：和已有算法没有本质区别
6. **应用太简单**：只做标准函数，没有真实世界的工程问题
7. **代码未公开**：越来越多期刊要求提交代码

---

## References

### 关键综述论文
- Abualigah, L., et al. (2023). "The arithmetic optimization algorithm: A review." *Archives of Computational Methods in Engineering*.
- Dokeroglu, T., et al. (2019). "A survey on new generation metaheuristic algorithms." *Computers & Industrial Engineering*, 137, 106040.
- Hussain, K., et al. (2019). "Metaheuristic research: a comprehensive survey." *Artificial Intelligence Review*, 52(4), 2191-2233.

### 高引经典算法论文
- Kennedy, J. & Eberhart, R. (1995). PSO. *ICNN*.
- Mirjalili, S., et al. (2014). "Grey Wolf Optimizer." *Advances in Engineering Software*, 69, 46-61.
- Karaboga, D. & Basturk, B. (2007). "Artificial Bee Colony algorithm." *J. Global Optimization*, 39(3), 459-471.

### Benchmark 资源
- CEC 测试套件：https://github.com/P-N-Suganthan/CEC-XXXX
- 代码平台：https://github.com/JerryIreya/Evolutionary-Computation-Best-Practise
