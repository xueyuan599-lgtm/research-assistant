"""
约束处理机制改进
解决 Phase 2 中工程问题陷入不可行区域的问题
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


class ConstraintHandler:
    """约束处理器：支持多种约束处理策略"""

    def __init__(self, method: str = 'adaptive_penalty'):
        """
        初始化约束处理器

        Args:
            method: 约束处理方法
                - 'penalty': 传统罚函数
                - 'adaptive_penalty': 自适应罚函数
                - 'feasibility_rule': 可行性规则
                - 'epsilon_constraint': ε-约束
                - 'stochastic_ranking': 随机排序
        """
        self.method = method
        self.penalty_factor = 1.0
        self.violation_history = []
        self.epsilon = 1e-6
        self.epsilon_decay = 0.95

    def evaluate_constraints(self, x: np.ndarray,
                            constraints: List[Callable]) -> Tuple[float, float]:
        """
        评估约束违反程度

        Args:
            x: 决策变量
            constraints: 约束函数列表（返回 <=0 为满足）

        Returns:
            total_violation: 总违反量
            num_violated: 违反约束数量
        """
        total_violation = 0.0
        num_violated = 0

        for constraint in constraints:
            violation = constraint(x)
            if violation > 0:  # 违反约束
                total_violation += violation
                num_violated += 1

        return total_violation, num_violated

    def penalty_function(self, obj_value: float, violation: float) -> float:
        """传统罚函数"""
        return obj_value + self.penalty_factor * violation

    def adaptive_penalty(self, obj_value: float, violation: float,
                        iteration: int, max_iterations: int) -> float:
        """
        自适应罚函数
        惩罚因子随迭代增加，早期允许探索不可行区域，后期强制可行
        """
        # 惩罚因子随迭代增加
        progress = iteration / max_iterations
        penalty_multiplier = 1.0 + 10.0 * progress  # 1 -> 11

        # 根据违反程度调整
        if violation > 0:
            penalty = self.penalty_factor * penalty_multiplier * violation
        else:
            penalty = 0.0

        return obj_value + penalty

    def feasibility_rule(self, obj1: float, viol1: float,
                        obj2: float, viol2: float) -> bool:
        """
        可行性规则（Deb's rules）
        1. 可行解优于不可行解
        2. 两个不可行解，违反量小的更优
        3. 两个可行解，目标函数值小的更优
        """
        # 规则1: 可行解优于不可行解
        if viol1 == 0 and viol2 > 0:
            return True
        if viol1 > 0 and viol2 == 0:
            return False

        # 规则2: 两个不可行解，违反量小的更优
        if viol1 > 0 and viol2 > 0:
            return viol1 < viol2

        # 规则3: 两个可行解，目标函数值小的更优
        return obj1 < obj2

    def epsilon_constraint(self, obj_value: float, violation: float,
                          iteration: int) -> float:
        """
        ε-约束方法
        允许一定程度的约束违反，ε随迭代减小
        """
        # ε随迭代减小
        current_epsilon = self.epsilon * (self.epsilon_decay ** iteration)

        if violation <= current_epsilon:
            # 满足ε-约束，返回原目标函数
            return obj_value
        else:
            # 违反ε-约束，施加惩罚
            return obj_value + self.penalty_factor * (violation - current_epsilon)

    def stochastic_ranking(self, obj1: float, viol1: float,
                          obj2: float, viol2: float,
                          pf: float = 0.45) -> bool:
        """
        随机排序（Stochastic Ranking）
        以概率pf比较目标函数，以概率1-pf比较约束违反
        """
        if viol1 == 0 and viol2 == 0:
            # 都可行，比较目标函数
            return obj1 < obj2

        if np.random.random() < pf:
            # 比较目标函数
            return obj1 < obj2
        else:
            # 比较约束违反
            return viol1 < viol2

    def select_better(self, x1: np.ndarray, obj1: float, viol1: float,
                     x2: np.ndarray, obj2: float, viol2: float,
                     iteration: int = 0, max_iterations: int = 1000) -> Tuple[np.ndarray, float, float]:
        """
        根据约束处理方法选择更优解

        Args:
            x1, obj1, viol1: 解1及其目标函数值和违反量
            x2, obj2, viol2: 解2及其目标函数值和违反量
            iteration: 当前迭代
            max_iterations: 最大迭代

        Returns:
            更优的解、目标函数值、违反量
        """
        if self.method == 'penalty':
            f1 = self.penalty_function(obj1, viol1)
            f2 = self.penalty_function(obj2, viol2)
            if f1 <= f2:
                return x1, obj1, viol1
            else:
                return x2, obj2, viol2

        elif self.method == 'adaptive_penalty':
            f1 = self.adaptive_penalty(obj1, viol1, iteration, max_iterations)
            f2 = self.adaptive_penalty(obj2, viol2, iteration, max_iterations)
            if f1 <= f2:
                return x1, obj1, viol1
            else:
                return x2, obj2, viol2

        elif self.method == 'feasibility_rule':
            if self.feasibility_rule(obj1, viol1, obj2, viol2):
                return x1, obj1, viol1
            else:
                return x2, obj2, viol2

        elif self.method == 'epsilon_constraint':
            f1 = self.epsilon_constraint(obj1, viol1, iteration)
            f2 = self.epsilon_constraint(obj2, viol2, iteration)
            if f1 <= f2:
                return x1, obj1, viol1
            else:
                return x2, obj2, viol2

        elif self.method == 'stochastic_ranking':
            if self.stochastic_ranking(obj1, viol1, obj2, viol2):
                return x1, obj1, viol1
            else:
                return x2, obj2, viol2

        else:
            raise ValueError(f"Unknown method: {self.method}")


class EngineeringProblemWithConstraints:
    """带约束的工程问题包装器"""

    def __init__(self, name: str, n_dim: int,
                 obj_func: Callable,
                 constraints: List[Callable],
                 bounds: List[Tuple[float, float]],
                 known_optimum: Optional[float] = None):
        """
        初始化工程问题

        Args:
            name: 问题名称
            n_dim: 变量维度
            obj_func: 目标函数
            constraints: 约束函数列表（返回 <=0 为满足）
            bounds: 变量边界
            known_optimum: 已知最优解（用于对比）
        """
        self.name = name
        self.n_dim = n_dim
        self.obj_func = obj_func
        self.constraints = constraints
        self.bounds = bounds
        self.known_optimum = known_optimum

    def evaluate(self, x: np.ndarray) -> Tuple[float, float]:
        """
        评估解（目标函数 + 约束违反）

        Returns:
            obj_value: 目标函数值
            violation: 总约束违反量
        """
        obj_value = self.obj_func(x)

        total_violation = 0.0
        for constraint in self.constraints:
            violation = constraint(x)
            if violation > 0:
                total_violation += violation

        return obj_value, total_violation

    def is_feasible(self, x: np.ndarray) -> bool:
        """检查解是否可行"""
        _, violation = self.evaluate(x)
        return violation == 0

    def gap_to_optimum(self, obj_value: float) -> Optional[float]:
        """与已知最优解的差距（百分比）"""
        if self.known_optimum is None:
            return None
        return (obj_value - self.known_optimum) / abs(self.known_optimum) * 100


# ============================================================
# 经典工程问题定义（带约束）
# ============================================================

def welded_beam_problem() -> EngineeringProblemWithConstraints:
    """焊接梁设计问题"""
    n_dim = 4

    def obj_func(x):
        h, l, t, b = x
        return 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)

    def constraint1(x):
        h, l, t, b = x
        tau_max = 13600
        P = 6000
        L = 14
        tau1 = P / (np.sqrt(2) * h * l)
        M = P * (L + l / 2)
        R = np.sqrt(l**2 / 4 + ((h + t) / 2)**2)
        J = 2 * (h * l * np.sqrt(2) * (l**2 / 12 + ((h + t) / 2)**2))
        tau2 = M * R / J
        tau = np.sqrt(tau1**2 + 2 * tau1 * tau2 * l / (2 * R) + tau2**2)
        return tau - tau_max

    def constraint2(x):
        h, l, t, b = x
        sigma_max = 30000
        P = 6000
        L = 14
        sigma = 6 * P * L / (b * t**2)
        return sigma - sigma_max

    def constraint3(x):
        h, l, t, b = x
        return h - b

    def constraint4(x):
        h, l, t, b = x
        P = 6000
        L = 14
        E = 30e6
        Pc = 4.013 * E * np.sqrt(t**6 * b**2 / 36) / (L**2) * (1 - t / (2 * L) * np.sqrt(E / (4 * 1.2e6)))
        return P - Pc

    bounds = [(0.125, 5.0), (0.1, 10.0), (0.1, 10.0), (0.125, 5.0)]

    return EngineeringProblemWithConstraints(
        name='welded_beam',
        n_dim=n_dim,
        obj_func=obj_func,
        constraints=[constraint1, constraint2, constraint3, constraint4],
        bounds=bounds,
        known_optimum=1.7248
    )


def pressure_vessel_problem() -> EngineeringProblemWithConstraints:
    """压力容器设计问题"""
    n_dim = 4

    def obj_func(x):
        Ts, Th, R, L = x
        return 0.6224 * Ts * R * L + 1.7781 * Th * R**2 + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R

    def constraint1(x):
        Ts, Th, R, L = x
        return -Ts + 0.0193 * R

    def constraint2(x):
        Ts, Th, R, L = x
        return -Th + 0.00954 * R

    def constraint3(x):
        Ts, Th, R, L = x
        return -np.pi * R**2 * L - (4/3) * np.pi * R**3 + 1296000

    def constraint4(x):
        Ts, Th, R, L = x
        return L - 240

    bounds = [(0.0625, 6.1875), (0.0625, 6.1875), (10.0, 200.0), (10.0, 240.0)]

    return EngineeringProblemWithConstraints(
        name='pressure_vessel',
        n_dim=n_dim,
        obj_func=obj_func,
        constraints=[constraint1, constraint2, constraint3, constraint4],
        bounds=bounds,
        known_optimum=6059.71
    )


def spring_problem() -> EngineeringProblemWithConstraints:
    """拉压弹簧设计问题"""
    n_dim = 3

    def obj_func(x):
        d, D, N = x
        return (N + 2) * D * d**2

    def constraint1(x):
        d, D, N = x
        return 1 - (D**3 * N) / (71785 * d**4)

    def constraint2(x):
        d, D, N = x
        return (4 * D**2 - d * D) / (12566 * (D * d**3 - d**4)) + 1 / (5108 * d**2) - 1

    def constraint3(x):
        d, D, N = x
        return 1 - (140.45 * d) / (D**2 * N)

    def constraint4(x):
        d, D, N = x
        return (D + d) / 1.5 - 1

    bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]

    return EngineeringProblemWithConstraints(
        name='spring',
        n_dim=n_dim,
        obj_func=obj_func,
        constraints=[constraint1, constraint2, constraint3, constraint4],
        bounds=bounds,
        known_optimum=0.012665
    )


def speed_reducer_problem() -> EngineeringProblemWithConstraints:
    """减速器设计问题"""
    n_dim = 7

    def obj_func(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934) -
                1.508 * x1 * (x6**2 + x7**2) + 7.477 * (x6**3 + x7**3) +
                0.7854 * (x4 * x6**2 + x5 * x7**2))

    def constraint1(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return 27 / (x1 * x2**2 * x3) - 1

    def constraint2(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return 397.5 / (x1 * x2**2 * x3**2) - 1

    def constraint3(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return 1.93 * x4**3 / (x2 * x3 * x6**4) - 1

    def constraint4(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return 1.93 * x5**3 / (x2 * x3 * x7**4) - 1

    def constraint5(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return np.sqrt((745 * x4 / (x2 * x3))**2 + 16.9e6) / (110 * x6**3) - 1

    def constraint6(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return np.sqrt((745 * x5 / (x2 * x3))**2 + 157.5e6) / (85 * x7**3) - 1

    def constraint7(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return x2 * x3 / 40 - 1

    def constraint8(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return 5 * x2 / x1 - 1

    def constraint9(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return x1 / (12 * x2) - 1

    def constraint10(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return (1.5 * x6 + 1.9) / x4 - 1

    def constraint11(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return (1.1 * x7 + 1.9) / x5 - 1

    bounds = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5.0, 5.5)]

    return EngineeringProblemWithConstraints(
        name='speed_reducer',
        n_dim=n_dim,
        obj_func=obj_func,
        constraints=[constraint1, constraint2, constraint3, constraint4, constraint5,
                    constraint6, constraint7, constraint8, constraint9, constraint10, constraint11],
        bounds=bounds,
        known_optimum=2994.42
    )


def three_bar_truss_problem() -> EngineeringProblemWithConstraints:
    """三杆桁架问题"""
    n_dim = 2

    def obj_func(x):
        x1, x2 = x
        return (2 * np.sqrt(2) * x1 + x2) * 100

    def constraint1(x):
        x1, x2 = x
        return (np.sqrt(2) * x1 + x2) / (np.sqrt(2) * x1**2 + 2 * x1 * x2) - 1

    def constraint2(x):
        x1, x2 = x
        return x2 / (np.sqrt(2) * x1**2 + 2 * x1 * x2) - 1

    def constraint3(x):
        x1, x2 = x
        return 1 / (np.sqrt(2) * x2 + x1) - 1

    bounds = [(0.001, 1.0), (0.001, 1.0)]

    return EngineeringProblemWithConstraints(
        name='three_bar_truss',
        n_dim=n_dim,
        obj_func=obj_func,
        constraints=[constraint1, constraint2, constraint3],
        bounds=bounds,
        known_optimum=263.896
    )


def get_all_engineering_problems() -> dict:
    """获取所有工程问题"""
    return {
        'welded_beam': welded_beam_problem(),
        'pressure_vessel': pressure_vessel_problem(),
        'spring': spring_problem(),
        'speed_reducer': speed_reducer_problem(),
        'three_bar_truss': three_bar_truss_problem(),
    }
