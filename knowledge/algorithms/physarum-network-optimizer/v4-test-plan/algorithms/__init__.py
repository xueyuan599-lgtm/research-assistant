"""
Phase 4/5 对比算法实现
======================
每个算法遵循统一接口:
    optimize(obj_func, n_dim, bounds, max_fes) -> (best_x, best_fit, convergence)
"""

from .gwo import GWO
from .pso import PSO
from .de import DE
from .shade import SHADE
from .cmaes import CMAES
from .lshade_cneeso import LSHADEcnEPSO
from .jso import JSO
from .ea4eig import EA4eig
from .cmode import CMODE

__all__ = ['GWO', 'PSO', 'DE', 'SHADE', 'CMAES',
           'LSHADEcnEPSO', 'JSO', 'EA4eig', 'CMODE']
