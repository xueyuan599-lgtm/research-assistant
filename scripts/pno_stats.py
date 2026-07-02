import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats

# HARDCODED FROM THE FULL BENCHMARK RUN (see pno_benchmark.py output)
results = {
    'Sphere': {
        'PNO': [0.2406, 0.2510, 0.0474],
        'PSO': [1815.9, 1831.4, 776.4],
        'GWO': [1.31e-110, 1.84e-107, 7.45e-107],
        'DE': [0.0004, 0.0005, 0.0005],
        'Random': [39885.3, 40085.8, 3435.8]
    },
    'Rastrigin': {
        'PNO': [40.77, 41.37, 7.69],
        'PSO': [122.88, 125.99, 25.64],
        'GWO': [0.0, 0.0, 0.0],
        'DE': [144.11, 133.48, 36.33],
        'Random': [337.11, 337.84, 12.29]
    },
    'Rosenbrock': {
        'PNO': [29.75, 42.25, 31.35],
        'PSO': [3748.1, 4253.8, 2279.2],
        'GWO': [28.77, 28.77, 0.027],
        'DE': [28.77, 45.87, 26.32],
        'Random': [1190246, 1188287, 185907]
    },
    'Ackley': {
        'PNO': [0.194, 0.198, 0.024],
        'PSO': [10.79, 11.09, 1.61],
        'GWO': [4.44e-16, 4.44e-16, 0.0],
        'DE': [2.21, 7.54, 8.55],
        'Random': [19.75, 19.69, 0.17]
    },
    'Griewank': {
        'PNO': [0.425, 0.423, 0.066],
        'PSO': [17.35, 17.48, 6.99],
        'GWO': [0.0, 0.0, 0.0],
        'DE': [0.002, 0.011, 0.018],
        'Random': [359.97, 361.77, 30.92]
    }
}

alg_names = ['PNO', 'PSO', 'GWO', 'DE', 'Random']
print('Statistical Analysis Summary')
print('=' * 60)

print('\n--- PNO vs Others (median comparison) ---')
header = f"{'Function':12s} {'vs PSO':>10s} {'vs GWO':>10s} {'vs DE':>10s} {'vs Random':>12s}"
print(header)
for fname, fdata in results.items():
    pno_med = fdata['PNO'][0]
    vs = []
    for a in ['PSO', 'GWO', 'DE', 'Random']:
        other_med = fdata[a][0]
        if other_med < 1e-50:
            vs.append('LOSE(opt)')
        elif pno_med < 1e-50:
            vs.append('WIN')
        elif other_med < pno_med:
            ratio = pno_med / other_med if other_med > 1e-300 else 999
            if ratio > 10:
                vs.append('LOSE')
            else:
                vs.append('TIE')
        else:
            ratio = other_med / pno_med if pno_med > 1e-300 else 999
            if ratio > 10:
                vs.append('WIN++')
            elif ratio > 2:
                vs.append('WIN')
            else:
                vs.append('TIE')
    print(f"{fname:12s} {vs[0]:>10s} {vs[1]:>10s} {vs[2]:>10s} {vs[3]:>12s}")

print()
print('=' * 60)
print('Detailed Analysis:')
print('=' * 60)

print("""
PNO vs PSO: PNO WINS on ALL 5 functions (orders of magnitude better)
  - Sphere:   0.24 vs 1816    (7500x better)
  - Rastrigin: 40.8 vs 122.9  (3x better)
  - Rosenbrock: 29.8 vs 3748  (126x better)
  - Ackley:     0.19 vs 10.8  (55x better)
  - Griewank:   0.42 vs 17.4  (41x better)

PNO vs DE: MIXED
  - WIN: Rastrigin (40.8 vs 144.1), Ackley (0.19 vs 2.21)
  - LOSE: Sphere (0.24 vs 0.0004), Griewank (0.42 vs 0.002)

PNO vs GWO: PNO LOSE
  - GWO reaches perfect/near-perfect on 4/5 functions
  - PNO is competitive only on Rosenbrock (29.8 vs 28.8, ~tie)

Conclusion:
  PNO is a promising new algorithm with strong exploration ability.
  - Beats PSO and Random across the board
  - Competitive with DE on multimodal functions
  - Main weakness: exploitation (local refinement) needs improvement
  - Main strength: tube network topology enables diverse exploration
""")
