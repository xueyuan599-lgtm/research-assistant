# Phase 5C: 高维可扩展性报告

## 1. 实验设置

- 函数: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank
- 维度: 100D (100K FES), 300D (300K FES), 500D (500K FES)
- 运行次数: 30
- 对比: PNO-GWO-v4.0, GWO, PSO, DE, CMA-ES

## 100D 结果

| 函数 | 算法 | 中位数 | 均值 | 时间(s) |
|------|------|--------|------|--------|
| Sphere | PNO-GWO-v4.0 | 5.359e-42 | 2.909e-40 | 9.36 |
| Sphere | GWO | 9.137e-129 | 2.250e-127 | 1.79 |
| Sphere | PSO | 6.637e+02 | 7.666e+02 | 1.86 |
| Sphere | DE | 2.756e+04 | 2.764e+04 | 7.90 |
| Sphere | CMA-ES | 5.983e-87 | 1.115e-86 | 2.48 |
| Rastrigin | PNO-GWO-v4.0 | 0.000e+00 | 0.000e+00 | 12.95 |
| Rastrigin | GWO | 0.000e+00 | 0.000e+00 | 2.42 |
| Rastrigin | PSO | 3.346e+02 | 3.357e+02 | 2.20 |
| Rastrigin | DE | 1.040e+03 | 1.044e+03 | 7.80 |
| Rastrigin | CMA-ES | 2.557e+02 | 2.586e+02 | 2.74 |
| Rosenbrock | PNO-GWO-v4.0 | 9.838e+01 | 9.840e+01 | 11.73 |
| Rosenbrock | GWO | 9.819e+01 | 9.818e+01 | 2.01 |
| Rosenbrock | PSO | 1.408e+05 | 1.725e+05 | 1.67 |
| Rosenbrock | DE | 2.830e+07 | 3.002e+07 | 5.94 |
| Rosenbrock | CMA-ES | 7.211e+01 | 7.602e+01 | 2.10 |
| Ackley | PNO-GWO-v4.0 | 3.997e-15 | 3.997e-15 | 9.86 |
| Ackley | GWO | 3.997e-15 | 2.813e-15 | 1.98 |
| Ackley | PSO | 5.237e+00 | 5.337e+00 | 1.86 |
| Ackley | DE | 1.594e+01 | 1.613e+01 | 5.88 |
| Ackley | CMA-ES | 2.531e-14 | 2.602e-14 | 2.17 |
| Griewank | PNO-GWO-v4.0 | 0.000e+00 | 2.900e-02 | 9.22 |
| Griewank | GWO | 0.000e+00 | 0.000e+00 | 1.90 |
| Griewank | PSO | 6.973e+00 | 7.899e+00 | 1.78 |
| Griewank | DE | 2.490e+02 | 2.498e+02 | 5.84 |
| Griewank | CMA-ES | 0.000e+00 | 1.150e-03 | 2.09 |
## 300D 结果

| 函数 | 算法 | 中位数 | 均值 | 时间(s) |
|------|------|--------|------|--------|
| Sphere | PNO-GWO-v4.0 | 6.751e-87 | 3.553e-83 | 20.18 |
| Sphere | GWO | 8.420e-260 | 5.744e-255 | 3.27 |
| Sphere | PSO | 1.743e+04 | 1.736e+04 | 3.12 |
| Sphere | DE | 6.704e+04 | 6.767e+04 | 21.05 |
| Sphere | CMA-ES | 2.762e-62 | 2.963e-62 | 31.26 |
| Rastrigin | PNO-GWO-v4.0 | 0.000e+00 | 0.000e+00 | 20.98 |
| Rastrigin | GWO | 0.000e+00 | 0.000e+00 | 3.80 |
| Rastrigin | PSO | 1.520e+03 | 1.531e+03 | 3.96 |
| Rastrigin | DE | 3.177e+03 | 3.172e+03 | 22.11 |
| Rastrigin | CMA-ES | 9.795e+02 | 9.839e+02 | 54.65 |
| Rosenbrock | PNO-GWO-v4.0 | 2.973e+02 | 2.973e+02 | 19.18 |
| Rosenbrock | GWO | 2.966e+02 | 2.966e+02 | 3.72 |
| Rosenbrock | PSO | 9.122e+06 | 9.387e+06 | 3.52 |
| Rosenbrock | DE | 8.066e+07 | 8.572e+07 | 21.46 |
| Rosenbrock | CMA-ES | 2.727e+02 | 2.814e+02 | 35.65 |
| Ackley | PNO-GWO-v4.0 | 3.997e-15 | 3.523e-15 | 20.84 |
| Ackley | GWO | 3.997e-15 | 3.523e-15 | 4.13 |
| Ackley | PSO | 1.002e+01 | 1.008e+01 | 4.43 |
| Ackley | DE | 2.002e+01 | 1.887e+01 | 21.89 |
| Ackley | CMA-ES | 6.084e-14 | 5.966e-14 | 35.28 |
| Griewank | PNO-GWO-v4.0 | 0.000e+00 | 0.000e+00 | 20.16 |
| Griewank | GWO | 0.000e+00 | 0.000e+00 | 4.06 |
| Griewank | PSO | 1.665e+02 | 1.678e+02 | 4.28 |
| Griewank | DE | 6.144e+02 | 6.188e+02 | 22.20 |
| Griewank | CMA-ES | 4.441e-16 | 9.855e-04 | 35.51 |
## 500D 结果

| 函数 | 算法 | 中位数 | 均值 | 时间(s) |
|------|------|--------|------|--------|
| Sphere | PNO-GWO-v4.0 | 1.054e-129 | 3.264e-123 | 31.29 |
| Sphere | GWO | 0.000e+00 | 0.000e+00 | 5.14 |
| Sphere | PSO | 5.983e+04 | 6.265e+04 | 5.08 |
| Sphere | DE | 1.020e+05 | 1.025e+05 | 45.93 |
| Sphere | CMA-ES | 3.300e-58 | 4.130e-58 | 340.48 |
| Rastrigin | PNO-GWO-v4.0 | 0.000e+00 | 0.000e+00 | 32.86 |
| Rastrigin | GWO | 0.000e+00 | 0.000e+00 | 5.79 |
| Rastrigin | PSO | 2.873e+03 | 2.883e+03 | 6.79 |
| Rastrigin | DE | 5.275e+03 | 5.260e+03 | 47.05 |
| Rastrigin | CMA-ES | 1.898e+03 | 1.869e+03 | 340.97 |
| Rosenbrock | PNO-GWO-v4.0 | 4.961e+02 | 4.961e+02 | 31.74 |
| Rosenbrock | GWO | 4.950e+02 | 4.951e+02 | 5.73 |
| Rosenbrock | PSO | 4.251e+07 | 4.283e+07 | 5.96 |
| Rosenbrock | DE | 3.472e+08 | 2.831e+09 | 46.07 |
| Rosenbrock | CMA-ES | 4.711e+02 | 4.819e+02 | 366.55 |
| Ackley | PNO-GWO-v4.0 | 3.997e-15 | 3.168e-15 | 45.35 |
| Ackley | GWO | 3.997e-15 | 2.813e-15 | 8.84 |
| Ackley | PSO | 1.276e+01 | 1.268e+01 | 9.92 |
| Ackley | DE | 1.562e+01 | 1.647e+01 | 62.85 |
| Ackley | CMA-ES | 9.281e-14 | 2.313e-01 | 397.11 |
| Griewank | PNO-GWO-v4.0 | 0.000e+00 | 0.000e+00 | 33.05 |
| Griewank | GWO | 0.000e+00 | 0.000e+00 | 6.38 |
| Griewank | PSO | 5.362e+02 | 5.647e+02 | 7.65 |
| Griewank | DE | 9.194e+02 | 9.239e+02 | 48.11 |
| Griewank | CMA-ES | 1.332e-15 | 1.068e-03 | 341.87 |

## 性能衰减趋势

| 算法 | 100D→300D (log10变化) | 300D→500D (log10变化) |
|------|----------------------|----------------------|
| PNO-GWO-v4.0 | -8.88 | -8.52 |
| GWO | -26.11 | -8.14 |
| PSO | +1.11 | +0.42 |
| DE | +0.36 | +0.22 |
| CMA-ES | +62.17 | +1.05 |

![适应度](figures/scalability_fitness.png)

![时间](figures/scalability_time.png)