# 函数型数据预处理参考

> 涵盖函数型数据的平滑去噪、基展开、SRSF 弹性对齐、FPCA 降维。

---

## 1. 函数型数据表示

```python
from skfda import datasets, FDataGrid
import numpy as np

t = np.linspace(0, 1, 101)
X = np.random.randn(50, 101)
fd = FDataGrid(X, grid_points=t)

fetch_growth = datasets.fetch_growth()
fd_growth = fetch_growth["data"]
```

---

## 2. 平滑去噪

### KernelSmoother

```python
from skfda.preprocessing.smoothing import KernelSmoother

ks = KernelSmoother(kernel_estimator="normal", bandwidth=0.5)
fd_smooth = ks.fit_transform(fd)
```

### BasisSmoother（基平滑）

```python
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import BSpline, Fourier

basis_bs = BSpline(n_basis=15, order=4)
bs = BasisSmoother(basis_bs, smoothing_parameter=1e-3)
fd_smooth = bs.fit_transform(fd)

basis_fourier = Fourier(n_basis=21)
bs_fourier = BasisSmoother(basis_fourier, smoothing_parameter=1e-4)
fd_smooth_fourier = bs_fourier.fit_transform(fd)
```

### 带宽选择

```python
for bw in [0.05, 0.2, 0.5, 1.0]:
    ks = KernelSmoother(bandwidth=bw)
    fd_s = ks.fit_transform(fd)
    mse = np.mean((fd_s.data_matrix - fd.data_matrix)**2)
    print(f"bw={bw:.2f}, MSE={mse:.6f}")
```

---

## 3. 基展开

```python
from skfda.representation.basis import BSpline, Fourier

basis_bs = BSpline(n_basis=15, order=4, domain_range=(0, 1))
fd_basis = fd.to_basis(basis_bs)

basis_fourier = Fourier(n_basis=21, period=1.0)
fd_basis_fourier = fd.to_basis(basis_fourier)

fd_fitted = fd_basis.to_grid()
ss_res = np.sum((fd.data_matrix - fd_fitted.data_matrix)**2)
ss_tot = np.sum((fd.data_matrix - fd.data_matrix.mean(axis=0))**2)
r2 = 1 - ss_res / ss_tot
print(f"R2 = {r2:.4f}")
```

---

## 4. SRSF 弹性对齐（Registration）

```python
from skfda.datasets import fetch_berkeley
import fdasrsf as fs
import numpy as np

data = fetch_berkeley()["data"].data_matrix[:, :, 0]
time = np.linspace(0, 1, data.shape[1])

warp = fs.fdawarp(data.T, time)
warp.srsf_align(method="mean", omethod="DP2", MaxItr=20)

aligned = warp.f_aligned.T
original = warp.f.T
var_before = np.var(original, axis=0).mean()
var_after = np.var(aligned, axis=0).mean()
print(f"Variance: {var_before:.4f} -> {var_after:.4f}")
```

---

## 5. FPCA 降维

```python
from skfda.preprocessing.dim_reduction import FPCA

fpca = FPCA(n_components=5, centering=True)
fpca.fit(fd_smooth)
scores = fpca.transform(fd_smooth)

cumulative = np.cumsum(fpca.explained_variance_ratio_)
print(f"Cumulative: {cumulative}")

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(scores, y)
```

---

## 6. 导数曲线

```python
fd_d1 = fd_smooth.derivative(order=1)
fd_d2 = fd_smooth.derivative(order=2)
velocities = fd_d1.data_matrix[:, :, 0]
accelerations = fd_d2.data_matrix[:, :, 0]
```

---

## 7. 函数型异常检测

```python
import fdasrsf as fs

data_matrix = fd.data_matrix[:, :, 0]
time = fd.grid_points[0]

warp = fs.fdawarp(data_matrix.T, time)
warp.srsf_align()
outliers = fs.outlier_detection(warp.f_aligned, time, method="bootstrap")
print(f"Outliers: {outliers['outliers']}")
```

---

## 参考

- [scikit-fda 文档](https://fda.readthedocs.io/)
- [fdasrsf 文档](https://fdasrsf-python.readthedocs.io/)
