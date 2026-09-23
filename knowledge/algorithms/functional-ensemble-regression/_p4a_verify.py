import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from data_loader import generate_s5
from simpls import SIMPLS
from ahfe import AHFE
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


def rmse_r2(pred, yt):
    rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
    ss = float(np.sum((yt - yt.mean()) ** 2))
    return rmse, 1 - float(np.sum((pred - yt) ** 2)) / ss


# 1) DGP property check
X, y, info = generate_s5(n=300, nonlinear=True, seed=0)
C = np.cov(X.T)
beta, phi = info["beta"], info["phi"]
orth = float(beta @ (C @ phi) / max(beta @ (C @ beta), 1e-12))
var_lin = float(((X @ beta * info["dt"]) ** 2).mean())
nl_pow = float((info["nl"] ** 2).mean())
print("DGP: <phi,C*beta>/<beta,C*beta>=", round(orth, 4),
      " nonlinear/linear power=", round(nl_pow / var_lin, 3),
      " lambda=", round(info["lambda_nl"], 3))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

# 2) AHFE vs SIMPLS on nonlinear S5
base_r = SIMPLS(4).fit(Xtr, ytr).predict(Xte)
ahfe = AHFE(A=4, gbdt="xgb", random_state=0).fit(Xtr, ytr)
ya, rb, w = ahfe.predict_with_components(Xte)
pred = ya + w * rb
r_base, r2_base = rmse_r2(base_r, yte)
r_ahfe, r2_ahfe = rmse_r2(pred, yte)
print(f"S5 nonlinear: SIMPLS R2={r2_base:.4f}  AHFE R2={r2_ahfe:.4f}  deltaR2={r2_ahfe - r2_base:+.4f}")

q_test = (Xte @ phi) * info["dt"]
NL_test = np.abs(info["lambda_nl"] * (q_test ** 2 - np.mean(q_test ** 2)))
sp, sp_p = spearmanr(w, NL_test)
print(f"  Spearman(w,NL)={sp:.4f} (p={sp_p:.2e})  w range [{w.min():.3f},{w.max():.3f}] mean {w.mean():.3f}")

# 3) S5b linear control: w -> 0, no gain
Xb, yb, _ = generate_s5(n=300, nonlinear=False, seed=1)
Xtrb, Xteb, ytrb, yteb = train_test_split(Xb, yb, test_size=0.3, random_state=0)
ahfe_b = AHFE(A=4, gbdt="xgb", random_state=0).fit(Xtrb, ytrb)
ya_b, rb_b, w_b = ahfe_b.predict_with_components(Xteb)
pred_b = ya_b + w_b * rb_b
r_base_b = SIMPLS(4).fit(Xtrb, ytrb).predict(Xteb)
_, r2_base_b = rmse_r2(r_base_b, yteb)
_, r2_ahfe_b = rmse_r2(pred_b, yteb)
print(f"S5b linear: SIMPLS R2={r2_base_b:.4f}  AHFE R2={r2_ahfe_b:.4f}  w mean={w_b.mean():.4f}")

# 4) beta recovery on nonlinear S5
b = SIMPLS(4).fit(Xtr, ytr).beta_
dt = info["dt"]
l2 = float(np.sqrt(np.sum((b - beta * dt) ** 2)))
print(f"beta recovery L2 (SIMPLS, nonlinear) = {l2:.4f}")
