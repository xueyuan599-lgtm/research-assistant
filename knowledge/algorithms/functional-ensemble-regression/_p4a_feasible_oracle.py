import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from scipy.stats import spearmanr
from data_loader import generate_s5
from ahfe import AHFE
from sklearn.model_selection import train_test_split


def r2(pred, yt):
    return 1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)


def oracle_w(eA, rB):
    return np.clip(eA * rB / (rB ** 2 + 1e-9), 0, 1)


X, y, info = generate_s5(n=500, nonlinear=True, seed=0)
phi, dt = info["phi"], info["dt"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
fix = AHFE(A=4, gbdt="xgb", random_state=0, gate_mode="fixed").fit(Xtr, ytr)
ada = AHFE(A=4, gbdt="xgb", lambda_g=1.0, random_state=0).fit(Xtr, ytr)
ya, rb, wl = ada.predict_with_components(Xte)
eA = yte - ya
w_orc = oracle_w(eA, rb)

# FEASIBLE oracle: E[w* | |rB|], smoothed via bins (uses only prediction-time |rB|)
bins = np.percentile(np.abs(rb), np.arange(0, 101, 10))
binid = np.clip(np.digitize(np.abs(rb), bins[1:-1]), 0, 9)
wz = np.array([w_orc[binid == b].mean() for b in range(10)])
w_feas = wz[binid]
q_te = (Xte @ phi) * dt
NL = np.abs(info["lambda_nl"] * (q_te ** 2 - np.mean(q_te ** 2)))

print(f"fixed                 R2={r2(fix.predict(Xte),yte):.4f}")
print(f"learned adaptive      R2={r2(ya+wl*rb,yte):.4f}")
print(f"FEASIBLE oracle(by|rB|) R2={r2(ya+w_feas*rb,yte):.4f}")
print(f"ORACLE(with-y)          R2={r2(ya+w_orc*rb,yte):.4f}")
print(f"E[w*| |rB|] per bin: {np.round(wz,2)}")
print(f"Spearman(w_feas,NL)={spearmanr(w_feas,NL).statistic:+.3f}")
# is the feasible w monotone in |rB|? (adaptive direction)
print(f"corr(w_feas, |rB|)={np.corrcoef(w_feas,np.abs(rb))[0,1]:+.3f}")
