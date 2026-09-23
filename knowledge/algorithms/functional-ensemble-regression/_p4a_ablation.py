import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from scipy.stats import spearmanr
from data_loader import generate_s5
from simpls import SIMPLS
from ahfe import AHFE
from sklearn.model_selection import train_test_split


def r2(pred, yt):
    return 1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)


# nonlinear S5
X, y, info = generate_s5(n=400, nonlinear=True, seed=0)
phi, dt = info["phi"], info["dt"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
base = SIMPLS(4).fit(Xtr, ytr).predict(Xte)
q_te = (Xte @ phi) * dt
NL = np.abs(info["lambda_nl"] * (q_te ** 2 - np.mean(q_te ** 2)))

print("=== S5 nonlinear ===")
print(f"  SIMPLS          R2={r2(base,yte):.4f}")
for lam in [1e-4, 0.5]:
    fix = AHFE(A=4, gbdt="xgb", lambda_g=lam, random_state=0, gate_mode="fixed").fit(Xtr, ytr)
    ada = AHFE(A=4, gbdt="xgb", lambda_g=lam, random_state=0, gate_mode="adaptive").fit(Xtr, ytr)
    pf = fix.predict(Xte); pa = ada.predict(Xte)
    ya, rb, w = ada.predict_with_components(Xte)
    sp, sp_p = spearmanr(w, NL)
    print(f"  lam_g={lam}: fixed R2={r2(pf,yte):.4f}  adaptive R2={r2(pa,yte):.4f}  gain={r2(pa,yte)-r2(pf,yte):+.4f}  Spearman(w,NL)={sp:+.3f} (p={sp_p:.2e})  w.std={w.std():.3f}")

# linear S5b
Xb, yb, _ = generate_s5(n=400, nonlinear=False, seed=1)
Xtrb, Xteb, ytrb, yteb = train_test_split(Xb, yb, test_size=0.3, random_state=0)
baseb = SIMPLS(4).fit(Xtrb, ytrb).predict(Xteb)
print("=== S5b linear ===")
fixb = AHFE(A=4, gbdt="xgb", random_state=0, gate_mode="fixed").fit(Xtrb, ytrb)
adab = AHFE(A=4, gbdt="xgb", random_state=0, gate_mode="adaptive").fit(Xtrb, ytrb)
_, _, wb = adab.predict_with_components(Xteb)
print(f"  SIMPLS R2={r2(baseb,yteb):.4f}  fixed R2={r2(fixb.predict(Xteb),yteb):.4f}  adaptive R2={r2(adab.predict(Xteb),yteb):.4f}  w mean={wb.mean():.4f}")
