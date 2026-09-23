import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from scipy.stats import spearmanr
from data_loader import generate_s5
from ahfe import AHFE
from sklearn.model_selection import train_test_split


def r2(pred, yt):
    return 1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)


X, y, info = generate_s5(n=400, nonlinear=True, seed=0)
phi, dt = info["phi"], info["dt"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

for lam in [1e-4, 1e-2, 0.1, 0.5, 1.0]:
    ahfe = AHFE(A=4, gbdt="xgb", lambda_g=lam, random_state=0).fit(Xtr, ytr)
    ya, rb, w = ahfe.predict_with_components(Xte)
    pred = ya + w * rb
    q_te = (Xte @ phi) * dt
    NL = np.abs(info["lambda_nl"] * (q_te ** 2 - np.mean(q_te ** 2)))
    sp, sp_p = spearmanr(w, NL)
    print(f"lam_g={lam:>6}: AHFE R2={r2(pred,yte):.4f}  w mean={w.mean():.3f} w std={w.std():.3f}  Spearman(w,NL)={sp:+.3f} (p={sp_p:.2e})")
