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
    # per-sample oracle ratio truncated to [0,1]
    return np.clip(eA * rB / (rB ** 2 + 1e-9), 0, 1)


X, y, info = generate_s5(n=400, nonlinear=True, seed=0)
phi, dt = info["phi"], info["dt"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

for lam in [1e-4, 0.5, 1.0]:
    ada = AHFE(A=4, gbdt="xgb", lambda_g=lam, random_state=0, gate_mode="adaptive").fit(Xtr, ytr)
    fix = AHFE(A=4, gbdt="xgb", lambda_g=lam, random_state=0, gate_mode="fixed").fit(Xtr, ytr)
    ya, rb, w = ada.predict_with_components(Xte)
    eA = yte - ya                       # true test residual of component A
    w_oracle = oracle_w(eA, rb)
    q_te = (Xte @ phi) * dt
    NL = np.abs(info["lambda_nl"] * (q_te ** 2 - np.mean(q_te ** 2)))
    print(f"lam_g={lam}: fixed={r2(fix.predict(Xte),yte):.4f}  learned_adaptive={r2(ya+w*rb,yte):.4f}  "
          f"ORACLE_adaptive={r2(ya+w_oracle*rb,yte):.4f}  Spearman(oracle_w,NL)={spearmanr(w_oracle,NL).statistic:+.3f}")
    # how well does learned w track oracle w?
    print(f"           corr(learned_w, oracle_w) spearman={spearmanr(w, w_oracle).statistic:+.3f}")
