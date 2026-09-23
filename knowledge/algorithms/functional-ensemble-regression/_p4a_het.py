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


# Heterogeneous-regime DGP: nonlinear term active only when |q| > tau.
def gen_het(n, p=100, snr=3.0, seed=0, frac_nl=0.5, length_scale=0.35):
    X, y, info = generate_s5(n=n, p=p, snr=snr, seed=seed, nonlinear=False,
                             length_scale=length_scale)
    # rebuild with regime switch: nonlinearity only for large |q|
    grid = info["grid"]; dt = info["dt"]
    q = info["q"]  # already computed? no - regenerate below
    # recompute q from X and phi
    q = (X @ info["phi"]) * dt
    tau = np.quantile(np.abs(q), 1 - frac_nl)
    lin = (X @ info["beta"]) * dt
    qc = q ** 2 - np.mean(q ** 2)
    nl_raw = np.where(np.abs(q) > tau, qc, 0.0)
    # scale so nonlinear subset has comparable power to linear
    nl_raw = nl_raw - nl_raw.mean()
    var_lin = lin.var()
    var_nl = nl_raw.var()
    lam = np.sqrt(1.0 * var_lin / var_nl) if var_nl > 0 else 0
    signal = lin + lam * nl_raw
    var_s = signal.var()
    rng = np.random.default_rng(seed + 1)
    eps = rng.standard_normal(n)
    eps = eps / (eps.std() + 1e-12) * np.sqrt(var_s / snr ** 2)
    y2 = 5.0 + signal + eps
    info2 = dict(beta=info["beta"], phi=info["phi"], grid=info["grid"],
                 dt=info["dt"], snr=snr, nonlinear=True,
                 nl=np.abs(lam * nl_raw), lambda_nl=lam, tau=tau,
                 nl_mask=(np.abs(q) > tau))
    return X, y2, info2


X, y, info = gen_het(500, seed=0)
print("frac nonlinear subset:", round(float(info["nl_mask"].mean()), 3),
      " lambda:", round(info["lambda_nl"], 2))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
fix = AHFE(A=4, gbdt="xgb", random_state=0, gate_mode="fixed").fit(Xtr, ytr)
for lam_g in [1e-4, 0.1, 1.0]:
    ada = AHFE(A=4, gbdt="xgb", lambda_g=lam_g, random_state=0).fit(Xtr, ytr)
    ya, rb, w = ada.predict_with_components(Xte)
    eA = yte - ya
    w_orc = oracle_w(eA, rb)
    NL_te = info["nl"][len(Xtr):]
    print(f"lam_g={lam_g}: fixed={r2(fix.predict(Xte),yte):.4f}  adaptive={r2(ya+w*rb,yte):.4f}  "
          f"oracle={r2(ya+w_orc*rb,yte):.4f}  Spearman(w,NL)={spearmanr(w,NL_te).statistic:+.3f}")
# feasible oracle by |rB|
ada = AHFE(A=4, gbdt="xgb", lambda_g=1.0, random_state=0).fit(Xtr, ytr)
ya, rb, w = ada.predict_with_components(Xte); eA = yte - ya
w_orc = oracle_w(eA, rb)
bins = np.percentile(np.abs(rb), np.arange(0, 101, 10))
binid = np.clip(np.digitize(np.abs(rb), bins[1:-1]), 0, 9)
wz = np.array([w_orc[binid == b].mean() for b in range(10)])
print("feasible oracle by |rB|:", round(r2(ya + wz[binid] * rb, yte), 4), " bins:", np.round(wz, 2))
