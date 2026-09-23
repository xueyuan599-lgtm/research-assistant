import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from scipy.stats import spearmanr, pearsonr
from data_loader import generate_s5
from ahfe import AHFE
from sklearn.model_selection import train_test_split

X, y, info = generate_s5(n=300, nonlinear=True, seed=0)
phi, dt = info["phi"], info["dt"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

ahfe = AHFE(A=4, gbdt="xgb", random_state=0).fit(Xtr, ytr)
ya, rb, w = ahfe.predict_with_components(Xte)
q_te = (Xte @ phi) * dt
NL = np.abs(info["lambda_nl"] * (q_te ** 2 - np.mean(q_te ** 2)))

# gate feature correlations with NL
gfe = ahfe.gfe_
z = gfe.transform(Xte, ya, rb, None)
names = [f"fpca{i}" for i in range(z.shape[1]-3)] + ["|rB|", "Qres", "lev"]
print("corr(z_col, NL):")
for i, nm in enumerate(names):
    print(f"  {nm:6s} pearson={pearsonr(z[:,i], NL).statistic:+.3f} spearman={spearmanr(z[:,i], NL).statistic:+.3f}")

print("corr(w, NL) spearman=", round(spearmanr(w, NL).statistic,3), " p=", round(spearmanr(w,NL).pvalue,3))
print("corr(|rB|, NL) spearman=", round(spearmanr(rb, NL).statistic,3))
print("corr(w, |rB|) spearman=", round(spearmanr(w, rb).statistic,3))
print("gate theta (unstd) =", np.round(ahfe.gate_.theta_,3))
print("w mean", round(w.mean(),3), "w std", round(w.std(),3), "frac>0.5", round((w>0.5).mean(),3))
# does NL predict |rB| well? |rB| is the corrector's output
print("NL vs |rB|: NL high quantile |rB| mean vs low:",
      round(rb[NL>np.quantile(NL,0.75)].mean(),3), round(rb[NL<np.quantile(NL,0.25)].mean(),3))
