"""Confirmatory benchmark runner for the AHFE study (plan.md §3, §3.1, §5).

Runs the core scenarios (S1 Tecator-fat, S5 nonlinear, S5b linear, S6a/S6b
contamination) across the method set (M1 FLM, M2 FunFeat+GBDT, M3 SIMPLS
ensemble, AHFE-fixed, AHFE-adaptive) on FRESH confirmatory seeds (P5). Reports:

  - RMSE / R2 per (scenario, method, seed)   -> tables
  - DeltaRMSE vs M3 (+95% CI, paired bootstrap over OOF losses)
  - beta recovery (S5: ||beta_hat - beta||_L2)
  - gate mechanism (S5): Spearman(w, NL), NL-quantile -> mean-w
  - robustness (S6): DeltaRMSE (contaminated - clean), w_outlier diagnostic
  - SHAP importance of the GBDT corrector (S5)

Usage:
  python -m src.benchmark --quick     # reduced (2 seeds x 3 fold)
  python -m src.benchmark             # full (5 seeds x 5 fold)
"""

from __future__ import annotations

import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data_loader import generate_s5, load_tecator
from methods import (M1_FLM, M2_FunFeatGBDT, M3_SIMPLSEmsemble,
                     contaminate_Y, contaminate_X_level_shift)
from ahfe import AHFE
from functional_features import FunctionalFeatureExtractor
from shap_analysis import shap_values, explain_features


def _rmse(pred, yt):
    return float(np.sqrt(np.mean((pred - yt) ** 2)))


def _r2(pred, yt):
    return float(1 - np.sum((pred - yt) ** 2) / np.sum((yt - yt.mean()) ** 2))


def _boot_ci(l_a, l_b, n_boot=2000, seed=0):
    """95% CI for DeltaRMSE = sqrt(mean(l_a)) - sqrt(mean(l_b)) via paired
    bootstrap of per-sample losses (l_a, l_b). Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(l_a)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d.append(np.sqrt(l_a[idx].mean()) - np.sqrt(l_b[idx].mean()))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def make_methods(seed: int, gbdt: str = "xgb"):
    return {
        "M1_FLM": M1_FLM(alpha=1.0),
        "M2_FunFeatGBDT": M2_FunFeatGBDT(gbdt=gbdt, random_state=seed),
        "M3_SIMPLSEmsemble": M3_SIMPLSEmsemble(A=4, n_boot=25, random_state=seed),
        "AHFE-fixed": AHFE(A=4, gbdt=gbdt, random_state=seed, gate_mode="fixed"),
        "AHFE-adaptive": AHFE(A=4, gbdt=gbdt, random_state=seed, gate_mode="adaptive"),
    }


def _fold_split(n, seed, n_folds):
    """Repeated-KFold index generator returning (train, test) folds."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


def run_tecator(seed, n_folds, gbdt="xgb", target="fat", contam=None):
    X, _, grid, y_full = load_tecator()
    col = {"fat": 0, "moisture": 1, "protein": 2}[target]
    y = y_full[:, col].ravel()
    if contam == "y_outlier":
        y = contaminate_Y(y, frac=0.10, sigma=5.0, seed=seed)
    elif contam == "x_shift":
        X = contaminate_X_level_shift(X, frac=0.10, shift=3.0, seed=seed)
    res = {m: {"rmse": [], "r2": []} for m in make_methods(seed)}
    for tr, te in _fold_split(len(X), seed, n_folds):
        methods = make_methods(seed)
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        for name, m in methods.items():
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
            res[name]["rmse"].append(_rmse(p, yte))
            res[name]["r2"].append(_r2(p, yte))
    return res


def run_synth(seed, n_folds, nonlinear, gbdt="xgb"):
    X, y, info = generate_s5(n=400, nonlinear=nonlinear, seed=seed)
    res = {m: {"rmse": [], "r2": []} for m in make_methods(seed)}
    gate = {"adaptive": [], "fixed": []}
    for tr, te in _fold_split(len(X), seed, n_folds):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        for name, m in make_methods(seed).items():
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
            res[name]["rmse"].append(_rmse(p, yte))
            res[name]["r2"].append(_r2(p, yte))
        if nonlinear:
            ada = AHFE(A=4, gbdt=gbdt, random_state=seed).fit(Xtr, ytr)
            _, _, w = ada.predict_with_components(Xte)
            gate["adaptive"].append(np.abs(w).ravel())
    return res, info, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gbdt", default="xgb")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    seeds = [100, 101, 102, 103, 104]
    n_folds = 3 if args.quick else 5
    if args.quick:
        seeds = [100, 101]

    out_dir = os.path.join(args.out)
    tab_dir = os.path.join(out_dir, "tables")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    summary = {"scenarios": {}}

    # --- S1 Tecator (fat) ---
    for contam in [None, "y_outlier", "x_shift"]:
        tag = "S1_clean" if contam is None else ("S6a" if contam == "y_outlier" else "S6b")
        agg = {m: {"rmse": [], "r2": []} for m in make_methods(0)}
        for s in seeds:
            r = run_tecator(s, n_folds, args.gbdt, target="fat", contam=contam)
            for m in r:
                agg[m]["rmse"].extend(r[m]["rmse"])
                agg[m]["r2"].extend(r[m]["r2"])
        summary["scenarios"][tag] = {m: {
            "rmse_mean": float(np.mean(agg[m]["rmse"])),
            "rmse_std": float(np.std(agg[m]["rmse"])),
            "r2_mean": float(np.mean(agg[m]["r2"])),
        } for m in agg}
        print(f"[{tag}] " + "  ".join(
            f"{m}: R2={summary['scenarios'][tag][m]['r2_mean']:.3f}"
            for m in agg))

    # --- S5 nonlinear / S5b linear ---
    for nonlinear, tag in [(True, "S5_nonlinear"), (False, "S5b_linear")]:
        agg = {m: {"rmse": [], "r2": []} for m in make_methods(0)}
        beta_err = []
        all_w = []
        for s in seeds:
            r, info, gate = run_synth(s, n_folds, nonlinear, args.gbdt)
            for m in r:
                agg[m]["rmse"].extend(r[m]["rmse"])
                agg[m]["r2"].extend(r[m]["r2"])
            if nonlinear:
                all_w.extend([ww for g in gate["adaptive"] for ww in g])
        summary["scenarios"][tag] = {m: {
            "rmse_mean": float(np.mean(agg[m]["rmse"])),
            "rmse_std": float(np.std(agg[m]["rmse"])),
            "r2_mean": float(np.mean(agg[m]["r2"])),
        } for m in agg}
        if nonlinear and all_w:
            summary["scenarios"][tag]["gate_w"] = {
                "mean": float(np.mean(all_w)), "std": float(np.std(all_w))}
        print(f"[{tag}] " + "  ".join(
            f"{m}: R2={summary['scenarios'][tag][m]['r2_mean']:.3f}"
            for m in agg))

    # --- S5 gate mechanism + beta recovery (confirmatory, honest) ---
    if True:
        from scipy.stats import spearmanr
        X, y, info = generate_s5(n=400, nonlinear=True, seed=100)
        tr, te = _fold_split(len(X), 100, n_folds)[0]
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        ada = AHFE(A=4, gbdt=args.gbdt, random_state=100).fit(Xtr, ytr)
        _, _, w = ada.predict_with_components(Xte)
        NL = info["nl"][te]
        sp = spearmanr(np.abs(w).ravel(), NL).statistic
        qs = np.quantile(NL, [0.2, 0.4, 0.6, 0.8])
        binid = np.clip(np.digitize(NL, qs), 0, 4)
        w_bin = [float(np.mean(np.abs(w)[binid == b])) for b in range(5)]
        # beta recovery of the linear component
        beta_hat = ada.a_model_.beta_
        beta_true = info["beta"]
        grid = info["grid"]
        dt = grid[1] - grid[0]
        beta_err = float(np.sqrt(np.sum((beta_hat - beta_true) ** 2 * dt)))
        summary["S5_gate_mechanism"] = {
            "spearman_w_NL": float(sp),
            "nl_quantile_mean_w": w_bin,
            "mean_w": float(np.mean(np.abs(w))),
            "std_w": float(np.std(np.abs(w))),
        }
        summary["S5_beta_recovery"] = {
            "L2_error_int": round(beta_err, 5),
            "corr_beta": float(np.corrcoef(beta_hat, beta_true)[0, 1]),
        }
        print(f"[S5 gate] Spearman(w,NL)={sp:+.3f}  mean_w={np.mean(np.abs(w)):.3f} "
              f"std_w={np.std(np.abs(w)):.3f}  w_bins={np.round(w_bin,3)}")
        print(f"[S5 beta] L2_err={beta_err:.4f}  corr(beta_hat,beta)="
              f"{np.corrcoef(beta_hat, beta_true)[0, 1]:.3f}")

    # --- paired bootstrap DeltaRMSE vs M3 on S5 nonlinear (pooled OOF losses) ---
    X, y, _ = generate_s5(n=400, nonlinear=True, seed=100)
    l3_all, la_all = [], []
    rmse3, rmsea = [], []
    for tr, te in _fold_split(len(X), 100, n_folds):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        m3 = M3_SIMPLSEmsemble(A=4, n_boot=25, random_state=100).fit(Xtr, ytr)
        ada = AHFE(A=4, gbdt=args.gbdt, random_state=100).fit(Xtr, ytr)
        p3 = m3.predict(Xte); pa = ada.predict(Xte)
        l3_all.append((yte - p3) ** 2); la_all.append((yte - pa) ** 2)
        rmse3.append(_rmse(p3, yte)); rmsea.append(_rmse(pa, yte))
    l3 = np.concatenate(l3_all); la = np.concatenate(la_all)
    mean, lo, hi = _boot_ci(l3, la, seed=100)
    d_rmse = float(np.mean(rmse3) - np.mean(rmsea))
    summary["S5_deltaRMSE_vs_M3"] = {
        "M3_RMSE_mean": float(np.mean(rmse3)), "AHFE_RMSE_mean": float(np.mean(rmsea)),
        "delta_RMSE": round(d_rmse, 4),
        "boot_CI_95": [round(lo, 4), round(hi, 4)],
        "boot_mean_deltaRMSE": round(mean, 5),
    }
    print(f"[S5] M3_RMSE={summary['S5_deltaRMSE_vs_M3']['M3_RMSE_mean']:.4f} "
          f"AHFE_RMSE={summary['S5_deltaRMSE_vs_M3']['AHFE_RMSE_mean']:.4f} "
          f"delta={summary['S5_deltaRMSE_vs_M3']['delta_RMSE']:.4f} "
          f"95%CI={summary['S5_deltaRMSE_vs_M3']['boot_CI_95']}")

    # --- SHAP of corrector on S5 ---
    X, y, _ = generate_s5(n=400, nonlinear=True, seed=100)
    Xtr, ytr = X[:300], y[:300]
    fe = FunctionalFeatureExtractor().fit(Xtr)
    F = fe.transform(Xtr)
    ada = AHFE(A=4, gbdt=args.gbdt, random_state=100).fit(Xtr, ytr)
    imp = shap_values(ada.b_model_, F)
    names = explain_features(F)
    summary["S5_shap"] = dict(zip(names, np.round(imp, 5)))
    print("[S5 SHAP top] " + ", ".join(
        f"{names[i]}={imp[i]:.4f}" for i in np.argsort(imp)[::-1][:5]))

    out = os.path.join(out_dir, "benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
