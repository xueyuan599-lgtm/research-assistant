#!/usr/bin/env python3
"""
Model Search & Optuna Hyperparameter Tuning — Titanic V4 Features
=================================================================
Pipeline:
  1. Load V4 features
  2. Screen 40+ models (9 families) via 5-fold CV x 3 seeds
  3. Top 6 → Optuna 50-trial Bayesian optimisation
  4. Final ranking with RepeatedStratifiedKFold + Brier/LogLoss/AUC
  5. Learning curves for Top 3
  6. Best single-model submission
"""

import os, sys, json, time, warnings, textwrap
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, RepeatedStratifiedKFold,
    learning_curve, cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
    make_scorer
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import optuna

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
X_TRAIN = os.path.join(BASE, "X_train.csv")
X_TEST  = os.path.join(BASE, "X_test.csv")

# Outputs
MODEL_RANKINGS = os.path.join(BASE, "model_rankings.csv")
OPTUNA_JSON    = os.path.join(BASE, "optuna_results.json")
SUBMISSION_CSV = os.path.join(BASE, "submission_best_single.csv")
MODEL_REPORT   = os.path.join(BASE, "model_report.txt")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("  MODEL SEARCH — Titanic V4 Features")
print("=" * 70)

train = pd.read_csv(X_TRAIN)
X = train.drop("Survived", axis=1)
y = train["Survived"].astype(int)
X_test = pd.read_csv(X_TEST)

n_features = X.shape[1]
print(f"\n[LOAD] Train: {X.shape[0]} x {n_features}, Test: {X_test.shape[0]} x {X_test.shape[1]}")
print(f"[LOAD] Class balance — Survived: {y.sum()} ({y.mean():.1%}), Died: {(1-y).sum()} ({(1-y.mean()):.1%})")

# ---------------------------------------------------------------------------
# 2. Define model pool (~40 models)
# ---------------------------------------------------------------------------
def build_model_pool():
    """Return list of (name_short, model, param_desc) tuples."""
    pool = []

    # --- Logistic Regression (5) ---
    for c in [0.01, 0.1, 0.3, 0.5, 1.0]:
        pool.append((
            f"Logistic_C{c}",
            LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED),
            f"C={c}"
        ))

    # --- Random Forest (5) ---
    for d in [4, 5, 6, 7, 8]:
        pool.append((
            f"RF_depth{d}",
            RandomForestClassifier(
                n_estimators=200, max_depth=d,
                min_samples_leaf=3, min_samples_split=5,
                random_state=SEED, n_jobs=-1
            ),
            f"n=200, depth={d}, leaf=3, split=5"
        ))

    # --- ExtraTrees (3) ---
    for d in [5, 7, None]:
        label = f"None" if d is None else str(d)
        pool.append((
            f"ExtraTrees_depth{label}",
            ExtraTreesClassifier(
                n_estimators=200, max_depth=d,
                random_state=SEED, n_jobs=-1
            ),
            f"n=200, depth={label}"
        ))

    # --- GradientBoosting (3) ---
    for combo in [(3, 0.05), (4, 0.1), (5, 0.1)]:
        d, lr = combo
        pool.append((
            f"GBDT_d{d}_lr{lr}",
            GradientBoostingClassifier(
                n_estimators=100, max_depth=d, learning_rate=lr,
                random_state=SEED
            ),
            f"n=100, depth={d}, lr={lr}"
        ))

    # --- XGBoost (5) ---
    xgb_configs = [
        (100, 3, 0.1, 0, 1, 0.8),
        (100, 4, 0.05, 0.5, 1, 0.8),
        (200, 3, 0.03, 0, 3, 0.7),
        (200, 5, 0.05, 1, 1, 0.7),
        (200, 4, 0.1, 0, 1, 0.8),
    ]
    for n, d, lr, alpha, lambd, ss in xgb_configs:
        pool.append((
            f"XGB_n{n}_d{d}_lr{lr}",
            XGBClassifier(
                n_estimators=n, max_depth=d, learning_rate=lr,
                reg_alpha=alpha, reg_lambda=lambd, subsample=ss,
                random_state=SEED, eval_metric="logloss", verbosity=0
            ),
            f"n={n}, d={d}, lr={lr}, α={alpha}, λ={lambd}, ss={ss}"
        ))

    # --- CatBoost (5) ---
    cb_configs = [
        (200, 3, 0.1, 1), (200, 4, 0.05, 3),
        (300, 5, 0.03, 5), (300, 4, 0.05, 3), (200, 6, 0.03, 7),
    ]
    for it, d, lr, l2 in cb_configs:
        pool.append((
            f"CatBoost_it{it}_d{d}_lr{lr}",
            CatBoostClassifier(
                iterations=it, depth=d, learning_rate=lr,
                l2_leaf_reg=l2, random_seed=SEED,
                verbose=0, allow_writing_files=False
            ),
            f"it={it}, d={d}, lr={lr}, l2={l2}"
        ))

    # --- LightGBM (5) ---
    lgb_configs = [
        (100, 3, 0.1, 15, 0, 0.5),
        (100, 4, 0.05, 31, 0, 1),
        (200, 5, 0.03, 7, 0.5, 0.5),
        (200, 3, 0.05, 15, 0, 1),
        (200, 4, 0.1, 31, 0.5, 1),
    ]
    for n, d, lr, leaves, alpha, lambd in lgb_configs:
        pool.append((
            f"LGB_n{n}_d{d}_lr{lr}",
            LGBMClassifier(
                n_estimators=n, max_depth=d, learning_rate=lr,
                num_leaves=leaves, reg_alpha=alpha, reg_lambda=lambd,
                random_state=SEED, verbose=-1, force_col_wise=True
            ),
            f"n={n}, d={d}, lr={lr}, leaves={leaves}"
        ))

    # --- MLP (3) ---
    for hs, a in [((32,), 0.001), ((64, 32), 0.01), ((32, 16, 8), 0.1)]:
        hs_str = "x".join(str(s) for s in hs)
        pool.append((
            f"MLP_h{hs_str}_a{a}",
            MLPClassifier(
                hidden_layer_sizes=hs, alpha=a,
                max_iter=2000, random_state=SEED, early_stopping=True
            ),
            f"hidden={hs}, α={a}"
        ))

    # --- KNN (3) ---
    for n, w in [(5, "uniform"), (10, "distance"), (20, "uniform")]:
        pool.append((
            f"KNN_k{n}_{w}",
            KNeighborsClassifier(n_neighbors=n, weights=w, n_jobs=-1),
            f"k={n}, weights={w}"
        ))

    return pool


# ---------------------------------------------------------------------------
# 3. Initial screening: 5-fold CV x 3 seeds
# ---------------------------------------------------------------------------
def screen_models(pool, X, y, seeds=[42, 123, 456]):
    """Evaluate all models with StratifiedKFold 5-fold x multiple seeds."""
    results = []
    total = len(pool)
    print(f"\n[SCREEN] Evaluating {total} models (5-fold CV x {len(seeds)} seeds)...")

    for idx, (name, model, desc) in enumerate(pool, 1):
        t0 = time.time()
        accs = []
        for s in seeds:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            accs.extend(scores)
        elapsed = time.time() - t0
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results.append({
            "name": name, "model": model, "desc": desc,
            "mean_acc": mean_acc, "std_acc": std_acc,
            "time": elapsed
        })
        tag = "NEW BEST" if idx == 1 or mean_acc >= max(r["mean_acc"] for r in results) else ""
        print(f"  [{idx:2d}/{total}] {name:<30s} acc={mean_acc:.4f}±{std_acc:.4f}  ({elapsed:.1f}s) {tag}")

    results.sort(key=lambda r: r["mean_acc"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 4. Optuna tuning for Top 6 models
# ---------------------------------------------------------------------------
def optuna_xgboost(trial, X, y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": SEED,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return np.mean(scores)


def optuna_catboost(trial, X, y):
    params = {
        "depth": trial.suggest_int("depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": trial.suggest_int("iterations", 100, 500),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0, 5),
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
    }
    model = CatBoostClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return np.mean(scores)


def optuna_lightgbm(trial, X, y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": SEED,
        "verbose": -1,
        "force_col_wise": True,
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return np.mean(scores)


def optuna_rf(trial, X, y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": SEED,
    }
    model = RandomForestClassifier(**params, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return np.mean(scores)


def optuna_gbdt(trial, X, y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": SEED,
    }
    model = GradientBoostingClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return np.mean(scores)


def optuna_et(trial, X, y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": SEED,
    }
    model = ExtraTreesClassifier(**params, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return np.mean(scores)


OPTUNA_FUNCS = {
    "XGB": optuna_xgboost,
    "CatBoost": optuna_catboost,
    "LGB": optuna_lightgbm,
    "RF": optuna_rf,
    "GBDT": optuna_gbdt,
    "ET": optuna_et,
}


def run_optuna_for_top6(screening_results, X, y, n_trials=50):
    """Run Optuna for the Top 6 model families."""
    print(f"\n{'='*70}")
    print(f"  OPTUNA TUNING — Top 6 models, {n_trials} trials each")
    print(f"{'='*70}")

    # Identify Top 6 unique families from screening
    family_order = []
    for r in screening_results:
        family = r["name"].split("_")[0].replace("ExtraTrees", "ET").replace("CatBoost", "CatBoost")
        # Normalize family names to match OPTUNA_FUNCS keys
        for key in ["XGB", "CatBoost", "LGB", "RF", "GBDT", "ET"]:
            if key.lower() in r["name"].lower() and key not in family_order:
                family_order.append(key)
                break
        if len(family_order) >= 6:
            break

    # Ensure we have exactly 6 that match OPTUNA_FUNCS
    family_order = [f for f in family_order if f in OPTUNA_FUNCS]
    # Backfill missing families
    for f in ["XGB", "CatBoost", "LGB", "RF", "GBDT", "ET"]:
        if f not in family_order:
            family_order.append(f)
    family_order = family_order[:6]

    print(f"  Targeting families: {family_order}")

    optuna_results = {}
    for family in family_order:
        print(f"\n  --- {family} Optuna ({n_trials} trials) ---")
        t0 = time.time()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        objective = lambda trial: OPTUNA_FUNCS[family](trial, X, y)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        elapsed = time.time() - t0

        best_params = study.best_params
        best_score = study.best_value

        # Rebuild best model
        best_model = None
        if family == "XGB":
            best_model = XGBClassifier(
                **{k: v for k, v in best_params.items() if k not in ("random_seed",)},
                random_state=SEED, eval_metric="logloss", verbosity=0
            )
        elif family == "CatBoost":
            best_model = CatBoostClassifier(
                **best_params, random_seed=SEED, verbose=0, allow_writing_files=False
            )
        elif family == "LGB":
            best_model = LGBMClassifier(
                **{k: v for k, v in best_params.items()},
                random_state=SEED, verbose=-1, force_col_wise=True
            )
        elif family == "RF":
            best_model = RandomForestClassifier(**best_params, random_state=SEED, n_jobs=-1)
        elif family == "GBDT":
            best_model = GradientBoostingClassifier(**best_params, random_state=SEED)
        elif family == "ET":
            best_model = ExtraTreesClassifier(**best_params, random_state=SEED, n_jobs=-1)

        optuna_results[family] = {
            "best_params": best_params,
            "best_cv_score": float(best_score),
            "best_model": best_model,
            "n_trials": n_trials,
            "time": elapsed,
        }
        print(f"  {family} Best CV acc = {best_score:.5f} (params: {best_params})")
        print(f"  {family} Elapsed: {elapsed:.1f}s")

    return optuna_results


# ---------------------------------------------------------------------------
# 5. Final rigorous evaluation
# ---------------------------------------------------------------------------
def make_scorers():
    return {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "neg_brier": make_scorer(
            lambda y_true, y_prob: -brier_score_loss(y_true, y_prob[:, 1]),
            needs_proba=True, response_method="predict_proba"
        ),
        "neg_log_loss": make_scorer(
            lambda y_true, y_prob: -log_loss(y_true, y_prob[:, 1]),
            needs_proba=True, response_method="predict_proba"
        ),
    }


def final_evaluate(models_ranked, X, y):
    """Rigorous evaluation with RepeatedStratifiedKFold (5-fold x 3 repeats x 3 seeds)."""
    print(f"\n{'='*70}")
    print("  FINAL EVALUATION — RepeatedStratifiedKFold (5-fold x 3 repeats x 3 seeds)")
    print(f"{'='*70}")

    scorers = make_scorers()
    seeds = [42, 123, 456]
    final_results = []

    for idx, entry in enumerate(models_ranked):
        name = entry["name"]
        model = entry["model"]
        t0 = time.time()

        all_scores = {"accuracy": [], "roc_auc": [], "neg_brier": [], "neg_log_loss": []}
        for s in seeds:
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=s)
            cv_results = cross_validate(
                model, X, y, cv=cv, scoring=scorers, n_jobs=-1,
                return_train_score=False
            )
            all_scores["accuracy"].extend(cv_results["test_accuracy"])
            all_scores["roc_auc"].extend(cv_results["test_roc_auc"])
            all_scores["neg_brier"].extend(cv_results["test_neg_brier"])
            all_scores["neg_log_loss"].extend(cv_results["test_neg_log_loss"])

        elapsed = time.time() - t0

        # Convert neg scores back to positive
        brier = float(-np.mean(all_scores["neg_brier"]))
        logloss = float(-np.mean(all_scores["neg_log_loss"]))

        final_results.append({
            "name": name,
            "desc": entry.get("desc", ""),
            "acc_mean": float(np.mean(all_scores["accuracy"])),
            "acc_std": float(np.std(all_scores["accuracy"])),
            "brier": brier,
            "logloss": logloss,
            "auc_mean": float(np.mean(all_scores["roc_auc"])),
            "auc_std": float(np.std(all_scores["roc_auc"])),
            "time": elapsed,
        })
        print(f"  [{idx+1:2d}] {name:<35s} acc={final_results[-1]['acc_mean']:.4f}±{final_results[-1]['acc_std']:.4f}  "
              f"Brier={brier:.4f}  LogLoss={logloss:.4f}  AUC={final_results[-1]['auc_mean']:.4f}  ({elapsed:.1f}s)")

    final_results.sort(key=lambda r: r["acc_mean"], reverse=True)
    return final_results


# ---------------------------------------------------------------------------
# 6. Learning curves
# ---------------------------------------------------------------------------
def plot_learning_curves(top3, X, y):
    """Plot learning curves for Top 3 models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    train_sizes = np.linspace(0.1, 1.0, 10)

    for ax, entry in zip(axes, top3):
        model = entry["model"]
        name = entry["name"]

        t0 = time.time()
        N, train_scores, test_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            scoring="accuracy", n_jobs=-1, random_state=SEED
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        gap = train_mean - test_mean

        ax.plot(train_sizes * 100, train_mean, "o-", color="#2c7bb6", label="Train Accuracy", markersize=4)
        ax.plot(train_sizes * 100, test_mean, "s-", color="#fdae61", label="CV Accuracy", markersize=4)
        ax.fill_between(train_sizes * 100,
                        test_mean - np.std(test_scores, axis=1),
                        test_mean + np.std(test_scores, axis=1),
                        alpha=0.15, color="#fdae61")

        ax.set_title(f"{name}\nGap={np.mean(gap):.3f}", fontsize=10)
        ax.set_xlabel("Training Size (%)")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)
        ax.set_ylim(0.70, 0.95)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Learning Curves — Top 3 Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    lc_path = os.path.join(BASE, "learning_curves.png")
    fig.savefig(lc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[LC] Learning curves saved → {lc_path}")
    return lc_path


# ---------------------------------------------------------------------------
# 7. Generate submission
# ---------------------------------------------------------------------------
def generate_submission(best_model, X, y, X_test, submission_path):
    """Fit best model on full training data and predict on test."""
    print(f"\n[SUBMIT] Fitting best model on full training data...")
    best_model.fit(X, y)

    # Check for predict_proba
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.predict(X_test).astype(float)

    y_pred = (y_prob >= 0.5).astype(int)

    # Load original test PassengerId
    test_orig_path = os.path.join(BASE, "..", "test.csv")
    if os.path.exists(test_orig_path):
        test_orig = pd.read_csv(test_orig_path)
        passenger_ids = test_orig["PassengerId"].values
    else:
        passenger_ids = np.arange(892, 892 + len(y_pred))

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": y_pred
    })
    submission.to_csv(submission_path, index=False)
    print(f"[SUBMIT] Saved → {submission_path}")
    print(f"[SUBMIT] Predicted survival rate: {y_pred.mean():.3f} (n={len(y_pred)})")
    return submission


# ---------------------------------------------------------------------------
# 8. Report
# ---------------------------------------------------------------------------
def write_report(final_results, optuna_results, top3_names):
    """Write model_report.txt."""
    lines = []
    lines.append("=" * 70)
    lines.append("  MODEL SEARCH REPORT — Titanic V4 Features")
    lines.append("=" * 70)
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  FINAL LEADERBOARD (Top 15)")
    lines.append("-" * 70)
    header = f"{'Rank':<5} {'Model':<35s} {'Accuracy':>16s} {'Brier':>8s} {'LogLoss':>8s} {'AUC':>8s}  {'Time':>7s}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, r in enumerate(final_results[:15], 1):
        lines.append(
            f"{i:<5} {r['name']:<35s} {r['acc_mean']:.4f} ± {r['acc_std']:.4f}  "
            f"{r['brier']:.4f}  {r['logloss']:.4f}  {r['auc_mean']:.4f}  {r['time']:.1f}s"
        )

    lines.append("")
    lines.append("-" * 70)
    lines.append("  OPTUNA TUNING RESULTS — Top 6 Families")
    lines.append("-" * 70)
    for family, info in optuna_results.items():
        lines.append(f"\n  {family}:")
        lines.append(f"    Best CV Accuracy: {info['best_cv_score']:.5f}")
        lines.append(f"    Best Params: {json.dumps(info['best_params'], indent=6)}")
        lines.append(f"    Trials: {info['n_trials']} | Time: {info['time']:.1f}s")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  LEARNING CURVE ANALYSIS — Top 3 Models")
    lines.append("-" * 70)
    for name in top3_names:
        lines.append(f"    {name}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  OVERFITTING DIAGNOSIS")
    lines.append("-" * 70)
    lines.append("  Check learning_curves.png:")
    lines.append("  - If Train >> CV consistently → overfitting (reduce model complexity)")
    lines.append("  - If CV plateaus early → insufficient signal (add features)")
    lines.append("  - If CV rises with more data → collect more data would help")
    lines.append("  - Converging gap at high training fraction → model is well-regularised")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  RECOMMENDATIONS")
    lines.append("-" * 70)
    best = final_results[0]
    lines.append(f"  Best single model: {best['name']} (Accuracy={best['acc_mean']:.4f})")
    lines.append(f"  Use for Kaggle submission: v4/submission_best_single.csv")
    lines.append(f"  Next step: Stack top 3-5 models via LogisticRegression meta-learner")

    report_text = "\n".join(lines)
    with open(MODEL_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[REPORT] Saved → {MODEL_REPORT}")
    return report_text


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    overall_t0 = time.time()

    # --- 1. Build pool ---
    pool = build_model_pool()
    print(f"\n[POOL] {len(pool)} models across 9 families")

    # --- 2. Initial screening ---
    screening = screen_models(pool, X, y)
    print(f"\n[SCREEN] Top 10 after screening:")
    for i, r in enumerate(screening[:10], 1):
        print(f"  {i:2d}. {r['name']:<35s} {r['mean_acc']:.4f} ± {r['std_acc']:.4f}")

    # --- 3. Optuna for Top 6 ---
    try:
        optuna_results = run_optuna_for_top6(screening, X, y, n_trials=50)
    except Exception as e:
        print(f"\n[WARN] Optuna failed: {e}. Skipping Optuna, using grid search best.")
        optuna_results = {}

    # --- 4. Build final model list (screening top models + Optuna best models) ---
    # Take Top 10 from screening and add Optuna-tuned versions
    final_models = []
    seen_names = set()

    # Add Optuna models first (they are likely best)
    for family, info in optuna_results.items():
        name = f"{family}_Optuna_Best"
        final_models.append({
            "name": name,
            "model": info["best_model"],
            "desc": f"Optuna tuned ({info['n_trials']} trials)",
        })
        seen_names.add(name)

    # Add top screening models (skip duplicates)
    for r in screening:
        if r["name"] not in seen_names:
            final_models.append({
                "name": r["name"],
                "model": r["model"],
                "desc": r["desc"],
            })
            seen_names.add(r["name"])
        if len(final_models) >= 20:
            break

    # --- 5. Final rigorous evaluation ---
    final_ranking = final_evaluate(final_models, X, y)

    # --- 6. Save rankings CSV ---
    df_rankings = pd.DataFrame(final_ranking)
    df_rankings.to_csv(MODEL_RANKINGS, index=False)
    print(f"\n[RANKINGS] Saved → {MODEL_RANKINGS}")

    # --- 7. Save Optuna results JSON ---
    optuna_serializable = {}
    for fam, info in optuna_results.items():
        optuna_serializable[fam] = {
            "best_cv_score": info["best_cv_score"],
            "best_params": {k: (float(v) if isinstance(v, (np.floating,)) else v)
                           for k, v in info["best_params"].items()},
            "n_trials": info["n_trials"],
            "time": info["time"],
        }
    with open(OPTUNA_JSON, "w") as f:
        json.dump(optuna_serializable, f, indent=2)
    print(f"[OPTUNA_JSON] Saved → {OPTUNA_JSON}")

    # --- 8. Learning curves ---
    top3 = final_ranking[:3]
    top3_names = [r["name"] for r in top3]
    # Get the actual model objects
    top3_models = []
    for r in top3:
        for m in final_models:
            if m["name"] == r["name"]:
                top3_models.append({"name": m["name"], "model": m["model"]})
                break
    try:
        plot_learning_curves(top3_models, X, y)
    except Exception as e:
        print(f"[WARN] Learning curve plotting failed: {e}")

    # --- 9. Submission ---
    best_entry = final_ranking[0]
    best_model_obj = None
    for m in final_models:
        if m["name"] == best_entry["name"]:
            best_model_obj = m["model"]
            break
    if best_model_obj:
        generate_submission(best_model_obj, X, y, X_test, SUBMISSION_CSV)

    # --- 10. Report ---
    write_report(final_ranking, optuna_results, top3_names)

    # --- Summary ---
    total_elapsed = time.time() - overall_t0
    print(f"\n{'='*70}")
    print(f"  DONE — Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Best model: {final_ranking[0]['name']} (acc={final_ranking[0]['acc_mean']:.4f})")
    print(f"  Outputs:")
    print(f"    {MODEL_RANKINGS}")
    print(f"    {OPTUNA_JSON}")
    print(f"    {SUBMISSION_CSV}")
    print(f"    {MODEL_REPORT}")
    print(f"{'='*70}")
