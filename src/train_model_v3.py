"""
Employee Attrition - v3 pipeline: feature engineering, expanded model
comparison, imbalance-method comparison, calibration check, feature
selection ablation, stability (CV mean +/- std), and a leak-free
head-to-head against the existing attrition_model_v2.pkl on the SAME
test split it was evaluated on.

Never touches attrition_model.pkl or attrition_model_v2.pkl. Only writes
attrition_model_v3.pkl if the evidence gathered here actually supports it
being better than v2 - otherwise v2 remains the recommended candidate and
this script says so explicitly.

Run with: python -m src.train_model_v3
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict,
    cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.preprocessing import MODEL_FEATURES, encode_raw_dataframe
from src.feature_engineering import ENGINEERED_FEATURES, add_engineered_features

RANDOM_STATE = 42
DATA_PATH = Path("IBM Dataset.csv")
MODELS_DIR = Path("models")
V2_MODEL_PATH = Path("attrition_model_v2.pkl")
V3_MODEL_PATH = Path("attrition_model_v3.pkl")
OLD_MODEL_PATH = Path("attrition_model.pkl")
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
THRESHOLDS = [round(t, 2) for t in np.arange(0.20, 0.71, 0.05)]
SCALE_POS_WEIGHT = 1.0  # set in main() from y_train; used for XGBoost's class_weight equivalent


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------
def base_estimator(name, balancing):
    """balancing in {'none','class_weight','smote'}. Returns an unfitted
    estimator/pipeline (SMOTE step included when balancing=='smote')."""
    class_weight = "balanced" if balancing == "class_weight" else None

    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight=class_weight)
        steps = [("scale", StandardScaler()), ("clf", clf)]
    elif name == "Random Forest":
        clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight=class_weight)
        steps = [("clf", clf)]
    elif name == "XGBoost":
        # XGBoost has no class_weight param - scale_pos_weight is its equivalent.
        spw = SCALE_POS_WEIGHT if balancing == "class_weight" else 1.0
        clf = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
                             scale_pos_weight=spw)
        steps = [("clf", clf)]
    elif name == "CatBoost":
        auto_cw = "Balanced" if balancing == "class_weight" else None
        # allow_writing_files=False: CatBoost otherwise writes training logs to a
        # relative ./catboost_info dir, which races and fails under parallel
        # (n_jobs=-1) cross-validation on Windows.
        clf = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, allow_writing_files=False,
                                  auto_class_weights=auto_cw, iterations=300)
        steps = [("clf", clf)]
    else:
        raise ValueError(name)

    if balancing == "smote":
        return ImbPipeline([("smote", SMOTE(random_state=RANDOM_STATE))] + steps)
    return ImbPipeline(steps)


def cv_scores(pipeline, X, y) -> dict:
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "roc_auc": "roc_auc"}
    result = cross_validate(pipeline, X, y, cv=CV, scoring=scoring, n_jobs=-1)
    return {k: (float(np.mean(result[f"test_{k}"])), float(np.std(result[f"test_{k}"]))) for k in scoring}


def flat(scores: dict, suffix="") -> dict:
    out = {}
    for k, (m, s) in scores.items():
        out[f"{k}{suffix}_mean"] = m
        out[f"{k}{suffix}_std"] = s
    return out


def threshold_sweep(y_true, proba) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        })
    return rows


def pick_operating_points(sweep: list[dict]) -> dict:
    balanced = max(sweep, key=lambda r: r["f1"])
    high_recall_candidates = [r for r in sweep if r["precision"] >= 0.35] or sweep
    high_recall = max(high_recall_candidates, key=lambda r: r["recall"])
    high_precision_candidates = [r for r in sweep if r["recall"] >= 0.25] or sweep
    high_precision = max(high_precision_candidates, key=lambda r: r["precision"])
    return {"high_recall": high_recall, "balanced": balanced, "high_precision": high_precision}


def evaluate_at_threshold(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


PARAM_DISTRIBUTIONS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "liblinear"],
    },
    "Random Forest": {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth": [6, 8, 10, 12, None],
        "clf__min_samples_split": [2, 4, 6],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
    "CatBoost": {
        "clf__depth": [4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__l2_leaf_reg": [1, 3, 5, 9],
        "clf__iterations": [200, 300, 400],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
    },
}


def main():
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    # ============================================================ PHASE 1 ==
    section("PHASE 1 - Baseline data checks")
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] == 1470, f"expected 1470 rows, got {df.shape[0]}"
    assert "Attrition" in df.columns
    print(f"shape={df.shape}, missing={int(df.isnull().sum().sum())}, "
          f"duplicates={int(df.duplicated().sum())}")
    print(df["Attrition"].value_counts())
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    print(f"{len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns")

    X_base, y = encode_raw_dataframe(df)
    assert list(X_base.columns) == MODEL_FEATURES
    # SAME split as train_model.py (same seed, same test_size) so v2 and v3
    # are compared on the identical, untouched held-out test set.
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_base, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train_base.shape}, Test: {X_test_base.shape} (untouched until Phase 13)")

    global SCALE_POS_WEIGHT
    SCALE_POS_WEIGHT = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"scale_pos_weight for XGBoost class-weighting = {SCALE_POS_WEIGHT:.3f}")

    # ============================================================ PHASE 2 ==
    section("PHASE 2 - Feature engineering ablation (CV on training data only)")
    X_train_eng = add_engineered_features(X_train_base)
    X_test_eng = add_engineered_features(X_test_base)  # not evaluated until Phase 13
    eng_feature_list = MODEL_FEATURES + ENGINEERED_FEATURES
    for feat, desc in __import__("src.feature_engineering", fromlist=["FEATURE_DESCRIPTIONS"]).FEATURE_DESCRIPTIONS.items():
        print(f"  {feat}: {desc}")

    fe_rows = []
    for feature_set_name, X_tr in [("base_49", X_train_base), ("engineered", X_train_eng)]:
        for model_name in ["Logistic Regression", "XGBoost"]:
            pipe = base_estimator(model_name, "class_weight" if model_name == "Logistic Regression" else "none")
            scores = cv_scores(pipe, X_tr, y_train)
            fe_rows.append({"feature_set": feature_set_name, "model": model_name, **flat(scores)})
    fe_df = pd.DataFrame(fe_rows)
    print(fe_df[["feature_set", "model", "f1_mean", "f1_std", "recall_mean", "roc_auc_mean"]].to_string(index=False))

    base_f1 = fe_df[fe_df.feature_set == "base_49"]["f1_mean"].mean()
    eng_f1 = fe_df[fe_df.feature_set == "engineered"]["f1_mean"].mean()
    IMPROVEMENT_MARGIN = 0.01
    use_engineered = eng_f1 > base_f1 + IMPROVEMENT_MARGIN
    print(f"\nMean CV F1 across LR+XGBoost: base={base_f1:.4f}, engineered={eng_f1:.4f} "
          f"(margin required: {IMPROVEMENT_MARGIN}) -> use_engineered_features = {use_engineered}")

    if use_engineered:
        X_train, X_test, FEATURE_LIST, feature_version = X_train_eng, X_test_eng, eng_feature_list, "v2_engineered"
    else:
        X_train, X_test, FEATURE_LIST, feature_version = X_train_base, X_test_base, MODEL_FEATURES, "v1_base49"
    print(f"Selected feature version: {feature_version} ({len(FEATURE_LIST)} features)")

    # ============================================================ PHASE 3 ==
    section("PHASE 3 - Imbalance handling comparison (5-fold CV, default hyperparams)")
    imbalance_rows = []
    for model_name in ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]:
        for balancing in ["none", "class_weight", "smote"]:
            pipe = base_estimator(model_name, balancing)
            scores = cv_scores(pipe, X_train, y_train)
            imbalance_rows.append({"model": model_name, "balancing": balancing, **flat(scores)})
    imbalance_df = pd.DataFrame(imbalance_rows)
    print(imbalance_df[["model", "balancing", "f1_mean", "f1_std", "recall_mean", "precision_mean", "roc_auc_mean"]]
          .to_string(index=False))

    best_balancing = {}
    for model_name in ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]:
        sub = imbalance_df[imbalance_df.model == model_name]
        best_row = sub.loc[sub["f1_mean"].idxmax()]
        best_balancing[model_name] = best_row["balancing"]
        print(f"{model_name}: best balancing = {best_row['balancing']} (F1={best_row['f1_mean']:.3f})")

    # ============================================================ PHASE 4 ==
    section("PHASE 4 - Hyperparameter tuning (RandomizedSearchCV, scoring=f1, 5-fold CV)")
    tuned_models, tuning_summary = {}, {}
    for model_name in ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]:
        balancing = best_balancing[model_name]
        pipe = base_estimator(model_name, balancing)
        search = RandomizedSearchCV(
            pipe, PARAM_DISTRIBUTIONS[model_name], n_iter=20, scoring="f1",
            cv=CV, random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        tuned_models[model_name] = search.best_estimator_
        tuning_summary[model_name] = {
            "balancing": balancing,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "best_cv_f1": float(search.best_score_),
        }
        print(f"{model_name} (balancing={balancing}): best CV F1={search.best_score_:.3f}, "
              f"params={tuning_summary[model_name]['best_params']}")

    # ============================================================ PHASE 8 ==
    # (computed here, right after tuning, so Phase 5's model selection can use it)
    section("PHASE 8 - Stability: mean +/- std across folds for each tuned model")
    stability_rows = []
    for model_name, model in tuned_models.items():
        scores = cv_scores(model, X_train, y_train)
        stability_rows.append({"model": model_name, **flat(scores)})
        print(f"{model_name}: F1={scores['f1'][0]:.3f}+/-{scores['f1'][1]:.3f}  "
              f"Recall={scores['recall'][0]:.3f}+/-{scores['recall'][1]:.3f}  "
              f"ROC-AUC={scores['roc_auc'][0]:.3f}+/-{scores['roc_auc'][1]:.3f}")
    stability_df = pd.DataFrame(stability_rows)

    # ============================================================ PHASE 5a ==
    section("PHASE 5a - Model TYPE selection (out-of-fold, full feature set, training data)")
    oof_rows = {}
    selection_rows = []
    for model_name, model in tuned_models.items():
        oof_proba = cross_val_predict(model, X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        oof_rows[model_name] = oof_proba
        sweep_m = threshold_sweep(y_train, oof_proba)
        best = max(sweep_m, key=lambda r: r["f1"])
        oof_roc_auc = float(roc_auc_score(y_train, oof_proba))
        oof_pr_auc = float(average_precision_score(y_train, oof_proba))
        selection_rows.append({
            "model": model_name, "best_threshold": best["threshold"],
            "oof_precision": best["precision"], "oof_recall": best["recall"], "oof_f1": best["f1"],
            "oof_roc_auc": oof_roc_auc, "oof_pr_auc": oof_pr_auc,
            "cv_f1_mean": stability_df.loc[stability_df.model == model_name, "f1_mean"].iloc[0],
            "cv_f1_std": stability_df.loc[stability_df.model == model_name, "f1_std"].iloc[0],
        })
    selection_df = pd.DataFrame(selection_rows)
    selection_df["rank_score"] = selection_df["oof_f1"] + 0.5 * selection_df["oof_recall"] + 0.25 * selection_df["oof_roc_auc"]
    print(selection_df.to_string(index=False))

    best_row = selection_df.sort_values("rank_score", ascending=False).iloc[0]
    best_name = best_row["model"]
    best_model = tuned_models[best_name]
    print(f"\nSelected model type: {best_name} (won by rank_score; CV F1 std across folds was "
          f"{best_row['cv_f1_std']:.3f} - {'consistent across folds' if best_row['cv_f1_std'] < 0.08 else 'notably variable across folds, treat the win with caution'})")

    # ============================================================ PHASE 7 ==
    section("PHASE 7 - Feature selection ablation, for the winning model type (CV on training data only)")
    inner_clf = best_model.named_steps["clf"] if hasattr(best_model, "named_steps") else best_model
    if hasattr(inner_clf, "feature_importances_"):
        importances = pd.Series(inner_clf.feature_importances_, index=FEATURE_LIST).sort_values(ascending=False)
    elif hasattr(inner_clf, "coef_"):
        importances = pd.Series(np.abs(inner_clf.coef_[0]), index=FEATURE_LIST).sort_values(ascending=False)
    else:
        importances = None

    if importances is not None:
        cumulative = importances.cumsum() / importances.sum()
        selected_features = importances[cumulative <= 0.90].index.tolist()
        if len(selected_features) < 5:
            selected_features = importances.head(max(5, len(importances) // 2)).index.tolist()
        print(f"Candidate subset: {len(selected_features)}/{len(FEATURE_LIST)} features covering ~90% of importance")

        balancing = best_balancing[best_name]
        pipe_all = base_estimator(best_name, balancing)
        pipe_sel = base_estimator(best_name, balancing)
        scores_all = cv_scores(pipe_all, X_train, y_train)
        scores_sel = cv_scores(pipe_sel, X_train[selected_features], y_train)
        print(f"All features    ({len(FEATURE_LIST)}): F1={scores_all['f1'][0]:.3f}+/-{scores_all['f1'][1]:.3f}")
        print(f"Selected subset ({len(selected_features)}): F1={scores_sel['f1'][0]:.3f}+/-{scores_sel['f1'][1]:.3f}")

        use_feature_selection = scores_sel["f1"][0] > scores_all["f1"][0] + 0.01
        print(f"-> use_feature_selection = {use_feature_selection} "
              f"(kept all {len(FEATURE_LIST)} features unless the subset was clearly better)")
        if not use_feature_selection:
            selected_features = FEATURE_LIST
    else:
        selected_features = FEATURE_LIST
        use_feature_selection = False

    final_feature_list = selected_features
    if use_feature_selection:
        # Refit the winning model's tuned hyperparameters on the reduced feature
        # set - the model used from here on is genuinely fit on final_feature_list,
        # not silently still using all features.
        from sklearn.base import clone
        best_model_final = clone(best_model)
        best_model_final.fit(X_train[final_feature_list], y_train)
    else:
        best_model_final = best_model  # already fit on FEATURE_LIST by RandomizedSearchCV(refit=True)

    # ============================================================ PHASE 5b ==
    section("PHASE 5b - Threshold optimization + operating points (out-of-fold, final feature set)")
    best_oof_proba = cross_val_predict(
        best_model_final, X_train[final_feature_list], y_train, cv=CV, method="predict_proba", n_jobs=-1
    )[:, 1]
    sweep = threshold_sweep(y_train, best_oof_proba)
    operating_points = pick_operating_points(sweep)
    for label, row in operating_points.items():
        print(f"  {label}: threshold={row['threshold']:.2f} precision={row['precision']:.3f} "
              f"recall={row['recall']:.3f} f1={row['f1']:.3f}")
    chosen_threshold = operating_points["balanced"]["threshold"]

    # ============================================================ PHASE 6 ==
    section("PHASE 6 - Calibration check (nested CV, final feature set, training data only)")
    calibrated = CalibratedClassifierCV(best_model_final, method="sigmoid", cv=3)
    oof_calibrated_proba = cross_val_predict(
        calibrated, X_train[final_feature_list], y_train, cv=CV, method="predict_proba", n_jobs=-1
    )[:, 1]

    brier_raw = float(brier_score_loss(y_train, best_oof_proba))
    brier_calibrated = float(brier_score_loss(y_train, oof_calibrated_proba))
    roc_auc_raw = float(roc_auc_score(y_train, best_oof_proba))
    roc_auc_calibrated = float(roc_auc_score(y_train, oof_calibrated_proba))
    sweep_calibrated = threshold_sweep(y_train, oof_calibrated_proba)
    best_f1_raw = max(sweep, key=lambda r: r["f1"])["f1"]
    best_f1_calibrated = max(sweep_calibrated, key=lambda r: r["f1"])["f1"]

    print(f"Raw:        Brier={brier_raw:.4f}  ROC-AUC={roc_auc_raw:.4f}  best-F1={best_f1_raw:.4f}")
    print(f"Calibrated: Brier={brier_calibrated:.4f}  ROC-AUC={roc_auc_calibrated:.4f}  best-F1={best_f1_calibrated:.4f}")

    BRIER_IMPROVEMENT_MARGIN = 0.003
    use_calibration = (brier_raw - brier_calibrated > BRIER_IMPROVEMENT_MARGIN) and (best_f1_calibrated >= best_f1_raw - 0.01)
    print(f"-> use_calibration = {use_calibration} "
          f"(requires Brier improvement > {BRIER_IMPROVEMENT_MARGIN} without hurting best-F1 by more than 0.01)")

    if use_calibration:
        final_model = CalibratedClassifierCV(best_model_final, method="sigmoid", cv=3)
        final_model.fit(X_train[final_feature_list], y_train)
        sweep = sweep_calibrated
        operating_points = pick_operating_points(sweep)
        chosen_threshold = operating_points["balanced"]["threshold"]
    else:
        final_model = best_model_final

    # =========================================================== PHASE 13 ==
    section("PHASE 13 - Final held-out test evaluation (test set touched for the first time)")
    test_proba = final_model.predict_proba(X_test[final_feature_list])[:, 1]

    final_test_metrics = evaluate_at_threshold(y_test, test_proba, chosen_threshold)
    final_test_metrics_05 = evaluate_at_threshold(y_test, test_proba, 0.5)
    print(f"{best_name} v3 @ threshold={chosen_threshold}: {final_test_metrics}")
    print(f"{best_name} v3 @ threshold=0.5: {final_test_metrics_05}")

    # Bootstrap 95% CI for test F1 (small test set - 294 rows - so point
    # estimates alone can be misleading; this quantifies that uncertainty).
    rng = np.random.RandomState(RANDOM_STATE)
    y_test_arr = y_test.to_numpy()
    boot_f1 = []
    for _ in range(1000):
        idx = rng.randint(0, len(y_test_arr), len(y_test_arr))
        pred_b = (test_proba[idx] >= chosen_threshold).astype(int)
        boot_f1.append(f1_score(y_test_arr[idx], pred_b, zero_division=0))
    ci_low, ci_high = float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5))
    print(f"Bootstrap 95% CI for test F1 (n=1000 resamples): [{ci_low:.3f}, {ci_high:.3f}]")

    roc_curve_test = roc_curve(y_test, test_proba)
    roc_data_v3 = {"fpr": roc_curve_test[0].tolist(), "tpr": roc_curve_test[1].tolist(),
                   "auc": final_test_metrics["roc_auc"]}

    # --- compare vs v2 on the SAME test set (same split, so this is fair) ---
    section("Comparison vs attrition_model_v2.pkl (identical test split)")
    v2_vs_v3 = None
    if V2_MODEL_PATH.exists():
        import joblib
        v2_model = joblib.load(V2_MODEL_PATH)
        v2_proba = v2_model.predict_proba(X_test_base)[:, 1]  # v2 uses base 49 features
        v2_metrics_v2_threshold = evaluate_at_threshold(y_test, v2_proba, 0.35)  # v2's own chosen threshold
        v2_vs_v3 = {
            "attrition_model_v2 (its own threshold=0.35)": v2_metrics_v2_threshold,
            f"{best_name} v3 (threshold={chosen_threshold})": final_test_metrics,
        }
        for label, res in v2_vs_v3.items():
            print(f"{label}: acc={res['accuracy']:.3f} prec={res['precision']:.3f} "
                  f"recall={res['recall']:.3f} f1={res['f1']:.3f} roc_auc={res['roc_auc']:.3f}")

        f1_improvement = final_test_metrics["f1"] - v2_metrics_v2_threshold["f1"]
        recall_improvement = final_test_metrics["recall"] - v2_metrics_v2_threshold["recall"]
        MEANINGFUL_MARGIN = 0.03
        v3_meaningfully_better = (f1_improvement > MEANINGFUL_MARGIN and recall_improvement >= -0.02) or \
                                  (recall_improvement > MEANINGFUL_MARGIN and f1_improvement >= -0.02)
        print(f"\nF1 change: {f1_improvement:+.3f}, Recall change: {recall_improvement:+.3f} "
              f"(meaningful margin: {MEANINGFUL_MARGIN})")
        print(f"-> v3 meaningfully better than v2 = {v3_meaningfully_better}")
    else:
        v3_meaningfully_better = False
        print("attrition_model_v2.pkl not found, skipping comparison.")

    # ============================================================ PHASE 9 ==
    section("PHASE 9 - Global SHAP summary (sample of training data)")
    shap_summary = None
    try:
        import shap
        background = X_train[final_feature_list].sample(n=min(100, len(X_train)), random_state=RANDOM_STATE)
        sample_for_shap = X_train[final_feature_list].sample(n=min(150, len(X_train)), random_state=RANDOM_STATE)

        def f(x):
            row_df = pd.DataFrame(np.array(x), columns=final_feature_list)
            return final_model.predict_proba(row_df)[:, 1]

        explainer = shap.Explainer(f, background)
        sv = explainer(sample_for_shap)
        mean_abs_shap = pd.Series(np.abs(sv.values).mean(axis=0), index=final_feature_list).sort_values(ascending=False)
        shap_summary = mean_abs_shap
        print(mean_abs_shap.head(10))
    except Exception as e:
        print(f"SHAP global summary skipped: {e}")

    # ---- save small real background sample for the app's per-employee SHAP ----
    X_train[final_feature_list].sample(n=min(50, len(X_train)), random_state=RANDOM_STATE).to_csv(
        MODELS_DIR / "shap_background_v3.csv", index=False
    )

    # ============================================================ SAVE ALL ==
    section("Saving artifacts")
    import joblib
    metadata = {
        "model_type": best_name,
        "balancing": best_balancing[best_name],
        "preprocessing_version": "src/preprocessing.py (unchanged)",
        "feature_version": feature_version,
        "feature_list": selected_features,
        "n_features": len(selected_features),
        "calibrated": use_calibration,
        "threshold": chosen_threshold,
        "operating_points": operating_points,
        "training_seed": RANDOM_STATE,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_metrics": final_test_metrics,
        "test_f1_bootstrap_95ci": [ci_low, ci_high],
    }

    if v3_meaningfully_better:
        final_model_to_save = final_model
        joblib.dump(final_model_to_save, V3_MODEL_PATH)
        print(f"v3 IS meaningfully better than v2 -> saved {V3_MODEL_PATH}")
        with open(MODELS_DIR / "model_v3_metadata.json", "w") as f_out:
            json.dump(metadata, f_out, indent=2)
    else:
        print("v3 is NOT meaningfully better than v2 on the held-out test set -> "
              "attrition_model_v3.pkl was NOT created. attrition_model_v2.pkl remains "
              "the recommended candidate. Full experiment results are still saved below "
              "for transparency.")

    metrics_out = {
        "trained_at": metadata["trained_at"],
        "n_rows": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "feature_engineering_ablation": fe_df.to_dict(orient="records"),
        "use_engineered_features": bool(use_engineered),
        "feature_version": feature_version,
        "imbalance_comparison": imbalance_df.to_dict(orient="records"),
        "best_balancing_per_model": best_balancing,
        "tuning_summary": tuning_summary,
        "stability_cv_mean_std": stability_df.to_dict(orient="records"),
        "oof_model_selection": selection_df.drop(columns=["rank_score"]).to_dict(orient="records"),
        "best_model": best_name,
        "operating_points": operating_points,
        "chosen_threshold": chosen_threshold,
        "calibration": {
            "brier_raw": brier_raw, "brier_calibrated": brier_calibrated,
            "roc_auc_raw": roc_auc_raw, "roc_auc_calibrated": roc_auc_calibrated,
            "best_f1_raw": best_f1_raw, "best_f1_calibrated": best_f1_calibrated,
            "use_calibration": use_calibration,
        },
        "feature_selection": {
            "n_all_features": len(FEATURE_LIST), "n_selected_features": len(selected_features),
            "use_feature_selection": use_feature_selection, "selected_features": selected_features,
        },
        "final_test_metrics_at_chosen_threshold": final_test_metrics,
        "final_test_metrics_at_0.5": final_test_metrics_05,
        "test_f1_bootstrap_95ci": [ci_low, ci_high],
        "v2_vs_v3_same_test_set": v2_vs_v3,
        "v3_meaningfully_better_than_v2": bool(v3_meaningfully_better),
        "global_shap_top10": shap_summary.head(10).to_dict() if shap_summary is not None else None,
        "metadata": metadata,
    }
    with open(MODELS_DIR / "metrics_v3.json", "w") as f_out:
        json.dump(metrics_out, f_out, indent=2)
    with open(MODELS_DIR / "roc_data_v3.json", "w") as f_out:
        json.dump({best_name: roc_data_v3}, f_out, indent=2)
    if importances is not None:
        importances.to_csv(MODELS_DIR / "feature_importance_v3.csv", header=["importance"])

    print(f"\nDone in {time.time() - t0:.1f}s. See models/metrics_v3.json for full results.")


if __name__ == "__main__":
    main()
