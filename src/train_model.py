"""
Employee Attrition - training pipeline.

Loads `IBM Dataset.csv`, cleans/encodes it with src.preprocessing (the same
module the Streamlit app uses for prediction, so training and inference
stay in sync), trains and compares Logistic Regression / Random Forest /
XGBoost with and without SMOTE, tunes the two tree-based models, picks a
decision threshold from out-of-fold predictions, evaluates the result
against the existing attrition_model.pkl on an identical held-out test
set, and writes:

  attrition_model_v2.pkl          the candidate model (NOT overwriting
                                   attrition_model.pkl)
  models/metrics.json             everything the Streamlit "Model
                                   Performance" tab needs to display
  models/feature_importance.csv   tree-based feature importances
  models/roc_data.json            fpr/tpr points for the ROC tab

Run with:  python -m src.train_model
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict,
    cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.preprocessing import MODEL_FEATURES, encode_raw_dataframe

RANDOM_STATE = 42
DATA_PATH = Path("IBM Dataset.csv")
MODELS_DIR = Path("models")
V2_MODEL_PATH = Path("attrition_model_v2.pkl")
OLD_MODEL_PATH = Path("attrition_model.pkl")

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# The only legitimate (non-leaked) out-of-sample numbers we have for the
# existing attrition_model.pkl: the classification_report it printed in
# notebooks/attrition_model.ipynb (cell 26) on ITS OWN held-out test split
# at the time it was trained. That split was unseeded and is not
# reproducible, so we cannot recompute it -- these are the actual printed
# values, transcribed as-is, not recomputed or estimated.
HISTORICAL_OLD_MODEL_METRICS = {
    "source": "notebooks/attrition_model.ipynb, cell 26 output (XGBoost Results), original training run",
    "note": (
        "Unseeded train_test_split -> not reproducible and not the same rows as our "
        "test set, but this IS the model's own genuine held-out evaluation from when "
        "it was trained, unlike the same-test-set comparison above which is confounded "
        "by likely row overlap."
    ),
    "test_support": 294,
    "accuracy": 0.88,
    "precision_attrition_yes": 0.80,
    "recall_attrition_yes": 0.40,
    "f1_attrition_yes": 0.53,
    "roc_auc": None,
}


def load_and_inspect() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}: shape={df.shape}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("Target distribution:")
    print(df["Attrition"].value_counts())
    print(f"Attrition rate: {(df['Attrition'] == 'Yes').mean():.1%}")
    return df


def make_candidates(use_smote: bool) -> dict:
    """Build the three candidate pipelines, optionally with SMOTE."""
    def wrap(steps):
        return ImbPipeline(([("smote", SMOTE(random_state=RANDOM_STATE))] if use_smote else []) + steps)

    return {
        "Logistic Regression": wrap([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": wrap([
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "XGBoost": wrap([
            ("clf", XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
            )),
        ]),
    }


def cv_scores(pipeline, X, y, cv) -> dict:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    result = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {k: float(np.mean(result[f"test_{k}"])) for k in scoring}


def compare_smote_vs_baseline(X_train, y_train) -> pd.DataFrame:
    """Compare each algorithm with vs without SMOTE using 5-fold CV on the
    training set only (SMOTE is fit inside each fold, never on validation
    data or the held-out test set)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for use_smote in (False, True):
        candidates = make_candidates(use_smote)
        for name, pipe in candidates.items():
            scores = cv_scores(pipe, X_train, y_train, cv)
            rows.append({"model": name, "smote": use_smote, **scores})
    return pd.DataFrame(rows)


PARAM_DISTRIBUTIONS = {
    "Random Forest": {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth": [6, 8, 10, 12, None],
        "clf__min_samples_split": [2, 4, 6],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
        "clf__class_weight": ["balanced", "balanced_subsample", None],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    },
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 0.3, 1, 3, 10],
        "clf__class_weight": ["balanced", None],
    },
}


def tune_model(name: str, base_pipeline, X_train, y_train, cv) -> tuple:
    search = RandomizedSearchCV(
        base_pipeline,
        PARAM_DISTRIBUTIONS[name],
        n_iter=20,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def evaluate_on_test(model, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred).tolist()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": cm,
    }


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


def main():
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_and_inspect()
    X, y = encode_raw_dataframe(df)
    assert list(X.columns) == MODEL_FEATURES

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train attrition rate: {y_train.mean():.1%}, Test attrition rate: {y_test.mean():.1%}")

    # --- Step 1: SMOTE vs no-SMOTE, cross-validated on training data only ---
    print("\n=== SMOTE vs baseline (5-fold CV on training data) ===")
    smote_comparison = compare_smote_vs_baseline(X_train, y_train)
    print(smote_comparison.to_string(index=False))

    # Decide per-model whether SMOTE helps, based on CV F1 for the positive class.
    use_smote_for = {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        sub = smote_comparison[smote_comparison["model"] == name]
        f1_no = sub[sub["smote"] == False]["f1"].iloc[0]
        f1_yes = sub[sub["smote"] == True]["f1"].iloc[0]
        use_smote_for[name] = bool(f1_yes > f1_no)
        print(f"{name}: F1 without SMOTE={f1_no:.3f}, with SMOTE={f1_yes:.3f} "
              f"-> using SMOTE = {use_smote_for[name]}")

    # --- Step 2: hyperparameter tuning for each model with its chosen SMOTE setting ---
    print("\n=== Hyperparameter tuning (RandomizedSearchCV, scoring=f1, 5-fold CV) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned_models, tuning_summary = {}, {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        base_pipeline = make_candidates(use_smote_for[name])[name]
        best_estimator, best_params, best_cv_f1 = tune_model(name, base_pipeline, X_train, y_train, cv)
        tuned_models[name] = best_estimator
        tuning_summary[name] = {
            "smote_used": use_smote_for[name],
            "best_params": {k.replace("clf__", ""): v for k, v in best_params.items()},
            "best_cv_f1": best_cv_f1,
        }
        print(f"\n{name} (SMOTE={use_smote_for[name]}): best CV F1={best_cv_f1:.3f}")
        print(f"  best params: {tuning_summary[name]['best_params']}")

    # --- Step 3: test-set evaluation at threshold=0.5, for ALL tuned models (reporting only) ---
    print("\n=== Test-set evaluation (threshold=0.5, all tuned models) ===")
    comparison_rows = []
    test_results = {}
    for name, model in tuned_models.items():
        result = evaluate_on_test(model, X_test, y_test, threshold=0.5)
        test_results[name] = result
        comparison_rows.append({"model": name, **{k: v for k, v in result.items() if k != "confusion_matrix"}})
        print(f"{name}: acc={result['accuracy']:.3f} prec={result['precision']:.3f} "
              f"recall={result['recall']:.3f} f1={result['f1']:.3f} roc_auc={result['roc_auc']:.3f}")
    comparison_df = pd.DataFrame(comparison_rows)

    # --- Step 4: threshold optimization using out-of-fold predictions on TRAIN only, ---
    # --- for every tuned model. Model selection uses these OOF scores (never the test ---
    # --- set), so the held-out test set stays untouched until final reporting. ---
    print("\n=== Threshold optimization (out-of-fold predictions on training data) ===")
    threshold_sweeps = {}
    oof_selection_rows = []
    for name, model in tuned_models.items():
        oof_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        sweep = threshold_sweep(y_train, oof_proba)
        threshold_sweeps[name] = sweep
        best_row = max(sweep, key=lambda r: r["f1"])
        oof_roc_auc = float(roc_auc_score(y_train, oof_proba))
        oof_selection_rows.append({
            "model": name,
            "best_threshold": best_row["threshold"],
            "oof_precision": best_row["precision"],
            "oof_recall": best_row["recall"],
            "oof_f1": best_row["f1"],
            "oof_roc_auc": oof_roc_auc,
        })
        print(f"\n{name} OOF threshold sweep:")
        for row in sweep:
            print(f"  t={row['threshold']:.2f}  precision={row['precision']:.3f}  "
                  f"recall={row['recall']:.3f}  f1={row['f1']:.3f}")
        print(f"  -> best threshold={best_row['threshold']} (F1={best_row['f1']:.3f}, "
              f"recall={best_row['recall']:.3f}), OOF ROC-AUC={oof_roc_auc:.3f}")

    oof_selection_df = pd.DataFrame(oof_selection_rows)
    # Selection is based on out-of-fold (training) performance at each model's own
    # optimal threshold, weighted toward recall/F1 for the positive (Attrition=Yes)
    # class, since missing an at-risk employee is costlier than a false alarm.
    oof_selection_df["rank_score"] = (
        oof_selection_df["oof_f1"] + 0.5 * oof_selection_df["oof_recall"] + 0.25 * oof_selection_df["oof_roc_auc"]
    )
    print("\n=== Model selection (based on out-of-fold training performance, not test set) ===")
    print(oof_selection_df.to_string(index=False))

    best_row = oof_selection_df.sort_values("rank_score", ascending=False).iloc[0]
    best_name = best_row["model"]
    best_model = tuned_models[best_name]
    chosen_threshold = float(best_row["best_threshold"])
    sweep = threshold_sweeps[best_name]
    print(f"\nSelected best model: {best_name} (threshold={chosen_threshold})")

    # Final test-set numbers at the chosen threshold (for reporting only;
    # neither model nor threshold was selected using test data).
    final_test_result = evaluate_on_test(best_model, X_test, y_test, threshold=chosen_threshold)
    print(f"\n{best_name} on test set @ threshold={chosen_threshold}: {final_test_result}")

    # --- Step 6: compare against the existing attrition_model.pkl on the SAME test set ---
    # CAVEAT: notebooks/attrition_model.ipynb used an UNSEEDED train_test_split, so we
    # cannot know which rows attrition_model.pkl was actually trained on. With only
    # 1470 rows total and ~80% used for its training, our "test" split here very likely
    # overlaps heavily with its own training data. Any strong score for the old model
    # below reflects that overlap (memorization), not genuine generalization -- it is
    # NOT a fair apples-to-apples comparison, and is reported only for transparency.
    print("\n=== Comparison against existing attrition_model.pkl (SAME test set - see leakage caveat) ===")
    old_vs_new = None
    if OLD_MODEL_PATH.exists():
        import joblib
        old_model = joblib.load(OLD_MODEL_PATH)
        old_result = evaluate_on_test(old_model, X_test, y_test, threshold=0.5)
        new_result_at_05 = evaluate_on_test(best_model, X_test, y_test, threshold=0.5)
        old_vs_new = {
            "caveat": (
                "attrition_model.pkl was trained with an unseeded train_test_split in "
                "the original notebook, so its training rows are unknown and likely "
                "overlap with this test set. Its score here is NOT a reliable measure "
                "of generalization and should not be read as 'old model is better'."
            ),
            "attrition_model.pkl (existing, threshold=0.5, likely leaked)": old_result,
            f"{best_name} v2 (threshold=0.5, clean held-out test)": new_result_at_05,
            f"{best_name} v2 (threshold={chosen_threshold}, tuned, clean held-out test)": final_test_result,
        }
        for label, res in old_vs_new.items():
            if label == "caveat":
                continue
            print(f"{label}: acc={res['accuracy']:.3f} prec={res['precision']:.3f} "
                  f"recall={res['recall']:.3f} f1={res['f1']:.3f} roc_auc={res['roc_auc']:.3f}")
        print(f"NOTE: {old_vs_new['caveat']}")
    else:
        print("attrition_model.pkl not found, skipping comparison.")

    # --- Step 7: ROC curve data for the best model ---
    proba_test = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_score(y_test, proba_test))}

    # --- Step 8: feature importance (tree-based) ---
    feature_importance = None
    inner_clf = best_model.named_steps["clf"]
    if hasattr(inner_clf, "feature_importances_"):
        feature_importance = (
            pd.Series(inner_clf.feature_importances_, index=MODEL_FEATURES)
            .sort_values(ascending=False)
        )
    elif hasattr(inner_clf, "coef_"):
        feature_importance = (
            pd.Series(np.abs(inner_clf.coef_[0]), index=MODEL_FEATURES)
            .sort_values(ascending=False)
        )
    if feature_importance is not None:
        feature_importance.to_csv(MODELS_DIR / "feature_importance.csv", header=["importance"])
        print("\nTop 10 features:")
        print(feature_importance.head(10))

    # --- Save everything ---
    import joblib
    joblib.dump(best_model, V2_MODEL_PATH)
    print(f"\nSaved candidate model to {V2_MODEL_PATH} (attrition_model.pkl left untouched)")

    # Small real-data sample for SHAP's background distribution in the app
    # (never synthetic - drawn straight from the actual training rows).
    X_train.sample(n=min(50, len(X_train)), random_state=RANDOM_STATE).to_csv(
        MODELS_DIR / "shap_background.csv", index=False
    )

    metrics_out = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(DATA_PATH),
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_attrition_rate": float(y_test.mean()),
        "smote_comparison": smote_comparison.to_dict(orient="records"),
        "use_smote_for": use_smote_for,
        "tuning_summary": tuning_summary,
        "model_comparison_at_0.5": comparison_df.to_dict(orient="records"),
        "oof_model_selection": oof_selection_df.to_dict(orient="records"),
        "best_model": best_name,
        "threshold_sweep_oof_train": sweep,
        "chosen_threshold": chosen_threshold,
        "best_model_test_at_0.5": test_results[best_name],
        "best_model_test_at_chosen_threshold": final_test_result,
        "old_vs_new_same_test_set_caveat_leakage": old_vs_new,
        "old_model_historical_self_reported": HISTORICAL_OLD_MODEL_METRICS,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    with open(MODELS_DIR / "roc_data.json", "w") as f:
        json.dump({best_name: roc_data}, f, indent=2)

    print(f"\nDone in {time.time() - t0:.1f}s. Metrics written to {MODELS_DIR}/metrics.json")


if __name__ == "__main__":
    main()
