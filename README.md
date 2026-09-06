# Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave a
company, deployed as an interactive Streamlit application for HR decision support.

## 1. Overview

The system estimates each employee's probability of attrition (leaving the company)
from demographic, job, compensation and satisfaction data, and surfaces that estimate
through a Streamlit app with a live prediction form, an HR analytics dashboard, and a
model performance page. It is a decision-support tool, not a certainty - predictions
should inform, not replace, manager judgement.

## 2. Problem Statement

Employee attrition is costly and often avoidable if at-risk employees are identified
early. This is framed as a **binary classification** problem: predict `Attrition`
(Yes/No) from an employee's HR record, with particular attention to **recall on the
Attrition = Yes class**, since for an HR early-warning system, missing an employee who
is about to leave is generally more costly than a false alarm.

## 3. Dataset

**IBM HR Analytics Employee Attrition & Performance** (`IBM Dataset.csv`), 1,470
employee records, 35 columns, no missing values, no duplicate rows.

| Attrition | Count | % |
|---|---|---|
| No  | 1,233 | 83.9% |
| Yes | 237   | 16.1% |

## 4. Dataset Statistics

26 numeric columns (Age, DailyRate, MonthlyIncome, JobSatisfaction, WorkLifeBalance,
YearsAtCompany, ...), 9 categorical/object columns (Attrition, BusinessTravel,
Department, EducationField, Gender, JobRole, MaritalStatus, Over18, OverTime).
`EmployeeNumber`, `EmployeeCount`, `Over18` and `StandardHours` are dropped before
modeling - an identifier and three constants carry no signal.

## 5. Data Preprocessing

- `Attrition`, `Gender`, `OverTime` mapped to 0/1
- `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`
  one-hot encoded (`pandas.get_dummies`) -> **49 base features**
- Numeric features standardized (`StandardScaler`) only inside the Logistic Regression
  pipeline - tree-based models don't need it
- Reproducible stratified 80/20 train/test split (`random_state=42`), used identically
  across every model version in this project so results are genuinely comparable
- The test set is never touched until final evaluation; SMOTE, scaling, and
  hyperparameter search all run **inside** cross-validation folds on the training set

The exact encoding rules live in `src/preprocessing.py` and are used by **both**
training and the Streamlit app, so a prediction request is guaranteed to be encoded
exactly the way the training data was encoded.

## 6. Feature Engineering

Nine features were engineered on top of the base 49 (`src/feature_engineering.py`),
each built only from fields available at prediction time (no target leakage):

| Feature | What it represents |
|---|---|
| CompanyTenureRatio | Share of total working years spent at this company |
| RoleTenureRatio | Share of company tenure spent in the current role |
| ManagerTenureRatio | Share of company tenure spent under the current manager |
| PromotionGapRatio | Share of company tenure spent without a promotion |
| ExternalExperienceRatio | Share of career experience gained elsewhere |
| IncomePerJobLevel | Pay relative to seniority level |
| SatisfactionIndex | Composite of 5 satisfaction/involvement scores |
| OvertimeLowBalance | Overtime AND poor work-life balance, combined |
| CareerExperienceRatio | Total working years relative to age |

**Tested, not assumed:** compared via 5-fold CV (Logistic Regression + XGBoost, training
data only), engineered features improved mean F1 from 0.468 to 0.487 (+0.019, above the
pre-set 0.01 keep/discard margin) - **kept**. All ratio features guard against
division-by-zero (11 employees have `TotalWorkingYears=0`, 44 have `YearsAtCompany=0`).

## 7. Class Imbalance

Three approaches compared per model (5-fold CV, default hyperparams, engineered
features, training data only - SMOTE fit inside folds only, never on the test set):

| Model | None | Class weighting | SMOTE | Best |
|---|---|---|---|---|
| Logistic Regression | F1=0.583 | F1=0.497 | F1=0.578 | **None** |
| Random Forest | F1=0.295 | F1=0.268 | F1=0.408 | **SMOTE** |
| XGBoost | F1=0.478 | F1=0.515 | F1=0.476 | **Class weighting** |
| CatBoost | F1=0.432 | F1=0.528 | F1=0.481 | **Class weighting** |

No single method wins for every model - each model kept whichever balancing method
actually helped it, decided by evidence, not assumption.

## 8. Models Tested

Logistic Regression, Random Forest, XGBoost, and **CatBoost** (added this round), each
with its best balancing method from Section 7, tuned with `RandomizedSearchCV`
(20 iterations, 5-fold stratified CV, `scoring="f1"`).

## 9. Hyperparameter Tuning

| Model | Best parameters |
|---|---|
| Logistic Regression | `C=3, penalty=l2, solver=liblinear` |
| Random Forest | `n_estimators=300, max_depth=6, min_samples_split=6, min_samples_leaf=2, max_features=log2` |
| XGBoost | `n_estimators=500, max_depth=5, learning_rate=0.01, subsample=0.7, reg_lambda=5.0, min_child_weight=1` |
| CatBoost | `iterations=400, depth=5, learning_rate=0.03, l2_leaf_reg=5, subsample=0.7` |

## 10. Model Stability (mean ± std across 5 folds)

| Model | F1 | Recall | ROC-AUC |
|---|---|---|---|
| **Logistic Regression** | **0.590 ± 0.090** | 0.495 ± 0.099 | 0.838 ± 0.028 |
| Random Forest | 0.478 ± 0.091 | 0.432 ± 0.089 | 0.780 ± 0.037 |
| XGBoost | 0.571 ± 0.029 | 0.521 ± 0.026 | 0.820 ± 0.029 |
| CatBoost | 0.571 ± 0.051 | 0.521 ± 0.051 | 0.823 ± 0.033 |

Logistic Regression has the highest mean F1 but also the **highest fold-to-fold
variance** of the four - one fold dropped to F1=0.42 while its best folds reached
0.65-0.68 (confirmed on an independent re-check with fixed hyperparameters). XGBoost
and CatBoost are markedly more consistent (std ~0.03-0.05). Logistic Regression was
still selected because even its worst fold is competitive with the other models'
typical performance, and its mean holds up across independent checks - but this is a
real trade-off, not a clean, unambiguous win, and is reported as such rather than
glossed over.

## 11. Feature Selection

Compared all 58 features (49 base + 9 engineered) against a reduced ~90%-cumulative-
importance subset (36 features) for the winning model, via 5-fold CV: all-features
F1=0.583±0.100 vs. selected-subset F1=0.627±0.080. The subset was **adopted** (better
mean, tighter variance) - full ranked importances are in `models/feature_importance_v3.csv`.

## 12. Threshold Optimization / Operating Points

0.5 is not assumed to be the best cutoff. Thresholds 0.20-0.70 were swept using
**out-of-fold predictions on the training set only** (never the test set). Three
operating points are exposed as presets next to the **HR Risk Tolerance** slider in the
app (for the deployed v2 candidate model):

| Operating point | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| High Recall | 0.20 | 0.417 | 0.711 | 0.525 |
| **Balanced** (default) | **0.35** | 0.617 | 0.595 | **0.606** |
| High Precision | 0.60 | 0.817 | 0.353 | 0.493 |

The HR Risk Tolerance slider changes only this cutoff - never the underlying model or
its predicted probability.

## 13. Calibration

Checked whether the model's predicted probabilities are reliable (a 70% prediction
should correspond to roughly 70% of such employees actually leaving), via nested
cross-validation (`CalibratedClassifierCV`, sigmoid method, fit on training data only):

| | Brier score (lower=better) | ROC-AUC | Best F1 |
|---|---|---|---|
| Raw probabilities | 0.0833 | 0.850 | 0.642 |
| Calibrated (Platt/sigmoid) | 0.0856 | 0.852 | 0.637 |

Calibration made the Brier score slightly **worse**, not better, so it was **not
adopted** - the app reports raw model probabilities. A probability shown in the app is
an estimated model probability, not a guarantee that an employee will leave.

## 14. Evaluation Metrics

Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix are computed for
every model - see `models/metrics.json` and `models/metrics_v3.json`, or the app's
**Model Performance** tab.

## 15. Final Selected Model

The production candidate remains **`attrition_model_v2.pkl`** (Logistic Regression, no
SMOTE, `C=0.3`, threshold 0.35, base 49 features): test accuracy 0.840, precision 0.500,
recall 0.447, F1 0.472, ROC-AUC 0.808.

**This round's v3 experiment (feature engineering + CatBoost + calibration check +
feature selection, `src/train_model_v3.py`) did NOT produce a meaningfully better
model** on the same held-out test set (F1 change -0.007, recall change -0.021 vs. v2 -
both within noise; a bootstrap 95% CI on test F1 was [0.328, 0.592], underscoring how
much a 294-row test set can vary run to run). **`attrition_model_v3.pkl` was therefore
NOT created.** `attrition_model.pkl` (production) was also not touched. Full v3 results
are saved in `models/metrics_v3.json` and shown in the app regardless of this negative
outcome - reported honestly rather than hidden.

## 16. Explainability

- Tree/linear feature importance for the best model (`models/feature_importance.csv`,
  `models/feature_importance_v3.csv`)
- Global SHAP summary computed on a training-data sample (Model Performance tab / v3 results)
- **Per-employee SHAP explanation** on the Prediction tab (on-demand checkbox), split
  into **top factors increasing risk** and **top factors decreasing risk** for that
  specific employee
- Falls back to global feature importance if SHAP isn't installed or fails
- Feature importance/SHAP show **association learned from historical data**, not
  causation - the app never claims a factor "caused" attrition

Top predictors (v3, tree/linear importance): `OverTime`, `MonthlyIncome`,
`ExternalExperienceRatio`, `JobLevel`, `NumCompaniesWorked`.

## 17. Streamlit Application

Three tabs:
- **Prediction** - "How this prediction works" walkthrough, employee input form, model
  selector (production vs. candidate v2), HR Risk Tolerance threshold slider with
  data-driven operating-point presets (High Recall / Balanced / High Precision), risk
  gauge, and a per-prediction explanation split into risk-increasing / risk-decreasing factors.
- **Dashboard** - KPIs and interactive Plotly charts (attrition distribution, by
  department, job role, overtime, job level, income range, tenure, job satisfaction),
  filterable by department, job role, gender, overtime and marital status, with graceful
  handling of filter combinations that match zero employees.
- **Model Performance** - model comparison table, out-of-fold selection table, operating
  points, confusion matrix, ROC curve, threshold sweep, feature importance, the full
  old-vs-v2 comparison (including a data-leakage caveat), and the v3 experiment writeup.

## 18. Dashboard

Source: `IBM Dataset.csv` (raw, unencoded, for readable labels). KPIs: total employees,
employees left, attrition rate - all recomputed live from the filtered subset.

## 19. Limitations

- Only 237 positive (Attrition=Yes) examples in the whole dataset, and ~190 in any
  training split - this caps how much hyperparameter tuning, SMOTE, or feature
  engineering can realistically improve recall/F1, and makes fold-to-fold variance
  (Section 10) a real concern, not a rounding error.
- The 294-row test set means metric differences of a few percentage points between
  models are often within noise (see the bootstrap CI in Section 15) - don't over-read
  small deltas in the comparison tables.
- `attrition_model.pkl`'s original train/test split was unseeded and is not
  reproducible, so it can only be fairly compared against new models using its own
  historical self-reported numbers, not a fresh identical test set.
- This is IBM's synthetic/anonymized HR dataset, not the deploying organization's real
  data - patterns learned here (e.g. OverTime as a top predictor) may not transfer
  directly to a different workforce.
- Predicted probabilities are the model's best estimate, not a guarantee, and were
  found to already be reasonably calibrated without further adjustment (Section 13).

## 20. Installation

```bash
git clone <this-repo>
cd employee_attrition_ml
pip install -r requirements.txt
```

Requires Python 3.10+ (developed/tested on 3.13).

## 21. How to Train

```bash
# Base pipeline - writes attrition_model_v2.pkl and models/metrics.json etc.
python -m src.train_model

# v3 experiment - feature engineering, +CatBoost, calibration, feature selection.
# Only writes attrition_model_v3.pkl if it's actually meaningfully better than v2.
# Needs one extra dependency not required by the app itself:
pip install -r requirements-train.txt
python -m src.train_model_v3
```

## 22. How to Run Streamlit

```bash
streamlit run app.py
```

`attrition_model.pkl` (production) must be present in the project root; `IBM
Dataset.csv` must be present for the dashboard and for retraining.

## 23. Deployment

Deployable as-is on Streamlit Cloud: point it at `app.py`, ensure `IBM Dataset.csv`,
`attrition_model.pkl`, `attrition_model_v2.pkl` and `models/*.json`/`models/*.csv` are
committed to the repo. If `shap` fails to build on a given platform, remove it from
`requirements.txt` - the app degrades gracefully to feature-importance-only explanations.

## 24. Future Improvements

- Collect more attrition-positive examples; 237 positive rows is the binding constraint
  on almost every metric in this project.
- Retrain `attrition_model.pkl` itself with a fixed seed, so a fully fair, non-overlapping
  comparison against candidate models becomes possible.
- Track prediction outcomes over time to validate the model against real attrition
  events, not just a held-out split of historical data.
- Revisit CatBoost/XGBoost with a larger tuning budget if a bigger dataset becomes
  available - their much lower fold-to-fold variance (Section 10) than Logistic
  Regression suggests they may generalize more reliably at scale, even though they
  didn't win outright here.

## Project Structure

```
app.py                          Streamlit application
src/preprocessing.py            Shared base encoding (training + app use the same code)
src/feature_engineering.py      Engineered feature definitions
src/train_model.py              v2 pipeline (base features, LR/RF/XGBoost)
src/train_model_v3.py           v3 experiment (feature engineering, +CatBoost, calibration,
                                 feature selection, stability reporting)
notebooks/attrition_model.ipynb Original exploratory notebook (kept for history)
attrition_model.pkl             Production model (XGBoost, in use by the app)
attrition_model_v2.pkl          Candidate model (Logistic Regression, tuned) - in production tab
models/metrics.json             v2 pipeline results
models/metrics_v3.json          v3 experiment results (full detail, incl. why it wasn't adopted)
models/feature_importance*.csv
models/roc_data*.json
models/shap_background*.csv     Small real-data samples used as SHAP's background distribution
IBM Dataset.csv                 Training data (IBM HR Analytics Employee Attrition)
requirements.txt                App + base pipeline dependencies
requirements-train.txt          Adds catboost, needed only for src/train_model_v3.py
```
