"""
Shared preprocessing for the Employee Attrition project.

This module is the single source of truth for how raw IBM HR Attrition
data is turned into the 49 numeric columns the model expects. It is used
by BOTH the training pipeline (src/train_model.py) and the Streamlit app
(app.py), so a prediction request is guaranteed to be encoded exactly the
way the training data was encoded.

Encoding scheme (matches notebooks/attrition_model.ipynb, the notebook the
original attrition_model.pkl was trained with):
- Attrition:  Yes -> 1, No -> 0 (target)
- Gender:     Female -> 1, Male -> 0
- OverTime:   Yes -> 1, No -> 0
- BusinessTravel, Department, EducationField, JobRole, MaritalStatus are
  one-hot encoded with pandas.get_dummies (BusinessTravel has no prefix,
  the others use Department_/Education_/Role_/Status_ prefixes).
- EmployeeNumber, EmployeeCount, Over18, StandardHours are dropped: they
  are either an identifier or constant across every row in this dataset,
  so they carry no predictive signal.
"""
from __future__ import annotations

import pandas as pd

# Exact column order the existing attrition_model.pkl (and the notebook it
# came from) was trained on. Do not reorder — tree ensembles trained with
# scikit-learn/XGBoost validate column order via feature_names_in_.
MODEL_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
    "Non-Travel", "Travel_Frequently", "Travel_Rarely",
    "Department_Human Resources", "Department_Research & Development", "Department_Sales",
    "Education_Human Resources", "Education_Life Sciences", "Education_Marketing",
    "Education_Medical", "Education_Other", "Education_Technical Degree",
    "Role_Healthcare Representative", "Role_Human Resources", "Role_Laboratory Technician",
    "Role_Manager", "Role_Manufacturing Director", "Role_Research Director",
    "Role_Research Scientist", "Role_Sales Executive", "Role_Sales Representative",
    "Status_Divorced", "Status_Married", "Status_Single",
]

DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
EDUCATION_FIELDS = ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]
BUSINESS_TRAVEL_OPTIONS = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
JOB_ROLES = [
    "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
    "Manufacturing Director", "Research Director", "Research Scientist",
    "Sales Executive", "Sales Representative",
]
MARITAL_STATUSES = ["Divorced", "Married", "Single"]

# Job Level that most commonly corresponds to each Job Role in the training
# data (mode of JobLevel within each JobRole group, computed from
# `IBM Dataset.csv`). Used to auto-fill Job Level in the app when a HR user
# picks a Job Role, since Job Level is not something an HR user filling in
# a candidate/employee profile would typically type in by hand.
# Standard scale definitions documented for the IBM HR Attrition dataset
# (used only to make the UI legible - the underlying numeric value fed to
# the model is unchanged).
EDUCATION_LEVEL_LABELS = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
SATISFACTION_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
WORK_LIFE_BALANCE_LABELS = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
PERFORMANCE_RATING_LABELS = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}

JOB_ROLE_TO_JOB_LEVEL = {
    "Healthcare Representative": 2,
    "Human Resources": 1,
    "Laboratory Technician": 1,
    "Manager": 4,
    "Manufacturing Director": 2,
    "Research Director": 3,
    "Research Scientist": 1,
    "Sales Executive": 2,
    "Sales Representative": 1,
}


def encode_raw_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode a raw IBM HR Attrition dataframe into (X, y).

    `df` must have the original IBM HR Attrition columns (Attrition,
    Gender, OverTime, BusinessTravel, Department, EducationField, JobRole,
    MaritalStatus, plus the numeric columns). Returns X with columns in
    MODEL_FEATURES order and y as 0/1 Attrition.
    """
    data = df.copy()

    data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    data["Gender"] = data["Gender"].map({"Female": 1, "Male": 0}).astype(int)
    data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0}).astype(int)

    for col, prefix in [
        ("BusinessTravel", None),
        ("Department", "Department"),
        ("EducationField", "Education"),
        ("JobRole", "Role"),
        ("MaritalStatus", "Status"),
    ]:
        dummies = pd.get_dummies(data[col], prefix=prefix, dtype=int)
        data = data.join(dummies).drop(columns=[col])

    data = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns])

    y = data["Attrition"]
    X = data.drop(columns=["Attrition"])

    missing = set(MODEL_FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f"Encoded data is missing expected columns: {sorted(missing)}")

    X = X[MODEL_FEATURES]
    return X, y


def build_single_input(raw: dict) -> pd.DataFrame:
    """Build a single-row, model-ready DataFrame from raw Streamlit inputs.

    `raw` keys are the plain field names (Age, Gender, Department, ...)
    with human-readable values (e.g. Gender="Female", not 1). Produces the
    same 49 MODEL_FEATURES columns, in the same order, as encode_raw_dataframe.
    """
    row = {
        "Age": raw["Age"],
        "DailyRate": raw["DailyRate"],
        "DistanceFromHome": raw["DistanceFromHome"],
        "Education": raw["Education"],
        "EnvironmentSatisfaction": raw["EnvironmentSatisfaction"],
        "Gender": 1 if raw["Gender"] == "Female" else 0,
        "HourlyRate": raw["HourlyRate"],
        "JobInvolvement": raw["JobInvolvement"],
        "JobLevel": raw["JobLevel"],
        "JobSatisfaction": raw["JobSatisfaction"],
        "MonthlyIncome": raw["MonthlyIncome"],
        "MonthlyRate": raw["MonthlyRate"],
        "NumCompaniesWorked": raw["NumCompaniesWorked"],
        "OverTime": 1 if raw["OverTime"] == "Yes" else 0,
        "PercentSalaryHike": raw["PercentSalaryHike"],
        "PerformanceRating": raw["PerformanceRating"],
        "RelationshipSatisfaction": raw["RelationshipSatisfaction"],
        "StockOptionLevel": raw["StockOptionLevel"],
        "TotalWorkingYears": raw["TotalWorkingYears"],
        "TrainingTimesLastYear": raw["TrainingTimesLastYear"],
        "WorkLifeBalance": raw["WorkLifeBalance"],
        "YearsAtCompany": raw["YearsAtCompany"],
        "YearsInCurrentRole": raw["YearsInCurrentRole"],
        "YearsSinceLastPromotion": raw["YearsSinceLastPromotion"],
        "YearsWithCurrManager": raw["YearsWithCurrManager"],
    }
    input_df = pd.DataFrame([row])

    input_df[f"Department_{raw['Department']}"] = 1
    input_df[f"Education_{raw['EducationField']}"] = 1
    input_df[f"Role_{raw['JobRole']}"] = 1
    input_df[f"Status_{raw['MaritalStatus']}"] = 1
    input_df[raw["BusinessTravel"]] = 1

    for col in MODEL_FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[MODEL_FEATURES]
