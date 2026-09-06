"""
Optional engineered features, layered on top of the base 49-column encoding
in src/preprocessing.py.

Each feature is built ONLY from fields present on an employee's HR record at
prediction time (nothing derived from Attrition), so none of them can leak
the target. Every one is tested via cross-validation in src/train_model_v3.py
before being kept - this module just defines what they are.

All ratio features guard against division by zero (TotalWorkingYears and
YearsAtCompany are both 0 for some real employees in this dataset - 11 and 44
rows respectively) by adding 1 to the denominator, which keeps the ratio
well-defined (0/1 = 0) without fabricating experience that isn't there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = [
    "CompanyTenureRatio",
    "RoleTenureRatio",
    "ManagerTenureRatio",
    "PromotionGapRatio",
    "ExternalExperienceRatio",
    "IncomePerJobLevel",
    "SatisfactionIndex",
    "OvertimeLowBalance",
    "CareerExperienceRatio",
]

FEATURE_DESCRIPTIONS = {
    "CompanyTenureRatio": "YearsAtCompany / (TotalWorkingYears + 1) - how much of this "
        "employee's whole career has been spent at this company.",
    "RoleTenureRatio": "YearsInCurrentRole / (YearsAtCompany + 1) - how much of their time "
        "at this company has been in the current role (low = recently moved/promoted).",
    "ManagerTenureRatio": "YearsWithCurrManager / (YearsAtCompany + 1) - how much of their "
        "company tenure has been under the current manager.",
    "PromotionGapRatio": "YearsSinceLastPromotion / (YearsAtCompany + 1) - proportion of "
        "company tenure spent without a promotion.",
    "ExternalExperienceRatio": "(TotalWorkingYears - YearsAtCompany) / (TotalWorkingYears + 1) "
        "- how much of their total career experience was gained elsewhere.",
    "IncomePerJobLevel": "MonthlyIncome / JobLevel - pay relative to seniority level.",
    "SatisfactionIndex": "Mean of EnvironmentSatisfaction, JobSatisfaction, "
        "RelationshipSatisfaction, JobInvolvement, WorkLifeBalance - a single composite "
        "satisfaction signal.",
    "OvertimeLowBalance": "OverTime AND WorkLifeBalance <= 2 - flags the specific combination "
        "of working overtime with a self-reported poor work-life balance.",
    "CareerExperienceRatio": "TotalWorkingYears / Age - rough proxy for how much of the "
        "employee's life has been spent in the workforce.",
}


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X (the base 49-column encoding) with engineered
    columns appended. Does not mutate X, does not reorder existing columns."""
    X = X.copy()

    X["CompanyTenureRatio"] = X["YearsAtCompany"] / (X["TotalWorkingYears"] + 1)
    X["RoleTenureRatio"] = X["YearsInCurrentRole"] / (X["YearsAtCompany"] + 1)
    X["ManagerTenureRatio"] = X["YearsWithCurrManager"] / (X["YearsAtCompany"] + 1)
    X["PromotionGapRatio"] = X["YearsSinceLastPromotion"] / (X["YearsAtCompany"] + 1)
    X["ExternalExperienceRatio"] = (X["TotalWorkingYears"] - X["YearsAtCompany"]) / (X["TotalWorkingYears"] + 1)
    X["IncomePerJobLevel"] = X["MonthlyIncome"] / X["JobLevel"]
    X["SatisfactionIndex"] = X[[
        "EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
        "JobInvolvement", "WorkLifeBalance",
    ]].mean(axis=1)
    X["OvertimeLowBalance"] = ((X["OverTime"] == 1) & (X["WorkLifeBalance"] <= 2)).astype(int)
    X["CareerExperienceRatio"] = X["TotalWorkingYears"] / X["Age"]

    return X
