"""
Corporate HR Analytics data layer - loads and cleans HRDataset_v14.csv for
the Streamlit "Corporate HR Analytics" tab ONLY.

This is a completely separate data source from IBM Dataset.csv and is never
used to train, validate, or modify attrition_model.pkl / attrition_model_v2.pkl.
The two datasets have different employees, different schemas, and different
base rates (see the investigation report) - they are deliberately kept apart
(see README "Two-Layer Architecture").

FAIRNESS / PRIVACY DESIGN (not just convention - enforced structurally):
- RaceDesc, HispanicLatino, CitizenDesc are never loaded into the returned
  analytics dataframe, so no chart, filter, or insight in the app can use
  them even by accident.
- Employee_Name, EmpID, Zip, ManagerID are dropped - not needed for any
  aggregate/organizational analytics this module supports.
- ManagerName is kept (an organizational role, not a private attribute) -
  explicitly needed for Manager Analytics.
- EmploymentStatus/TermReason/DateofTermination are kept for DESCRIPTIVE
  analytics only (e.g. "why did people leave" breakdowns) - never treated
  as predictive features, since this dataset never feeds any ML model.

DATA QUALITY FIXES (see investigation report for the verification evidence):
- DeptID, PerfScoreID, PositionID are dropped - confirmed NOT to be
  reliable 1:1 encodings of their string counterparts (e.g. PerfScoreID=1
  maps to both "Fully Meets" and "PIP"). The string columns
  (Department/PerformanceScore/Position) are used instead.
- Whitespace-only duplicate categories (e.g. 'Production       ' vs
  'Production', 'Data Analyst ' vs 'Data Analyst') are stripped.
- DOB's two-digit year is corrected: default parsing puts ~42/311 birth
  years in 2051-2068 instead of 1951-1968. Fixed by subtracting 100 years
  from any parsed date that lands in the future.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HR_DATA_PATH = Path("HRDataset_v14.csv")

# The latest date recorded anywhere in the raw file (max of DateofHire,
# DateofTermination, LastPerformanceReview_Date) - used only as a stand-in
# "as of" date for computing tenure/age of currently active employees.
# Not fabricated: it's the most recent real date present in the data.
REFERENCE_DATE = pd.Timestamp("2019-02-28")

# Columns intentionally never loaded into the analytics dataframe.
PROTECTED_DEMOGRAPHIC_COLUMNS = ["RaceDesc", "HispanicLatino", "CitizenDesc"]
IDENTIFIER_COLUMNS = ["Employee_Name", "EmpID", "Zip", "ManagerID", "State"]
UNRELIABLE_ID_COLUMNS = ["DeptID", "PerfScoreID", "PositionID", "MaritalStatusID",
                          "GenderID", "EmpStatusID", "MarriedID", "FromDiversityJobFairID"]

MIN_GROUP_SIZE = 5  # groups smaller than this are flagged as small-sample throughout


def _fix_two_digit_year(dob_series: pd.Series) -> pd.Series:
    """Parse MM/DD/YY birth dates and correct the 2-digit-year ambiguity:
    pandas' default pivot reads e.g. '01/02/51' as 2051, not 1951. Any
    parsed date in the future is shifted back exactly 100 years."""
    parsed = pd.to_datetime(dob_series, format="%m/%d/%y", errors="coerce")
    future = parsed > REFERENCE_DATE
    parsed.loc[future] = parsed.loc[future] - pd.DateOffset(years=100)
    return parsed


def load_hr_dataset(path: Path = HR_DATA_PATH) -> pd.DataFrame:
    """Load and clean HRDataset_v14.csv into an analytics-ready dataframe.
    The original CSV file on disk is never modified."""
    raw = pd.read_csv(path)

    string_cols = ["Department", "Position", "EmploymentStatus", "RecruitmentSource",
                   "PerformanceScore", "ManagerName", "TermReason", "MaritalDesc", "Sex"]
    for col in string_cols:
        raw[col] = raw[col].astype(str).str.strip()

    df = pd.DataFrame({
        "Department": raw["Department"],
        "Position": raw["Position"],
        "ManagerName": raw["ManagerName"],
        "RecruitmentSource": raw["RecruitmentSource"],
        "PerformanceScore": raw["PerformanceScore"],
        "EmploymentStatus": raw["EmploymentStatus"],
        "TermReason": raw["TermReason"],
        "Termd": raw["Termd"].astype(int),
        "Salary": raw["Salary"].astype(float),
        "EngagementSurvey": raw["EngagementSurvey"].astype(float),
        "EmpSatisfaction": raw["EmpSatisfaction"].astype(float),
        "SpecialProjectsCount": raw["SpecialProjectsCount"].astype(float),
        "DaysLateLast30": raw["DaysLateLast30"].astype(float),
        "Absences": raw["Absences"].astype(float),
    })

    date_of_hire = pd.to_datetime(raw["DateofHire"])
    date_of_termination = pd.to_datetime(raw["DateofTermination"])
    df["DateofHire"] = date_of_hire
    df["DateofTermination"] = date_of_termination
    tenure_end = date_of_termination.fillna(REFERENCE_DATE)
    df["TenureYears"] = ((tenure_end - date_of_hire).dt.days / 365.25).round(1)

    dob_corrected = _fix_two_digit_year(raw["DOB"])
    df["AgeYears"] = ((REFERENCE_DATE - dob_corrected).dt.days / 365.25).round(0)

    df["Status"] = np.where(df["Termd"] == 1, "Terminated", "Active")

    return df


def turnover_summary(df: pd.DataFrame, group_col: str, min_n: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    """Headcount, terminated count, and turnover rate per group, with a
    small_sample flag for groups below min_n - never present a rate from a
    tiny group as if it were as meaningful as one from a large group."""
    g = df.groupby(group_col)["Termd"].agg(headcount="count", terminated="sum").reset_index()
    g["turnover_rate"] = (g["terminated"] / g["headcount"]).round(3)
    g["small_sample"] = g["headcount"] < min_n
    return g.sort_values("turnover_rate", ascending=False)


def numeric_by_group(df: pd.DataFrame, group_col: str, value_col: str, min_n: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    g = df.groupby(group_col)[value_col].agg(mean="mean", count="count").reset_index()
    g["small_sample"] = g["count"] < min_n
    return g.sort_values("mean", ascending=False)


def generate_key_insights(df: pd.DataFrame, min_n: int = MIN_GROUP_SIZE) -> list[str]:
    """Data-driven, non-causal observations computed from the CURRENTLY
    FILTERED dataframe. Every number here is computed live - nothing is
    hardcoded. Returns an empty-safe list (never raises on small/empty data)."""
    insights = []
    if len(df) == 0:
        return ["No employees match the current filters."]

    overall_rate = df["Termd"].mean()
    insights.append(
        f"Observed workforce turnover rate in the current selection: {overall_rate:.1%} "
        f"(n={len(df)})."
    )

    dept = turnover_summary(df, "Department", min_n)
    dept_eligible = dept[~dept["small_sample"]]
    if len(dept_eligible) >= 2:
        top = dept_eligible.iloc[0]
        bottom = dept_eligible.iloc[-1]
        insights.append(
            f"Among departments with at least {min_n} employees, {top['Department']} has the "
            f"highest observed turnover rate ({top['turnover_rate']:.1%}, n={int(top['headcount'])}), "
            f"and {bottom['Department']} has the lowest ({bottom['turnover_rate']:.1%}, "
            f"n={int(bottom['headcount'])}) in this dataset."
        )

    rec = turnover_summary(df, "RecruitmentSource", min_n)
    rec_eligible = rec[~rec["small_sample"]]
    if len(rec_eligible) >= 2:
        top = rec_eligible.iloc[0]
        bottom = rec_eligible.iloc[-1]
        insights.append(
            f"Among recruitment sources with at least {min_n} hires, employees recruited via "
            f"{top['RecruitmentSource']} showed a higher observed turnover rate "
            f"({top['turnover_rate']:.1%}, n={int(top['headcount'])}) than those via "
            f"{bottom['RecruitmentSource']} ({bottom['turnover_rate']:.1%}, "
            f"n={int(bottom['headcount'])}) in this dataset."
        )

    if df["Termd"].nunique() == 2:
        eng_by_status = df.groupby("Status")["EngagementSurvey"].mean()
        if "Active" in eng_by_status.index and "Terminated" in eng_by_status.index:
            direction = "lower" if eng_by_status["Terminated"] < eng_by_status["Active"] else "higher"
            insights.append(
                f"Average engagement survey score is {direction} among terminated employees "
                f"({eng_by_status['Terminated']:.2f}) than currently active employees "
                f"({eng_by_status['Active']:.2f}) in this dataset."
            )

        sat_by_status = df.groupby("Status")["EmpSatisfaction"].mean()
        if "Active" in sat_by_status.index and "Terminated" in sat_by_status.index:
            direction = "lower" if sat_by_status["Terminated"] < sat_by_status["Active"] else "higher"
            insights.append(
                f"Average self-reported satisfaction is {direction} among terminated employees "
                f"({sat_by_status['Terminated']:.2f}/5) than active employees "
                f"({sat_by_status['Active']:.2f}/5) in this dataset."
            )

        abs_by_status = df.groupby("Status")["Absences"].mean()
        if "Active" in abs_by_status.index and "Terminated" in abs_by_status.index:
            direction = "higher" if abs_by_status["Terminated"] > abs_by_status["Active"] else "lower"
            insights.append(
                f"Average recorded absences are {direction} among terminated employees "
                f"({abs_by_status['Terminated']:.1f}) than active employees "
                f"({abs_by_status['Active']:.1f}) in this dataset."
            )

    mgr = turnover_summary(df, "ManagerName", min_n)
    mgr_eligible = mgr[~mgr["small_sample"]]
    if len(mgr_eligible) >= 2:
        top = mgr_eligible.iloc[0]
        insights.append(
            f"Among managers with at least {min_n} direct reports, {top['ManagerName']}'s team "
            f"has the highest observed turnover rate in this dataset ({top['turnover_rate']:.1%}, "
            f"n={int(top['headcount'])})."
        )

    return insights
