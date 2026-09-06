import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import (
    BUSINESS_TRAVEL_OPTIONS, DEPARTMENTS, EDUCATION_FIELDS,
    EDUCATION_LEVEL_LABELS, JOB_ROLE_TO_JOB_LEVEL, JOB_ROLES,
    MARITAL_STATUSES, MODEL_FEATURES, PERFORMANCE_RATING_LABELS,
    SATISFACTION_LABELS, WORK_LIFE_BALANCE_LABELS, build_single_input,
)
from src.hr_analytics import (
    MIN_GROUP_SIZE, generate_key_insights, load_hr_dataset,
    numeric_by_group, turnover_summary,
)

st.set_page_config(page_title="Employee Attrition Prediction System", layout="wide")

PURPLE_BLUE_CSS = """
<style>
.main {
    background: linear-gradient(135deg, #f3f0ff 0%, #eef4ff 55%, #eaf6ff 100%);
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffffcc, #eef1ffcc);
    border: 1px solid #d9d3fb;
    border-radius: 10px;
    padding: 12px 14px;
}
h1, h2, h3 { color: #4b3f9e; }
.risk-card {
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 8px;
    color: white;
    font-size: 1.05rem;
}
.risk-low { background: linear-gradient(135deg, #34c98f, #2fae7f); }
.risk-medium { background: linear-gradient(135deg, #f2b84b, #e0942a); }
.risk-high { background: linear-gradient(135deg, #ef5f6f, #d43d4f); }
div.stButton > button {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: bold;
    border: none;
}
.stSlider label, .stNumberInput label, .stSelectbox label, .stMultiSelect label {
    color: #1e1e2f !important;
}
.stSlider span { color: #1e1e2f !important; }
[data-testid="stMetricLabel"] { color: #1e1e2f !important; }
[data-testid="stMetricValue"] { color: #2c2650 !important; }
</style>
"""
st.markdown(PURPLE_BLUE_CSS, unsafe_allow_html=True)

DATA_PATH = Path("IBM Dataset.csv")
OLD_MODEL_PATH = Path("attrition_model.pkl")
V2_MODEL_PATH = Path("attrition_model_v2.pkl")
METRICS_PATH = Path("models/metrics.json")
METRICS_V3_PATH = Path("models/metrics_v3.json")
FEATURE_IMPORTANCE_PATH = Path("models/feature_importance.csv")
ROC_DATA_PATH = Path("models/roc_data.json")
SHAP_BACKGROUND_PATH = Path("models/shap_background.csv")


@st.cache_resource
def load_models():
    models = {}
    if OLD_MODEL_PATH.exists():
        models["Production model (attrition_model.pkl)"] = {
            "model": joblib.load(OLD_MODEL_PATH),
            "default_threshold": 0.5,
            "caption": "Original XGBoost model. No leak-free threshold optimization is "
                       "available for it (its original train/test split is not reproducible), "
                       "so it uses the conventional 0.5 default.",
        }
    if V2_MODEL_PATH.exists():
        models["Candidate model v2 (tuned, evaluated - see Model Performance tab)"] = {
            "model": joblib.load(V2_MODEL_PATH),
            "default_threshold": None,  # filled in from metrics.json once loaded
            "caption": "Rebuilt pipeline (Logistic Regression, tuned) with a properly "
                       "cross-validated decision threshold. See the Model Performance tab "
                       "for why this is offered alongside, not instead of, the production model.",
        }
    return models


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_hr_analytics_dataset():
    return load_hr_dataset()


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.load(open(METRICS_PATH))
    return None


@st.cache_data
def load_metrics_v3():
    if METRICS_V3_PATH.exists():
        return json.load(open(METRICS_V3_PATH))
    return None


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        s = pd.read_csv(FEATURE_IMPORTANCE_PATH, index_col=0)["importance"]
        return s
    return None


@st.cache_data
def load_roc_data():
    if ROC_DATA_PATH.exists():
        return json.load(open(ROC_DATA_PATH))
    return None


@st.cache_data
def load_shap_background():
    if SHAP_BACKGROUND_PATH.exists():
        return pd.read_csv(SHAP_BACKGROUND_PATH)[MODEL_FEATURES]
    return None


models = load_models()
df = load_dataset()
hr_df = load_hr_analytics_dataset()
metrics = load_metrics()
metrics_v3 = load_metrics_v3()
feature_importance = load_feature_importance()
roc_data = load_roc_data()
shap_background = load_shap_background()

if metrics and "Candidate model v2 (tuned, evaluated - see Model Performance tab)" in models:
    models["Candidate model v2 (tuned, evaluated - see Model Performance tab)"]["default_threshold"] = metrics["chosen_threshold"]

if not models:
    st.error("No trained model found. Run `python -m src.train_model` first, or make sure "
              "attrition_model.pkl is present.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Prediction", "Dashboard", "Corporate HR Analytics", "Model Performance"]
)

# ============================================================= PREDICTION ==
with tab1:
    st.title("Employee Attrition Prediction System")
    st.caption("Source: IBM HR Analytics dataset (`IBM Dataset.csv`) - Layer 1 of this "
               "application. Trains and powers the ML prediction below.")
    st.info(
        "**Decision-support tool, not a decision-maker.** This estimates an attrition "
        "probability from historical patterns in the IBM HR Attrition dataset. It should "
        "inform a conversation with an employee's manager, not be used as the sole basis "
        "for any employment decision."
    )

    with st.expander("How this prediction works"):
        st.markdown(
            "```\n"
            "Employee details you enter\n"
            "      |\n"
            "      v\n"
            "Preprocessing  (same encoding used at training time - src/preprocessing.py)\n"
            "      |\n"
            "      v\n"
            "ML model       (predicts a probability, not a yes/no answer)\n"
            "      |\n"
            "      v\n"
            "Attrition probability   e.g. 62%\n"
            "      |\n"
            "      v\n"
            "HR Risk Tolerance threshold   e.g. 35% (a cutoff HR sets, not part of the model)\n"
            "      |\n"
            "      v\n"
            "Risk label   Low / Medium / High\n"
            "```\n"
            "The model never outputs \"High Risk\" directly - it outputs a probability. "
            "Turning that probability into a risk label is a separate, adjustable step "
            "controlled by the HR Risk Tolerance slider below."
        )

    model_choice = st.selectbox("Prediction model", list(models.keys()), index=0)
    active = models[model_choice]
    st.caption(active["caption"])

    st.subheader("Employee details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Personal**")
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", MARITAL_STATUSES)
        distance = st.number_input("Distance From Home (km)", 1, 50, 5)
        num_comp = st.slider("Num Companies Worked", 0, 10, 1)

        education = st.selectbox(
            "Education Level", list(EDUCATION_LEVEL_LABELS.keys()),
            format_func=lambda k: f"{k} - {EDUCATION_LEVEL_LABELS[k]}", index=2,
        )
        with st.expander("What do Education Levels mean?"):
            st.write(pd.DataFrame(
                {"Level": EDUCATION_LEVEL_LABELS.keys(), "Meaning": EDUCATION_LEVEL_LABELS.values()}
            ).set_index("Level"))
        education_field = st.selectbox("Education Field", EDUCATION_FIELDS)

    with c2:
        st.markdown("**Job**")
        department = st.selectbox("Department", DEPARTMENTS)
        job_role = st.selectbox("Job Role", JOB_ROLES)

        job_level = JOB_ROLE_TO_JOB_LEVEL[job_role]
        st.metric("Job Level (auto-determined from Job Role)", job_level)
        with st.expander("How is Job Level determined?"):
            st.write(
                "Job Level is set automatically from the selected Job Role, using the most "
                "common Job Level observed for that role in the training data (1 = entry-level "
                "up to 5 = top-level management). It is not a free input because, in the "
                "training data, Job Role and Job Level are tightly linked - letting HR pick "
                "both independently could describe combinations the model never saw."
            )

        business_travel = st.selectbox("Business Travel", BUSINESS_TRAVEL_OPTIONS)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        job_involve = st.selectbox(
            "Job Involvement", list(SATISFACTION_LABELS.keys()),
            format_func=lambda k: f"{k} - {SATISFACTION_LABELS[k]}", index=2,
        )
        perf = st.selectbox(
            "Performance Rating", list(PERFORMANCE_RATING_LABELS.keys()),
            format_func=lambda k: f"{k} - {PERFORMANCE_RATING_LABELS[k]}", index=2,
        )

    with c3:
        st.markdown("**Compensation & Tenure**")
        monthly_income = st.number_input("Monthly Income", 1000, 200000, 5000, step=500)
        monthly_rate = st.number_input("Monthly Rate", 1000, 30000, 10000)
        daily_rate = st.number_input("Daily Rate", 100, 2000, 800)
        hourly_rate = st.number_input("Hourly Rate", 10, 200, 60)
        percent_hike = st.number_input("Percent Salary Hike", 10, 35, 15)
        stock = st.slider("Stock Option Level", 0, 3, 1)
        total_years = st.slider("Total Working Years", 0, 40, 10)
        years_company = st.slider("Years at Company", 0, 40, 3)
        years_role = st.slider("Years in Current Role", 0, 20, 2)
        years_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_manager = st.slider("Years with Current Manager", 0, 17, 2)
        training = st.slider("Training Times Last Year", 0, 10, 2)

    st.subheader("Satisfaction & work-life")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        env_sat = st.selectbox("Environment Satisfaction", list(SATISFACTION_LABELS.keys()),
                                format_func=lambda k: f"{k} - {SATISFACTION_LABELS[k]}", index=2)
    with c5:
        job_sat = st.selectbox("Job Satisfaction", list(SATISFACTION_LABELS.keys()),
                                format_func=lambda k: f"{k} - {SATISFACTION_LABELS[k]}", index=2)
    with c6:
        relation_sat = st.selectbox("Relationship Satisfaction", list(SATISFACTION_LABELS.keys()),
                                     format_func=lambda k: f"{k} - {SATISFACTION_LABELS[k]}", index=2)
    with c7:
        work_balance = st.selectbox("Work Life Balance", list(WORK_LIFE_BALANCE_LABELS.keys()),
                                     format_func=lambda k: f"{k} - {WORK_LIFE_BALANCE_LABELS[k]}", index=1)

    # --- Logical validation: warn, never silently change the user's input ---
    warnings = []
    if years_company > total_years:
        warnings.append("Years at Company exceeds Total Working Years.")
    if years_role > years_company:
        warnings.append("Years in Current Role exceeds Years at Company.")
    if years_manager > years_company:
        warnings.append("Years with Current Manager exceeds Years at Company.")
    if years_promo > years_company:
        warnings.append("Years Since Last Promotion exceeds Years at Company.")
    # Training Times Last Year counts training EVENTS in the past year, not years of
    # experience, so it is not compared against tenure - only flagged if unusually high.
    if training >= 9:
        warnings.append("Training Times Last Year is unusually high (9+) - please confirm this is correct.")

    for w in warnings:
        st.warning(w)

    st.subheader("HR risk tolerance")
    st.caption(
        "This does **not** change the model. It only changes the probability cutoff used to "
        "label an employee High/Medium/Low risk - i.e. how cautious HR wants to be. A lower "
        "threshold flags more employees as high-risk (catches more true leavers, but more "
        "false alarms); a higher threshold flags fewer (fewer false alarms, but risks missing "
        "some leavers)."
    )
    if "threshold_pct" not in st.session_state:
        st.session_state["threshold_pct"] = int(round(active["default_threshold"] * 100))

    operating_points = metrics.get("operating_points") if metrics else None
    is_v2 = model_choice.startswith("Candidate model v2")
    if operating_points and is_v2:
        st.caption("Presets below come from cross-validated, out-of-fold analysis of this "
                   "model (not the test set) - see the Model Performance tab for the full sweep.")
        p1, p2, p3 = st.columns(3)
        hr, bal, hp = operating_points["high_recall"], operating_points["balanced"], operating_points["high_precision"]
        if p1.button(f"High Recall ({hr['threshold']:.0%}) - catches more leavers, more false alarms"):
            st.session_state["threshold_pct"] = int(round(hr["threshold"] * 100))
        if p2.button(f"Balanced ({bal['threshold']:.0%}) - best F1 trade-off"):
            st.session_state["threshold_pct"] = int(round(bal["threshold"] * 100))
        if p3.button(f"High Precision ({hp['threshold']:.0%}) - fewer false alarms, may miss some leavers"):
            st.session_state["threshold_pct"] = int(round(hp["threshold"] * 100))

    threshold_pct = st.slider("Decision threshold (%)", 10, 90, key="threshold_pct")
    threshold = threshold_pct / 100

    if threshold_pct <= 25:
        tolerance_label = "Low tolerance - flags risk early, more false alarms"
    elif threshold_pct <= 55:
        tolerance_label = "Medium tolerance - balanced"
    else:
        tolerance_label = "High tolerance - flags only clear risk, fewer false alarms"
    st.caption(f"**{tolerance_label}** (threshold = {threshold_pct}%)")

    if st.button("Predict", type="primary"):
        raw = {
            "Age": age, "DailyRate": daily_rate, "DistanceFromHome": distance,
            "Education": education, "EnvironmentSatisfaction": env_sat, "Gender": gender,
            "HourlyRate": hourly_rate, "JobInvolvement": job_involve, "JobLevel": job_level,
            "JobSatisfaction": job_sat, "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_comp, "OverTime": overtime, "PercentSalaryHike": percent_hike,
            "PerformanceRating": perf, "RelationshipSatisfaction": relation_sat, "StockOptionLevel": stock,
            "TotalWorkingYears": total_years, "TrainingTimesLastYear": training,
            "WorkLifeBalance": work_balance, "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role, "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_manager, "Department": department,
            "EducationField": education_field, "BusinessTravel": business_travel,
            "JobRole": job_role, "MaritalStatus": marital,
        }
        X_row = build_single_input(raw)
        proba = float(active["model"].predict_proba(X_row)[0][1])
        st.session_state["last_prediction"] = {"proba": proba, "threshold": threshold, "X_row": X_row}

    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]
        proba, threshold_used, X_row = pred["proba"], pred["threshold"], pred["X_row"]

        if proba >= threshold_used:
            risk_level, css_class = "High", "risk-high"
        elif proba >= max(threshold_used - 0.20, 0):
            risk_level, css_class = "Medium", "risk-medium"
        else:
            risk_level, css_class = "Low", "risk-low"

        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.markdown(
                f'<div class="risk-card {css_class}">'
                f'<b>Attrition Probability:</b> {proba:.0%}<br>'
                f'<b>Risk Level:</b> {risk_level}<br>'
                f'<b>Current HR Threshold:</b> {threshold_used:.0%}'
                f'</div>', unsafe_allow_html=True,
            )

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4b3f9e"},
                    "steps": [
                        {"range": [0, max(threshold_used - 0.20, 0) * 100], "color": "#d8f3e6"},
                        {"range": [max(threshold_used - 0.20, 0) * 100, threshold_used * 100], "color": "#fde8c2"},
                        {"range": [threshold_used * 100, 100], "color": "#fbdadd"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.85,
                        "value": threshold_used * 100,
                    },
                },
            ))
            gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(gauge, width="stretch")

        with rc2:
            st.markdown("**Factors contributing to risk**")
            st.caption(
                "These features contributed strongly to this model's prediction. This is "
                "association learned from historical data, not proof that a factor *caused* "
                "this employee's risk level."
            )
            show_shap = st.checkbox("Compute a per-employee explanation (SHAP, takes a few seconds)")
            explained = False
            if show_shap and shap_background is not None:
                try:
                    import shap
                    with st.spinner("Computing explanation..."):
                        def f(x):
                            return active["model"].predict_proba(pd.DataFrame(np.array(x), columns=MODEL_FEATURES))[:, 1]
                        explainer = shap.Explainer(f, shap_background)
                        sv = explainer(X_row)
                    contrib = pd.Series(sv.values[0], index=MODEL_FEATURES)
                    increasing = contrib[contrib > 0].sort_values(ascending=False).head(5)
                    decreasing = contrib[contrib < 0].sort_values().head(5)

                    fc1, fc2 = st.columns(2)
                    with fc1:
                        st.markdown("*Increasing risk*")
                        if len(increasing):
                            st.dataframe(increasing.rename("impact").to_frame(), use_container_width=True)
                        else:
                            st.caption("No features pushed risk up for this employee.")
                    with fc2:
                        st.markdown("*Decreasing risk*")
                        if len(decreasing):
                            st.dataframe(decreasing.rename("impact").to_frame(), use_container_width=True)
                        else:
                            st.caption("No features pushed risk down for this employee.")

                    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
                    fig = px.bar(
                        top[::-1], orientation="h",
                        labels={"value": "Impact on predicted probability", "index": ""},
                        color=top[::-1].values, color_continuous_scale=["#34c98f", "#eeeeee", "#d43d4f"],
                    )
                    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=300)
                    st.plotly_chart(fig, width="stretch")
                    explained = True
                except Exception as e:
                    st.info(f"SHAP explanation unavailable ({e}); showing overall model feature importance instead.")

            if not explained and feature_importance is not None:
                top = feature_importance.head(8).sort_values()
                fig = px.bar(top, orientation="h",
                             labels={"value": "Overall model importance", "index": ""})
                fig.update_layout(showlegend=False, height=340)
                st.plotly_chart(fig, width="stretch")

# ============================================================== DASHBOARD ==
with tab2:
    st.header("HR Analytics Dashboard")
    st.caption(f"Source: {DATA_PATH.name} ({len(df)} employees) - Layer 1, the same dataset "
               "the ML model is trained on.")

    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    dept_f = fc1.multiselect("Department", sorted(df["Department"].unique()))
    role_f = fc2.multiselect("Job Role", sorted(df["JobRole"].unique()))
    gender_f = fc3.multiselect("Gender", sorted(df["Gender"].unique()))
    overtime_f = fc4.multiselect("OverTime", sorted(df["OverTime"].unique()))
    marital_f = fc5.multiselect("Marital Status", sorted(df["MaritalStatus"].unique()))

    fdf = df.copy()
    if dept_f:
        fdf = fdf[fdf["Department"].isin(dept_f)]
    if role_f:
        fdf = fdf[fdf["JobRole"].isin(role_f)]
    if gender_f:
        fdf = fdf[fdf["Gender"].isin(gender_f)]
    if overtime_f:
        fdf = fdf[fdf["OverTime"].isin(overtime_f)]
    if marital_f:
        fdf = fdf[fdf["MaritalStatus"].isin(marital_f)]

    if fdf.empty:
        st.warning("No employees match the selected filters.")
    else:
        total = len(fdf)
        left = int((fdf["Attrition"] == "Yes").sum())
        rate = left / total

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Employees", f"{total:,}")
        k2.metric("Employees Left", f"{left:,}")
        k3.metric("Attrition Rate", f"{rate:.1%}")

        purple_blue = ["#7c6ff0", "#5aa9e6"]

        g1, g2 = st.columns(2)
        with g1:
            dist = fdf["Attrition"].value_counts().reset_index()
            dist.columns = ["Attrition", "Count"]
            fig = px.pie(dist, names="Attrition", values="Count", hole=0.5,
                         title="Attrition Distribution", color_discrete_sequence=purple_blue)
            st.plotly_chart(fig, width="stretch")
        with g2:
            by_dept = fdf.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
            fig = px.bar(by_dept, x="Department", y="Count", color="Attrition", barmode="group",
                         title="Attrition by Department", color_discrete_sequence=purple_blue)
            st.plotly_chart(fig, width="stretch")

        g3, g4 = st.columns(2)
        with g3:
            by_role = fdf.groupby(["JobRole", "Attrition"]).size().reset_index(name="Count")
            fig = px.bar(by_role, x="JobRole", y="Count", color="Attrition", barmode="group",
                         title="Attrition by Job Role", color_discrete_sequence=purple_blue)
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, width="stretch")
        with g4:
            by_ot = fdf.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
            fig = px.bar(by_ot, x="OverTime", y="Count", color="Attrition", barmode="group",
                         title="Attrition by OverTime", color_discrete_sequence=purple_blue)
            st.plotly_chart(fig, width="stretch")

        g5, g6 = st.columns(2)
        with g5:
            by_level = fdf.groupby(["JobLevel", "Attrition"]).size().reset_index(name="Count")
            fig = px.bar(by_level, x="JobLevel", y="Count", color="Attrition", barmode="group",
                         title="Attrition by Job Level", color_discrete_sequence=purple_blue)
            st.plotly_chart(fig, width="stretch")
        with g6:
            income_bins = [0, 3000, 6000, 10000, 15000, 20000]
            income_labels = ["<3k", "3k-6k", "6k-10k", "10k-15k", "15k+"]
            tmp = fdf.copy()
            tmp["IncomeRange"] = pd.cut(tmp["MonthlyIncome"], bins=income_bins, labels=income_labels)
            by_income = tmp.groupby(["IncomeRange", "Attrition"], observed=True).size().reset_index(name="Count")
            fig = px.bar(by_income, x="IncomeRange", y="Count", color="Attrition", barmode="group",
                         title="Attrition by Monthly Income Range", color_discrete_sequence=purple_blue)
            st.plotly_chart(fig, width="stretch")

        g7, g8 = st.columns(2)
        with g7:
            year_bins = [-1, 2, 5, 10, 20, 100]
            year_labels = ["0-2", "3-5", "6-10", "11-20", "20+"]
            tmp = fdf.copy()
            tmp["YearsRange"] = pd.cut(tmp["YearsAtCompany"], bins=year_bins, labels=year_labels)
            by_years = tmp.groupby(["YearsRange", "Attrition"], observed=True).size().reset_index(name="Count")
            fig = px.bar(by_years, x="YearsRange", y="Count", color="Attrition", barmode="group",
                         title="Attrition by Years at Company", color_discrete_sequence=purple_blue)
            st.plotly_chart(fig, width="stretch")
        with g8:
            by_jobsat = fdf.groupby(["JobSatisfaction", "Attrition"]).size().reset_index(name="Count")
            fig = px.bar(by_jobsat, x="JobSatisfaction", y="Count", color="Attrition", barmode="group",
                         title="Attrition by Job Satisfaction", color_discrete_sequence=purple_blue)
            st.plotly_chart(fig, width="stretch")

# ================================================ CORPORATE HR ANALYTICS ==
with tab3:
    st.header("Corporate HR Analytics")
    st.info(
        "**This dashboard uses a separate corporate HR dataset for organizational "
        "analytics. It is not used to train or modify the employee attrition "
        "prediction model.** Source: `HRDataset_v14.csv` (311 employees) - Layer 2, "
        "a different company/population than the IBM dataset used for prediction "
        "(Layer 1). The two are never merged: different employees, different "
        "schemas, different base rates - see Model Performance tab for why."
    )

    hr_filter_cols = st.columns(6)
    dept_f = hr_filter_cols[0].multiselect("Department", sorted(hr_df["Department"].unique()))
    pos_f = hr_filter_cols[1].multiselect("Position", sorted(hr_df["Position"].unique()))
    mgr_f = hr_filter_cols[2].multiselect("Manager", sorted(hr_df["ManagerName"].unique()))
    rec_f = hr_filter_cols[3].multiselect("Recruitment Source", sorted(hr_df["RecruitmentSource"].unique()))
    status_f = hr_filter_cols[4].multiselect("Employment Status", sorted(hr_df["EmploymentStatus"].unique()))
    perf_f = hr_filter_cols[5].multiselect("Performance Score", sorted(hr_df["PerformanceScore"].unique()))

    hdf = hr_df.copy()
    if dept_f:
        hdf = hdf[hdf["Department"].isin(dept_f)]
    if pos_f:
        hdf = hdf[hdf["Position"].isin(pos_f)]
    if mgr_f:
        hdf = hdf[hdf["ManagerName"].isin(mgr_f)]
    if rec_f:
        hdf = hdf[hdf["RecruitmentSource"].isin(rec_f)]
    if status_f:
        hdf = hdf[hdf["EmploymentStatus"].isin(status_f)]
    if perf_f:
        hdf = hdf[hdf["PerformanceScore"].isin(perf_f)]

    if hdf.empty:
        st.warning("No employees match the selected filters.")
    else:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Employees", f"{len(hdf):,}")
        k2.metric("Employees Terminated", f"{int(hdf['Termd'].sum()):,}")
        k3.metric("Workforce Turnover Rate", f"{hdf['Termd'].mean():.1%}")
        k4.metric("Average Salary", f"${hdf['Salary'].mean():,.0f}")
        k5.metric("Average Engagement", f"{hdf['EngagementSurvey'].mean():.2f}/5")
        k6.metric("Average Absences", f"{hdf['Absences'].mean():.1f}")
        st.caption(f"Source: HRDataset_v14.csv - KPIs computed live from {len(hdf)} filtered employees.")

        sub_overview, sub_dept, sub_mgr, sub_rec, sub_comp, sub_eng, sub_att, sub_perf, sub_insights = st.tabs([
            "Workforce Overview", "Department", "Manager", "Recruitment",
            "Compensation", "Engagement & Satisfaction", "Attendance & Workload",
            "Performance", "Key HR Insights",
        ])
        purple_blue2 = ["#7c6ff0", "#5aa9e6"]

        # ---- Workforce Overview ----
        with sub_overview:
            st.caption("Source: HRDataset_v14.csv")
            oc1, oc2 = st.columns(2)
            with oc1:
                by_dept = hdf["Department"].value_counts().reset_index()
                by_dept.columns = ["Department", "Count"]
                fig = px.bar(by_dept, x="Department", y="Count", title="Workforce by Department",
                             color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with oc2:
                by_status = hdf["EmploymentStatus"].value_counts().reset_index()
                by_status.columns = ["EmploymentStatus", "Count"]
                fig = px.pie(by_status, names="EmploymentStatus", values="Count", hole=0.5,
                             title="Workforce by Employment Status", color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")

            oc3, oc4 = st.columns(2)
            with oc3:
                by_pos = hdf["Position"].value_counts().head(12).reset_index()
                by_pos.columns = ["Position", "Count"]
                fig = px.bar(by_pos, x="Count", y="Position", orientation="h",
                             title="Workforce by Position (top 12)", color_discrete_sequence=purple_blue2)
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, width="stretch")
            with oc4:
                fig = px.histogram(hdf, x="TenureYears", nbins=20, title="Tenure Distribution (years)",
                                   color_discrete_sequence=purple_blue2)
                fig.update_layout(yaxis_title="Employees", xaxis_title="Tenure (years)")
                st.plotly_chart(fig, width="stretch")

            oc5, oc6 = st.columns(2)
            with oc5:
                fig = px.histogram(hdf, x="Salary", nbins=20, title="Salary Distribution",
                                   color_discrete_sequence=purple_blue2)
                fig.update_layout(yaxis_title="Employees")
                st.plotly_chart(fig, width="stretch")
            with oc6:
                fig = px.histogram(hdf, x="EngagementSurvey", nbins=15, title="Engagement Distribution",
                                   color_discrete_sequence=purple_blue2)
                fig.update_layout(yaxis_title="Employees")
                st.plotly_chart(fig, width="stretch")

        # ---- Department Analytics ----
        with sub_dept:
            st.caption("Source: HRDataset_v14.csv. Departments with fewer than "
                       f"{MIN_GROUP_SIZE} employees are flagged - their turnover rate is not "
                       "statistically meaningful and should not be over-interpreted.")
            dept_sum = turnover_summary(hdf, "Department")
            dc1, dc2 = st.columns(2)
            with dc1:
                fig = px.bar(dept_sum, x="Department", y="headcount", title="Employee Count by Department",
                             text="headcount", color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with dc2:
                fig = px.bar(dept_sum, x="Department", y="terminated", title="Terminated Count by Department",
                             text="terminated", color_discrete_sequence=["#d43d4f"])
                st.plotly_chart(fig, width="stretch")

            fig = px.bar(
                dept_sum, x="Department", y="turnover_rate",
                title="Observed Turnover Rate by Department (bar label = headcount)",
                text=dept_sum["headcount"].apply(lambda n: f"n={n}"),
                color="small_sample", color_discrete_map={True: "#f2b84b", False: "#7c6ff0"},
            )
            fig.update_layout(yaxis_tickformat=".0%", legend_title="Small sample (n<5)")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(dept_sum.style.format({"turnover_rate": "{:.1%}"}), use_container_width=True)

        # ---- Manager Analytics ----
        with sub_mgr:
            st.caption(
                "Source: HRDataset_v14.csv. ManagerName is used for organizational analytics "
                "only - never as a predictive feature. Managers with small teams "
                f"(n<{MIN_GROUP_SIZE}) are flagged: their turnover rate can swing wildly from a "
                "single departure and should be read with caution, not as a performance verdict "
                "on the manager."
            )
            mgr_sum = turnover_summary(hdf, "ManagerName").sort_values("headcount", ascending=False)
            mc1, mc2 = st.columns(2)
            with mc1:
                fig = px.bar(mgr_sum, x="ManagerName", y="headcount", title="Employee Count by Manager",
                             color_discrete_sequence=purple_blue2)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width="stretch")
            with mc2:
                fig = px.bar(mgr_sum, x="ManagerName", y="terminated", title="Terminated Employees by Manager",
                             color_discrete_sequence=["#d43d4f"])
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width="stretch")

            mgr_sum_by_rate = mgr_sum.sort_values("turnover_rate", ascending=False)
            fig = px.bar(
                mgr_sum_by_rate, x="ManagerName", y="turnover_rate",
                title="Manager-Level Turnover Rate (bar label = headcount)",
                text=mgr_sum_by_rate["headcount"].apply(lambda n: f"n={n}"),
                color="small_sample", color_discrete_map={True: "#f2b84b", False: "#7c6ff0"},
            )
            fig.update_layout(yaxis_tickformat=".0%", legend_title="Small sample (n<5)")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(mgr_sum_by_rate.style.format({"turnover_rate": "{:.1%}"}), use_container_width=True)

        # ---- Recruitment Analytics ----
        with sub_rec:
            st.caption(
                "Source: HRDataset_v14.csv. Observed associations only - a recruitment source "
                "with higher turnover in this dataset is not necessarily a worse channel; other "
                "factors (role mix, timing) are not controlled for here."
            )
            rec_sum = turnover_summary(hdf, "RecruitmentSource")
            rc1, rc2 = st.columns(2)
            with rc1:
                fig = px.bar(rec_sum, x="RecruitmentSource", y="headcount",
                             title="Employees by Recruitment Source", color_discrete_sequence=purple_blue2)
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, width="stretch")
            with rc2:
                fig = px.bar(rec_sum, x="RecruitmentSource", y="terminated",
                             title="Turnover by Recruitment Source", color_discrete_sequence=["#d43d4f"])
                fig.update_xaxes(tickangle=30)
                st.plotly_chart(fig, width="stretch")

            fig = px.bar(
                rec_sum, x="RecruitmentSource", y="turnover_rate",
                title="Observed Turnover Rate by Recruitment Source (bar label = headcount)",
                text=rec_sum["headcount"].apply(lambda n: f"n={n}"),
                color="small_sample", color_discrete_map={True: "#f2b84b", False: "#7c6ff0"},
            )
            fig.update_layout(yaxis_tickformat=".0%", legend_title="Small sample (n<5)")
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, width="stretch")

        # ---- Compensation ----
        with sub_comp:
            st.caption("Source: HRDataset_v14.csv. Salary patterns associated with workforce "
                       "turnover - not evidence that salary causes attrition.")
            cc1, cc2 = st.columns(2)
            with cc1:
                fig = px.histogram(hdf, x="Salary", nbins=20, title="Salary Distribution",
                                   color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with cc2:
                sal_by_dept = numeric_by_group(hdf, "Department", "Salary")
                fig = px.bar(sal_by_dept, x="Department", y="mean", title="Average Salary by Department",
                             text=sal_by_dept["count"].apply(lambda n: f"n={n}"),
                             color_discrete_sequence=purple_blue2)
                fig.update_layout(yaxis_title="Average Salary ($)")
                st.plotly_chart(fig, width="stretch")

            sal_by_pos = numeric_by_group(hdf, "Position", "Salary")
            fig = px.bar(sal_by_pos, x="Position", y="mean", title="Average Salary by Position",
                         text=sal_by_pos["count"].apply(lambda n: f"n={n}"),
                         color_discrete_sequence=purple_blue2)
            fig.update_layout(yaxis_title="Average Salary ($)")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width="stretch")

            fig = px.box(hdf, x="Status", y="Salary", title="Salary Distribution by Turnover Status",
                         color="Status", color_discrete_sequence=purple_blue2)
            st.plotly_chart(fig, width="stretch")

        # ---- Engagement & Satisfaction ----
        with sub_eng:
            st.caption("Source: HRDataset_v14.csv. Associations observed in this dataset, not "
                       "causal claims.")
            ec1, ec2 = st.columns(2)
            with ec1:
                fig = px.histogram(hdf, x="EngagementSurvey", nbins=15, title="Engagement Distribution",
                                   color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with ec2:
                fig = px.histogram(hdf, x="EmpSatisfaction", nbins=5, title="Satisfaction Distribution",
                                   color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")

            ec3, ec4 = st.columns(2)
            with ec3:
                eng_by_dept = numeric_by_group(hdf, "Department", "EngagementSurvey")
                fig = px.bar(eng_by_dept, x="Department", y="mean", title="Average Engagement by Department",
                             text=eng_by_dept["count"].apply(lambda n: f"n={n}"), color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with ec4:
                sat_by_dept = numeric_by_group(hdf, "Department", "EmpSatisfaction")
                fig = px.bar(sat_by_dept, x="Department", y="mean", title="Average Satisfaction by Department",
                             text=sat_by_dept["count"].apply(lambda n: f"n={n}"), color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")

            ec5, ec6 = st.columns(2)
            with ec5:
                fig = px.box(hdf, x="Status", y="EngagementSurvey", title="Engagement vs. Turnover Status",
                             color="Status", color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with ec6:
                fig = px.box(hdf, x="Status", y="EmpSatisfaction", title="Satisfaction vs. Turnover Status",
                             color="Status", color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")

        # ---- Attendance & Workload ----
        with sub_att:
            st.caption("Source: HRDataset_v14.csv. Associations only - the dataset does not "
                       "document whether these figures were measured well before, or right "
                       "around, an employee's departure.")
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                fig = px.histogram(hdf, x="Absences", nbins=20, title="Absence Distribution",
                                   color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with ac2:
                fig = px.histogram(hdf, x="DaysLateLast30", nbins=7, title="Lateness Distribution",
                                   color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with ac3:
                fig = px.histogram(hdf, x="SpecialProjectsCount", nbins=9, title="Special Projects Distribution",
                                   color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")

            ac4, ac5, ac6 = st.columns(3)
            with ac4:
                fig = px.box(hdf, x="Status", y="Absences", title="Absences vs. Turnover",
                             color="Status", color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with ac5:
                fig = px.box(hdf, x="Status", y="DaysLateLast30", title="Lateness vs. Turnover",
                             color="Status", color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with ac6:
                fig = px.box(hdf, x="Status", y="SpecialProjectsCount", title="Special Projects vs. Turnover",
                             color="Status", color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")

            insight_lines = [i for i in generate_key_insights(hdf) if "absences" in i.lower() or "engagement" in i.lower()]
            if insight_lines:
                st.markdown("**HR insight (from the currently filtered data):**")
                for line in insight_lines:
                    st.caption(f"- {line}")

        # ---- Performance ----
        with sub_perf:
            st.warning(
                "**Descriptive/observational only.** PerformanceScore is shown here for HR "
                "analytics, but is deliberately NOT used as a feature in the attrition ML "
                "model - the dataset does not establish whether performance information was "
                "recorded before or around termination, so using it predictively would risk "
                "target leakage."
            )
            st.caption("Source: HRDataset_v14.csv")
            perf_order = ["PIP", "Needs Improvement", "Fully Meets", "Exceeds"]
            present_order = [p for p in perf_order if p in hdf["PerformanceScore"].unique()]

            pc1, pc2 = st.columns(2)
            with pc1:
                perf_counts = hdf["PerformanceScore"].value_counts().reindex(present_order).reset_index()
                perf_counts.columns = ["PerformanceScore", "Count"]
                fig = px.bar(perf_counts, x="PerformanceScore", y="Count", title="Performance Distribution",
                             color_discrete_sequence=purple_blue2)
                st.plotly_chart(fig, width="stretch")
            with pc2:
                perf_dept = hdf.groupby(["Department", "PerformanceScore"]).size().reset_index(name="Count")
                fig = px.bar(perf_dept, x="Department", y="Count", color="PerformanceScore",
                             title="Performance by Department", barmode="stack",
                             category_orders={"PerformanceScore": present_order})
                st.plotly_chart(fig, width="stretch")

            perf_sum = turnover_summary(hdf, "PerformanceScore")
            fig = px.bar(
                perf_sum, x="PerformanceScore", y="turnover_rate",
                title="Observed Turnover Rate by Performance Score (bar label = headcount)",
                text=perf_sum["headcount"].apply(lambda n: f"n={n}"),
                color="small_sample", color_discrete_map={True: "#f2b84b", False: "#7c6ff0"},
                category_orders={"PerformanceScore": present_order},
            )
            fig.update_layout(yaxis_tickformat=".0%", legend_title="Small sample (n<5)")
            st.plotly_chart(fig, width="stretch")

        # ---- Key HR Insights ----
        with sub_insights:
            st.subheader("Key HR Insights")
            st.caption(
                "Data-driven observations computed live from the currently filtered data - "
                "not hardcoded, not causal claims, and not recommendations to act on any "
                "individual employee."
            )
            for line in generate_key_insights(hdf):
                st.markdown(f"- {line}")
            st.caption(
                f"Sample size note: groups with fewer than {MIN_GROUP_SIZE} employees are "
                "excluded from these insights, and this dataset has only 311 employees "
                "overall - percentages from small groups can swing dramatically from a single "
                "employee and are not statistically tested here."
            )

# ======================================================= MODEL PERFORMANCE ==
with tab4:
    st.header("Model Performance")
    st.caption(
        "The Corporate HR Analytics dataset (`HRDataset_v14.csv`) is intentionally not "
        "merged with the ML training dataset (`IBM Dataset.csv`) because the datasets "
        "represent different employee populations and feature schemas. Everything below "
        "reflects the IBM dataset only."
    )
    if metrics is None:
        st.info("No metrics found. Run `python -m src.train_model` to generate model_comparison "
                 "results, feature importance and ROC data.")
    else:
        st.caption(
            f"Trained on {metrics['n_rows']} rows ({metrics['n_train']} train / {metrics['n_test']} test, "
            f"stratified split, test attrition rate {metrics['test_attrition_rate']:.1%})."
        )

        st.subheader("Model comparison (held-out test set, threshold = 0.5)")
        comp_df = pd.DataFrame(metrics["model_comparison_at_0.5"])
        st.dataframe(comp_df.set_index("model").style.format("{:.3f}"), use_container_width=True)

        st.subheader("Out-of-fold model selection (used to pick the best model + threshold)")
        st.caption(
            "Selection is based on cross-validated, out-of-fold predictions on the training "
            "set only - the test set above was never used to choose a model or a threshold."
        )
        oof_df = pd.DataFrame(metrics["oof_model_selection"])
        st.dataframe(oof_df.set_index("model").style.format(
            {"best_threshold": "{:.2f}", "oof_precision": "{:.3f}", "oof_recall": "{:.3f}",
             "oof_f1": "{:.3f}", "oof_roc_auc": "{:.3f}", "rank_score": "{:.3f}"}
        ), use_container_width=True)

        st.success(
            f"Best model selected: **{metrics['best_model']}** "
            f"(SMOTE {'used' if metrics['use_smote_for'][metrics['best_model']] else 'not used'} - "
            f"chosen because it gave the best cross-validated F1/recall trade-off), "
            f"default decision threshold = {metrics['chosen_threshold']}."
        )

        if metrics.get("operating_points"):
            st.subheader("Candidate operating points (out-of-fold, training data)")
            st.caption("These feed the preset buttons next to the HR Risk Tolerance slider on "
                       "the Prediction tab (candidate v2 model only).")
            op_df = pd.DataFrame(metrics["operating_points"]).T
            st.dataframe(op_df.style.format("{:.3f}"), use_container_width=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Confusion matrix (best model, chosen threshold, test set)")
            cm = metrics["best_model_test_at_chosen_threshold"]["confusion_matrix"]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples",
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=["No", "Yes"], y=["No", "Yes"])
            st.plotly_chart(fig, width="stretch")
        with cc2:
            st.subheader("ROC curve (best model, test set)")
            if roc_data:
                rd = list(roc_data.values())[0]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rd["fpr"], y=rd["tpr"], mode="lines",
                                          name=f"AUC = {rd['auc']:.3f}", line=dict(color="#7c6ff0")))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                          line=dict(color="gray", dash="dash")))
                fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig, width="stretch")

        st.subheader("Threshold sweep (out-of-fold, training data)")
        sweep_df = pd.DataFrame(metrics["threshold_sweep_oof_train"])
        fig = px.line(sweep_df, x="threshold", y=["precision", "recall", "f1"], markers=True,
                       color_discrete_sequence=["#5aa9e6", "#d43d4f", "#7c6ff0"])
        fig.update_layout(yaxis_title="Score", legend_title="Metric")
        st.plotly_chart(fig, width="stretch")

        if feature_importance is not None:
            st.subheader("Feature importance (best model)")
            top15 = feature_importance.head(15).sort_values()
            fig = px.bar(top15, orientation="h", color_discrete_sequence=["#7c6ff0"])
            fig.update_layout(showlegend=False, xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

        with st.expander("Existing attrition_model.pkl vs candidate v2 - full comparison notes"):
            leak = metrics.get("old_vs_new_same_test_set_caveat_leakage")
            if leak:
                st.warning(leak["caveat"])
                st.json({k: v for k, v in leak.items() if k != "caveat"})
            hist = metrics.get("old_model_historical_self_reported")
            if hist:
                st.markdown("**attrition_model.pkl's own historical self-reported test metrics** "
                             "(from the original notebook run, genuinely held out at the time, "
                             "but not reproducible since that split was unseeded):")
                st.json(hist)
            st.markdown(
                "**Bottom line:** on a like-for-like read of both models' real held-out "
                "performance, attrition_model.pkl's own reported numbers (precision 0.80 / "
                "recall 0.40 / F1 0.53) are not clearly beaten by the new tuned pipeline "
                f"({metrics['best_model']}: precision "
                f"{metrics['best_model_test_at_chosen_threshold']['precision']:.2f} / recall "
                f"{metrics['best_model_test_at_chosen_threshold']['recall']:.2f} / F1 "
                f"{metrics['best_model_test_at_chosen_threshold']['f1']:.2f}). "
                "attrition_model.pkl has NOT been replaced. The candidate model is offered "
                "in the Prediction tab for side-by-side comparison, with full methodology "
                "(SMOTE decision, tuning, threshold selection) documented above."
            )

        if metrics_v3:
            st.subheader("v3 experiment: feature engineering, more models, calibration, feature selection")
            st.caption(
                "A further round of experiments (src/train_model_v3.py) tried engineered "
                "features, Random Forest/XGBoost/CatBoost with per-model imbalance handling, "
                "probability calibration, and feature-selection ablation. Full results below - "
                "this is reported whether or not it beat v2, per the project's no-fabrication rule."
            )
            if metrics_v3["v3_meaningfully_better_than_v2"]:
                st.success("v3 WAS found to be meaningfully better than v2 on the held-out test "
                           "set - see attrition_model_v3.pkl and models/model_v3_metadata.json.")
            else:
                st.warning(
                    "**v3 was NOT meaningfully better than v2** on the held-out test set "
                    "(F1 and recall changes were within noise). `attrition_model_v3.pkl` was "
                    "therefore not created, and `attrition_model_v2.pkl` remains the recommended "
                    "candidate. This is reported as a genuine negative result, not hidden."
                )

            with st.expander("v3 experiment details"):
                st.markdown("**Feature engineering ablation** (mean CV F1 across Logistic "
                             "Regression + XGBoost, training data only):")
                fe_df = pd.DataFrame(metrics_v3["feature_engineering_ablation"])
                st.dataframe(fe_df.set_index(["feature_set", "model"]).style.format("{:.3f}"),
                             use_container_width=True)
                st.caption(f"-> engineered features {'were' if metrics_v3['use_engineered_features'] else 'were NOT'} kept.")

                st.markdown("**Imbalance handling comparison** (5-fold CV, default hyperparams, "
                             "per model x balancing method):")
                imb_df = pd.DataFrame(metrics_v3["imbalance_comparison"])
                st.dataframe(imb_df.set_index(["model", "balancing"]).style.format("{:.3f}"),
                             use_container_width=True)

                st.markdown("**Stability across folds** (mean +/- std - a model that wins only "
                             "because of one favorable split will show a large std):")
                stab_df = pd.DataFrame(metrics_v3["stability_cv_mean_std"])[
                    ["model", "f1_mean", "f1_std", "recall_mean", "recall_std", "roc_auc_mean", "roc_auc_std"]
                ]
                st.dataframe(stab_df.set_index("model").style.format("{:.3f}"), use_container_width=True)
                lr_std = stab_df.loc[stab_df.model == "Logistic Regression", "f1_std"].iloc[0] if "Logistic Regression" in stab_df.model.values else None
                st.caption(
                    "Logistic Regression won on mean CV F1 but with the highest fold-to-fold "
                    "std of the four models tested - its win is real (confirmed independently "
                    "on a second run) but should be read with that variability in mind, not as "
                    "an unambiguous, rock-solid win."
                )

                st.markdown("**Calibration check** (nested cross-validation, training data only):")
                st.json(metrics_v3["calibration"])

                st.markdown("**Feature selection ablation:**")
                fsel = metrics_v3["feature_selection"]
                st.write(f"All features: {fsel['n_all_features']}, selected subset: "
                        f"{fsel['n_selected_features']}, subset adopted: {fsel['use_feature_selection']}")

                st.markdown("**v2 vs v3, same held-out test set:**")
                st.json(metrics_v3["v2_vs_v3_same_test_set"])

                st.markdown(f"**Test F1 95% bootstrap confidence interval:** "
                            f"[{metrics_v3['test_f1_bootstrap_95ci'][0]:.3f}, "
                            f"{metrics_v3['test_f1_bootstrap_95ci'][1]:.3f}] - the held-out test "
                            "set has only 294 rows, so point-estimate differences of a few "
                            "percentage points between models are not necessarily meaningful.")
