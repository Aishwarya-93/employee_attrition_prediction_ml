import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #e0c3fc, #8ec5fc);
}

/* Titles */
h1 {
    color: #4c1d95;
    text-align: center;
}

h2, h3 {
    color: #312e81;
}

/* FIX LABEL TEXT */
.stSlider label,
.stNumberInput label,
.stSelectbox label {
    color: #1e1e2f !important;
    font-weight: 500;
}

/* Slider value */
.stSlider span {
    color: #1e1e2f !important;
}

/* KPI Cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    padding: 18px;
    border-radius: 14px;
    color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

/* KPI Text */
[data-testid="stMetricLabel"] {
    color: #e0e7ff !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 28px;
    font-weight: bold;
}

/* Inputs */
.stTextInput, .stNumberInput, .stSelectbox, .stSlider {
    background-color: white;
    border-radius: 10px;
    padding: 6px;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    opacity: 0.9;
}
/* Dashboard Header */
h2 {
    color: black !important;
}

/* KPI Titles */
[data-testid="stMetricLabel"] {
    color: black !important;
}

/* KPI Values */
[data-testid="stMetricValue"] {
    color: black !important;
}
/* Success box */
[data-testid="stSuccess"] {
    background-color: #dcfce7 !important;
}

[data-testid="stSuccess"] p {
    color: #14532d !important;
    font-weight: 700;
}

/* Warning box */
[data-testid="stWarning"] {
    background-color: #fef9c3 !important;
}

[data-testid="stWarning"] p {
    color: #854d0e !important;
    font-weight: 700;
}

/* Error box */
[data-testid="stError"] {
    background-color: #fee2e2 !important;
}

[data-testid="stError"] p {
    color: #7f1d1d !important;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


model = joblib.load("attrition_model.pkl")
df = pd.read_csv("employee_attrition_dataset.csv")

model_features = [
'Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','Gender',
'HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate',
'NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction',
'StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany',
'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','Non-Travel','Travel_Frequently',
'Travel_Rarely','Department_Human Resources','Department_Research & Development','Department_Sales',
'Education_Human Resources','Education_Life Sciences','Education_Marketing','Education_Medical',
'Education_Other','Education_Technical Degree','Role_Healthcare Representative','Role_Human Resources',
'Role_Laboratory Technician','Role_Manager','Role_Manufacturing Director','Role_Research Director',
'Role_Research Scientist','Role_Sales Executive','Role_Sales Representative','Status_Divorced',
'Status_Married','Status_Single'
]

tab1, tab2 = st.tabs(["Prediction", "Dashboard"])

with tab1:

    st.title("Employee Attrition Prediction System")

    age = st.slider("Age", 18, 60, 30)
    daily_rate = st.number_input("DailyRate", 100, 2000, 800)
    distance = st.number_input("Distance From Home", 1, 50, 5)
    education = st.slider("Education Level (1-5)", 1, 5, 3)
    env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hourly_rate = st.number_input("HourlyRate", 10, 200, 60)
    job_involve = st.slider("Job Involvement (1-4)", 1, 4, 3)
    job_level = st.slider("Job Level (1-5)", 1, 5, 2)
    job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 20000)
    monthly_rate = st.number_input("Monthly Rate", 1000, 30000, 10000)
    num_comp = st.slider("Num Companies Worked", 0, 10, 1)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    percent_hike = st.number_input("Percent Salary Hike", 10, 35, 15)
    perf = st.slider("Performance Rating (1-4)", 1, 4, 3)
    relation_sat = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
    stock = st.slider("Stock Option Level (0-3)", 0, 3, 1)
    total_years = st.slider("Total Working Years", 0, 40, 5)
    training = st.slider("Training Times Last Year", 0, 10, 2)
    work_balance = st.slider("Work Life Balance (1-4)", 1, 4, 2)
    years_company = st.slider("Years at Company", 0, 40, 3)
    years_role = st.slider("Years in Current Role", 0, 20, 2)
    years_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
    years_manager = st.slider("Years with Current Manager", 0, 17, 2)

    department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
    education_field = st.selectbox("Education Field",
        ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])
    business_travel = st.selectbox("Business Travel",
        ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])
    job_role = st.selectbox("Job Role",
        ["Healthcare Representative","Human Resources","Laboratory Technician","Manager",
         "Manufacturing Director","Research Director","Research Scientist",
         "Sales Executive","Sales Representative"])
    marital = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])

    st.subheader("HR Risk Tolerance Settings")

    risk_tolerance = st.slider("Select HR Risk Tolerance", 1, 3, 2)

    if risk_tolerance == 1:
        low_threshold = 0.40
        high_threshold = 0.65
    elif risk_tolerance == 2:
        low_threshold = 0.30
        high_threshold = 0.55
    else:
        low_threshold = 0.20
        high_threshold = 0.45

    input_df = pd.DataFrame([{
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance,
        "Education": education,
        "EnvironmentSatisfaction": env_sat,
        "Gender": 1 if gender == "Female" else 0,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involve,
        "JobLevel": job_level,
        "JobSatisfaction": job_sat,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_comp,
        "OverTime": 1 if overtime == "Yes" else 0,
        "PercentSalaryHike": percent_hike,
        "PerformanceRating": perf,
        "RelationshipSatisfaction": relation_sat,
        "StockOptionLevel": stock,
        "TotalWorkingYears": total_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": work_balance,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promo,
        "YearsWithCurrManager": years_manager
    }])

    input_df[f"Department_{department}"] = 1
    input_df[f"Education_{education_field}"] = 1
    input_df[f"Role_{job_role}"] = 1
    input_df[f"Status_{marital}"] = 1
    input_df[business_travel] = 1

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_features]

    if st.button("Predict"):
        prob = model.predict_proba(input_df)[0][1]

        if prob < low_threshold:
            st.success(f"Low Risk\nProbability: {prob:.2f}")
        elif prob < high_threshold:
            st.warning(f"Medium Risk\nProbability: {prob:.2f}")
        else:
            st.error(f"High Risk\nProbability: {prob:.2f}")


with tab2:

    st.header("HR Analytics Dashboard")

    df.columns = df.columns.str.replace(" ", "")

    color_scale = ["#6a11cb", "#2575fc"]

    col1, col2, col3 = st.columns(3)

    total_employees = len(df)
    attrition_count = df[df["Attrition"] == "Yes"].shape[0]
    attrition_rate = (attrition_count / total_employees) * 100

    col1.metric("Total Employees", total_employees)
    col2.metric("Employees Left", attrition_count)
    col3.metric("Attrition Rate (%)", f"{attrition_rate:.2f}%")

    colA, colB = st.columns(2)

    with colA:
        fig1 = px.histogram(df, x="Attrition", color="Attrition",
                            color_discrete_sequence=color_scale,
                            title="Attrition Distribution")
        fig1.update_layout(template="plotly_white")
        fig1.update_yaxes(title="Employee Count")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.histogram(df, x="Department", color="Attrition",
                            barmode="group",
                            color_discrete_sequence=color_scale,
                            title="Attrition by Department")
        fig2.update_layout(template="plotly_white")
        fig2.update_yaxes(title="Employee Count")
        st.plotly_chart(fig2, use_container_width=True)
