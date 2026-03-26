import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #e0c3fc, #8ec5fc);
}

h1 {
    color: #4c1d95;
    text-align: center;
}

h2, h3 {
    color: #312e81;
}

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

.stSlider label,
.stNumberInput label,
.stSelectbox label {
    color: #1e1e2f !important;
}

.stSlider span {
    color: #1e1e2f !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: bold;
    border: none;
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

# -------------------- TAB 1 --------------------
with tab1:

    st.title("Employee Attrition Prediction System")

    age = st.slider("Age", 18, 60, 30)
    daily_rate = st.number_input("DailyRate", 100, 2000, 800)
    distance = st.number_input("Distance From Home", 1, 50, 5)
    education = st.slider("Education Level (1-5)", 1, 5, 3)
    with st.expander("What do Education Levels mean?"):
        st.write("""
        1 - Below College  
        2 - College  
        3 - Bachelor  
        4 - Master  
        5 - Doctor  
        """)
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

    job_role_level_map = {
        "Laboratory Technician": 1,
        "Sales Representative": 1,
        "Healthcare Representative": 2,
        "Human Resources": 2,
        "Sales Executive": 2,
        "Research Scientist": 3,
        "Manufacturing Director": 3,
        "Manager": 4,
        "Research Director": 4,
    }

    job_level = job_role_level_map.get(job_role, 2)

    st.write(f"Suggested Job Level based on role: {job_level}")

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
    # -------------------- VALIDATIONS --------------------

    if years_company > total_years:
        st.warning("Years at Company cannot be greater than Total Working Years")

    if years_role > years_company:
        st.warning("Years in Current Role cannot exceed Years at Company")

    if years_manager > years_company:
        st.warning("Years with Manager cannot exceed Years at Company")

    if years_promo > years_company:
        st.warning("Years Since Last Promotion cannot exceed Years at Company")

    if training > total_years:
        st.warning("Training Times Last Year seems unrealistic compared to experience")
        
    if training > years_company:
        st.warning("Training Times Last Year seems unrealistic compared to experience")
    if st.button("Predict"):

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

        prob = model.predict_proba(input_df)[0][1]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Attrition Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#6a11cb"},
                'steps': [
                    {'range': [0, low_threshold*100], 'color': "#dcfce7"},
                    {'range': [low_threshold*100, high_threshold*100], 'color': "#fef9c3"},
                    {'range': [high_threshold*100, 100], 'color': "#fee2e2"}
                ],
            }
        ))

        fig.update_layout(height=300)

        st.plotly_chart(fig, use_container_width=True)

        if prob < low_threshold:
            st.markdown(f"<div style='background:#dcfce7;padding:15px;border-radius:10px;border-left:6px solid #22c55e;color:#14532d;font-weight:600;'>Low Risk Probability: {prob:.2f}</div>", unsafe_allow_html=True)
        elif prob < high_threshold:
            st.markdown(f"<div style='background:#fef9c3;padding:15px;border-radius:10px;border-left:6px solid #eab308;color:#854d0e;font-weight:600;'>Medium Risk Probability: {prob:.2f}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#fee2e2;padding:15px;border-radius:10px;border-left:6px solid #ef4444;color:#7f1d1d;font-weight:600;'>High Risk Probability: {prob:.2f}</div>", unsafe_allow_html=True)

# -------------------- TAB 2 --------------------
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
