import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained XGBoost model
model = joblib.load("attrition_model.pkl")

# Expected model feature order:
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

st.title("Employee Attrition Prediction System")

# Inputs
age = st.slider("Age", 18, 60, 30)
daily_rate = st.number_input("DailyRate", min_value=100, max_value=2000, value=800)
distance = st.number_input("Distance From Home", 1, 50, 5)
education = st.slider("Education Level (1-5)", 1, 5, 3)
env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
gender = st.selectbox("Gender", ["Male", "Female"])
hourly_rate = st.number_input("HourlyRate", min_value=10, max_value=200, value=60)
job_involve = st.slider("Job Involvement (1-4)", 1, 4, 3)
with st.expander("What is Job Level?"):
    st.write("""
    | Level | Meaning | Example Roles |
    |------:|---------|---------------|
    | 1 | Entry-level employee | Technician, Junior Analyst |
    | 2 | Intermediate | Sales Executive, HR Assistant |
    | 3 | Senior / Experienced | Senior Scientist, Senior Developer |
    | 4 | Managerial | Manager, Team Lead |
    | 5 | Top-Level Management | Director, VP |
    """)

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

department = st.selectbox("Department", 
    ["Human Resources", "Research & Development", "Sales"])

education_field = st.selectbox("Education Field",
    ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"])

business_travel = st.selectbox("Business Travel",
    ["Non-Travel", "Travel_Frequently", "Travel_Rarely"])

job_role = st.selectbox("Job Role",
    ["Healthcare Representative","Human Resources","Laboratory Technician","Manager",
     "Manufacturing Director","Research Director","Research Scientist",
     "Sales Executive","Sales Representative"])

marital = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])


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

if years_company > total_years:
    st.warning("Years at company cannot be greater than total working years.")

if years_role > total_years:
    st.warning("Years in Current Role cannot be greater than total working years.")
    
if years_manager > total_years:
    st.warning("Years with Current Manager cannot be greater than total working years.")

if years_promo > total_years:
    st.warning("Years Since Last Promotion cannot be greater than total working years.")

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠ High Attrition Risk! Probability: {prob:.2f}")
    else:
        st.success(f"✔ Low Attrition Risk. Probability: {prob:.2f}")
