# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and columns
model = joblib.load("loan_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🏦 Loan Approval Prediction")
st.write("Using ID3 (Entropy-based Decision Tree)")

# Collect user input
def get_user_input():
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])

    # Convert to model input format
    input_dict = {
        "ApplicantIncome": ApplicantIncome,
        "LoanAmount": LoanAmount,
        "Credit_History": Credit_History,
        "Gender_Male": 1 if Gender == "Male" else 0,
        "Married_Yes": 1 if Married == "Yes" else 0,
        "Education_Not Graduate": 1 if Education == "Not Graduate" else 0
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]
    return input_df

input_data = get_user_input()

# Predict
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader(f"Loan Status: {result}")
