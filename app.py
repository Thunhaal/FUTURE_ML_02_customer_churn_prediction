import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("🚨 Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn using Machine Learning.")

# User inputs
credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure (years)", 0, 10, 5)
balance = st.number_input("Account Balance", 0.0, 300000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 2)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# Input array
input_data = np.array([[credit_score, age, tenure, balance,
                         num_products, has_card, is_active, salary]])

input_data = scaler.transform(input_data)

# Prediction
if st.button("Predict Churn"):
    probability = model.predict_proba(input_data)[0][1]

    if probability > 0.7:
        st.error(f"🔴 High Risk of Churn ({probability:.2f})")
    elif probability > 0.4:
        st.warning(f"🟠 Medium Risk of Churn ({probability:.2f})")
    else:
        st.success(f"🟢 Low Risk of Churn ({probability:.2f})")
