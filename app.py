import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ======================
# Page Configuration
# ======================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🚨",
    layout="centered"
)

# ======================
# Load model and scaler
# ======================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# Custom Styling (Colorful UI)
# ======================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #F63366;
}
label {
    color: #FAFAFA !important;
}
.stButton>button {
    background-color: #F63366;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
.stProgress > div > div {
    background-color: #F63366;
}
</style>
""", unsafe_allow_html=True)

# ======================
# Title & Description
# ======================
st.title("🚨 Customer Churn Prediction System")
st.markdown(
    "Predict whether a customer is likely to **churn** using a trained **Machine Learning model**. "
    "This tool helps businesses take **proactive retention actions**."
)

st.divider()

# ======================
# User Inputs Section
# ======================
st.subheader("🧾 Customer Information")

credit_score = st.number_input("💳 Credit Score", 300, 900, 650)
age = st.number_input("🎂 Age", 18, 100, 35)
tenure = st.number_input("⏳ Tenure (years)", 0, 10, 5)
balance = st.number_input("🏦 Account Balance", 0.0, 300000.0, 50000.0)
num_products = st.number_input("📦 Number of Products", 1, 4, 2)
has_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active = st.selectbox("⚡ Is Active Member", [0, 1])
salary = st.number_input("💰 Estimated Salary", 0.0, 200000.0, 50000.0)

gender = st.selectbox("🧑 Gender", ["Female", "Male"])
geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"])

# ======================
# Encoding (MATCH TRAINING)
# ======================
gender_male = 1 if gender == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
# France → both 0

# ======================
# Input Array (ORDER IS CRITICAL)
# ======================
input_data = np.array([[
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    has_card,
    is_active,
    salary,
    gender_male,
    geo_germany,
    geo_spain
]])

input_data = scaler.transform(input_data)

st.divider()

# ======================
# Prediction Section
# ======================
if st.button("🔮 Predict Churn"):
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Churn Probability")
    st.progress(int(probability * 100))
    st.metric("Churn Probability", f"{probability*100:.2f}%")

    st.subheader("🚦 Risk Assessment")

    if probability > 0.7:
        st.error("🔴 **High Risk of Churn**")
        st.subheader("💡 Recommended Actions")
        st.write("""
        - Offer personalized retention incentives  
        - Assign a relationship manager  
        - Monitor customer engagement closely  
        """)
    elif probability > 0.4:
        st.warning("🟠 **Medium Risk of Churn**")
        st.subheader("💡 Recommended Actions")
        st.write("""
        - Provide loyalty benefits  
        - Send engagement reminders  
        - Promote relevant add-on services  
        """)
    else:
        st.success("🟢 **Low Risk of Churn**")
        st.subheader("💡 Recommended Actions")
        st.write("""
        - Maintain regular engagement  
        - Encourage premium product adoption  
        """)

    st.caption("⚠️ Predictions are probabilistic and intended for decision support.")

# ======================
# Feature Importance (Optional)
# ======================
if st.checkbox("📌 Show Feature Importance"):
    feature_names = [
        "Credit Score", "Age", "Tenure", "Balance",
        "Num Products", "Has Card", "Active Member",
        "Salary", "Gender_Male", "Geo_Germany", "Geo_Spain"
    ]

    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi_df.set_index("Feature"))
