import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained model & encoders
# -------------------------------
model = joblib.load("extra_trees_credit_model.pkl")

encoders = {
    "Sex": joblib.load("Sex_encoder.pkl"),
    "Housing": joblib.load("Housing_encoder.pkl"),
    "Saving accounts": joblib.load("Saving accounts_encoder.pkl"),
    "Checking account": joblib.load("Checking account_encoder.pkl")
}

# EXACT feature names used during training
feature_names = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration"
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.write("Enter applicant details to predict **credit risk**")

# -------------------------------
# User Inputs
# -------------------------------
age = st.number_input("Age", min_value=18, max_value=80, value=30)

sex = st.selectbox("Sex", ["male", "female"])

job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1)

housing = st.selectbox("Housing", ["own", "rent", "free"])

saving_accounts = st.selectbox(
    "Saving accounts",
    ["little", "moderate", "rich", "quite rich"]
)

checking_account = st.selectbox(
    "Checking account",
    ["little", "moderate", "rich"]
)

credit_amount = st.number_input(
    "Credit amount",
    min_value=0,
    value=1000
)

duration = st.number_input(
    "Duration (months)",
    min_value=1,
    value=12
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Credit Risk"):

    # Build input dataframe EXACTLY like training
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": encoders["Sex"].transform([sex])[0],
        "Job": job,
        "Housing": encoders["Housing"].transform([housing])[0],
        "Saving accounts": encoders["Saving accounts"].transform([saving_accounts])[0],
        "Checking account": encoders["Checking account"].transform([checking_account])[0],
        "Credit amount": credit_amount,
        "Duration": duration
    }])

    # Ensure correct column order
    input_df = input_df[feature_names]

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    # -------------------------------
    # Output
    # -------------------------------
    if prediction == 1:
        st.success(f"✅ Credit Risk: **GOOD**   Probability: **{probability:.2%}**")
    else:
        st.error(f"❌ Credit Risk: **BAD**  Probability: **{probability:.2%}**")
